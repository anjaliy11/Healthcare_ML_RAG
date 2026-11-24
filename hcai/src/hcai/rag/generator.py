"""Grounded answer generation: LLM with citation validation, or a no-LLM extractive fallback."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from hcai.config import LLMConfig
from hcai.errors import LLMError
from hcai.logging import get_logger, log_event
from hcai.rag.embeddings import tokenize
from hcai.rag.prompts import NOT_IN_DOCUMENTS, SYSTEM_PROMPT, build_user_prompt
from hcai.rag.retriever import Hit

log = get_logger("rag.generator")
# One bracket group may hold several tags: "[C1]", "[C1, C2]", "[C1][C2]", "[C1; C3]".
_GROUP_RE = re.compile(r"\[\s*C\d+(?:\s*[,;/&]\s*(?:and\s+)?C\d+|\s+and\s+C\d+)*\s*\]", re.I)
_TAG_RE = re.compile(r"C(\d+)", re.I)
_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class LLMReply:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> LLMReply: ...


class AnthropicClient:
    """Thin wrapper over the Anthropic Messages API with retry + exponential backoff."""

    def __init__(
        self,
        cfg: LLMConfig,
        api_key: str | None = None,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg, self._sleep = cfg, sleep
        if client is None:
            import anthropic

            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise LLMError("ANTHROPIC_API_KEY is not set.")
            client = anthropic.Anthropic(api_key=key, timeout=cfg.timeout_s, max_retries=0)
        self.client = client

    def _retryable(self, exc: Exception) -> bool:
        import anthropic

        return isinstance(
            exc,
            anthropic.RateLimitError
            | anthropic.APITimeoutError
            | anthropic.APIConnectionError
            | anthropic.InternalServerError,
        )

    def complete(self, system: str, user: str) -> LLMReply:
        last: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                msg = self.client.messages.create(
                    model=self.cfg.model,
                    max_tokens=self.cfg.max_tokens,
                    temperature=self.cfg.temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
                if not text.strip():
                    raise LLMError("LLM returned an empty reply.")
                return LLMReply(text, msg.usage.input_tokens, msg.usage.output_tokens)
            except LLMError:
                raise
            except Exception as exc:  # SDK errors: retry only the transient ones
                last = exc
                if not self._retryable(exc) or attempt == self.cfg.max_retries:
                    break
                self._sleep(self.cfg.backoff_s * 2**attempt)
        raise LLMError(f"LLM call failed ({type(last).__name__}).") from last


def parse_retry_delay(exc: Exception) -> float | None:
    """Seconds the API asked us to wait (RetryInfo.retryDelay "7s", or "retry in 7.2s" text)."""
    details = getattr(exc, "details", None)
    err = details.get("error", details) if isinstance(details, dict) else {}
    for d in (err.get("details") or []) if isinstance(err, dict) else []:
        delay = d.get("retryDelay") if isinstance(d, dict) else None
        if isinstance(delay, str):
            m = re.fullmatch(r"\s*([\d.]+)\s*s\s*", delay)
            if m:
                return float(m.group(1))
    m2 = re.search(r"retry in ([\d.]+)\s*s", str(exc), re.I)
    return float(m2.group(1)) if m2 else None


class GeminiClient:
    """Google Gemini (free tier) via the google-genai SDK.

    Free-tier quotas are per model and unpublished. Latency is bounded by ONE deadline shared by
    every model and retry (the first real run took 70 s because per-call 30 s timeouts chained):
    - a timeout is not retried on the same model, we move on;
    - a 429 waits for the server's retryDelay only if it is short, else we move to the next model;
    - no attempt starts if the remaining budget is too small to be useful.
    Only retrieved passages and the question are sent; patient values never are."""

    MIN_ATTEMPT_S = 3.0  # an attempt with less budget than this cannot finish; skip it

    def __init__(
        self,
        cfg: LLMConfig,
        api_key: str | None = None,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg, self._sleep, self._clock = cfg, sleep, clock
        if client is None:
            from google import genai

            key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise LLMError("GEMINI_API_KEY is not set.")
            client = genai.Client(api_key=key)
        self.client = client
        self.models = [cfg.model, *cfg.fallback_models]
        self._last_call: float | None = None

    @staticmethod
    def _code(exc: Exception) -> int | None:
        code = getattr(exc, "code", None)
        return code if isinstance(code, int) else None

    @staticmethod
    def _is_timeout(exc: Exception) -> bool:
        name = type(exc).__name__.lower()
        return "timeout" in name or "timedout" in name or isinstance(exc, TimeoutError)

    def _config(self, system: str, timeout_s: float) -> Any:
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.cfg.temperature,
            max_output_tokens=self.cfg.max_tokens,
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
        )

    def _pace(self, deadline: float) -> None:
        """Keep under the free-tier requests-per-minute limit when called in a burst (eval)."""
        if self._last_call is not None and self.cfg.min_interval_s > 0:
            wait = self.cfg.min_interval_s - (self._clock() - self._last_call)
            if 0 < wait < deadline - self._clock():
                self._sleep(wait)
        self._last_call = self._clock()

    def complete(self, system: str, user: str) -> LLMReply:
        deadline = self._clock() + self.cfg.deadline_s
        last: Exception | None = None
        for model in self.models:
            for attempt in range(self.cfg.max_retries + 1):
                remaining = deadline - self._clock()
                if remaining < self.MIN_ATTEMPT_S:
                    raise LLMError(
                        f"Gemini did not answer within {self.cfg.deadline_s:.0f}s "
                        f"({type(last).__name__ if last else 'no attempt'})."
                    ) from last
                self._pace(deadline)
                try:
                    resp = self.client.models.generate_content(
                        model=model,
                        contents=user,
                        config=self._config(system, min(self.cfg.timeout_s, remaining)),
                    )
                    text = (getattr(resp, "text", None) or "").strip()
                    if not text:  # blocked by safety filters, or budget spent on reasoning
                        raise LLMError(f"{model} returned an empty reply.")
                    usage = getattr(resp, "usage_metadata", None)
                    return LLMReply(
                        text,
                        int(getattr(usage, "prompt_token_count", 0) or 0),
                        int(getattr(usage, "candidates_token_count", 0) or 0),
                    )
                except LLMError as exc:
                    last = exc
                    break  # empty reply: do not retry the same model
                except Exception as exc:
                    last = exc
                    if self._is_timeout(exc):
                        break  # a slow model will be slow again: next model
                    code = self._code(exc)
                    if code not in {429, 500, 502, 503, 504} or attempt == self.cfg.max_retries:
                        break  # not found, bad request, or retries spent: next model
                    delay = parse_retry_delay(exc) if code == 429 else None
                    wait = self.cfg.backoff_s * 2**attempt if delay is None else delay + 0.25
                    if wait > self.cfg.max_wait_s or wait >= deadline - self._clock():
                        break  # quota window too long to sit out: try the next model instead
                    self._sleep(wait)
        raise LLMError(f"Gemini call failed ({type(last).__name__}).") from last


def _rewrite_group(group: str, n_hits: int) -> str:
    """Keep only the valid tags of one bracket group; drop the group if none are valid."""
    keep = [f"C{n}" for n in map(int, _TAG_RE.findall(group)) if 1 <= n <= n_hits]
    return f"[{', '.join(keep)}]" if keep else ""


@dataclass
class GenResult:
    text: str
    mode: str  # "llm" | "extractive" | "refusal"
    cited: list[int] = field(default_factory=list)  # 1-based indices into the hits passed in
    input_tokens: int = 0
    output_tokens: int = 0
    warnings: list[str] = field(default_factory=list)


def extractive_answer(question: str, hits: list[Hit], n_sentences: int) -> GenResult:
    """No-LLM answer: the passage sentences that best overlap the question, verbatim, cited."""
    q_tokens = set(tokenize(question))
    scored: list[tuple[int, int, int, str]] = []  # (-overlap, hit idx, sent idx, text)
    for h_idx, h in enumerate(hits, start=1):
        for s_idx, sent in enumerate(_SENT_RE.split(h.chunk.text)):
            sent = sent.strip(" -•*")
            if len(sent.split()) < 4:
                continue
            overlap = len(q_tokens & set(tokenize(sent)))
            scored.append((-overlap, h_idx, s_idx, sent))
    scored.sort()
    chosen = sorted(scored[:n_sentences], key=lambda t: (t[1], t[2]))
    if not chosen:
        return GenResult("", "refusal")
    lines = [f"- {s} [C{h}]" for _, h, _, s in chosen]
    text = (
        "Relevant passages from the documents (extractive, no AI-written summary):\n"
        + "\n".join(lines)
    )
    return GenResult(text, "extractive", sorted({h for _, h, _, _ in chosen}))


class Generator:
    def __init__(self, llm: LLMClient | None, extractive_sentences: int = 3) -> None:
        self.llm, self.n_sent = llm, extractive_sentences

    def _fallback(
        self, question: str, hits: list[Hit], reason: str, base: GenResult | None = None
    ) -> GenResult:
        res = extractive_answer(question, hits, self.n_sent)
        res.warnings.append(f"llm_fallback:{reason}")
        if base:
            res.input_tokens, res.output_tokens = base.input_tokens, base.output_tokens
            res.warnings.extend(base.warnings)
        return res

    def generate(self, question: str, hits: list[Hit]) -> GenResult:
        if self.llm is None:
            return extractive_answer(question, hits, self.n_sent)
        try:
            reply = self.llm.complete(SYSTEM_PROMPT, build_user_prompt(question, hits))
        except LLMError as exc:
            log_event(log, "llm_failed", error=str(exc)[:80])
            return self._fallback(question, hits, "llm_error")
        text = reply.text.strip()
        base = GenResult(
            "", "llm", input_tokens=reply.input_tokens, output_tokens=reply.output_tokens
        )
        if NOT_IN_DOCUMENTS in text:
            base.mode = "refusal"
            return base
        found = [int(n) for g in _GROUP_RE.findall(text) for n in _TAG_RE.findall(g)]
        valid = {n for n in found if 1 <= n <= len(hits)}
        invalid = [n for n in found if n not in valid]
        if invalid:  # fabricated source tags are stripped, never shown
            text = _GROUP_RE.sub(lambda m: _rewrite_group(m.group(0), len(hits)), text)
            text = re.sub(r"[ \t]{2,}", " ", text)
            text = re.sub(r"\s+([.,;:!?])", r"\1", text).strip()
            base.warnings.append(f"invalid_citations_stripped:{len(invalid)}")
            log_event(log, "invalid_citations_stripped", count=len(invalid))
        if not valid:  # never show uncited prose
            return self._fallback(question, hits, "no_valid_citation", base)
        base.text, base.cited = text, sorted(valid)
        return base
