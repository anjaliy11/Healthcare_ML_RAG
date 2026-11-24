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
_CITE_RE = re.compile(r"\[C(\d+)\]")
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


class GeminiClient:
    """Google Gemini (free tier) via the google-genai SDK.

    Free-tier quotas are per model and unpublished, so on 429 / 404 / 5xx after retries we move
    to the next model in `fallback_models`. Only the retrieved passages and the question are
    sent; patient values never are (free-tier data may be used by Google to improve products)."""

    def __init__(
        self,
        cfg: LLMConfig,
        api_key: str | None = None,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg, self._sleep = cfg, sleep
        if client is None:
            from google import genai
            from google.genai import types

            key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise LLMError("GEMINI_API_KEY is not set.")
            client = genai.Client(
                api_key=key, http_options=types.HttpOptions(timeout=int(cfg.timeout_s * 1000))
            )
        self.client = client
        self.models = [cfg.model, *cfg.fallback_models]

    @staticmethod
    def _code(exc: Exception) -> int | None:
        code = getattr(exc, "code", None)
        return code if isinstance(code, int) else None

    def _config(self, system: str) -> Any:
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.cfg.temperature,
            max_output_tokens=self.cfg.max_tokens,
        )

    def complete(self, system: str, user: str) -> LLMReply:
        last: Exception | None = None
        for model in self.models:
            for attempt in range(self.cfg.max_retries + 1):
                try:
                    resp = self.client.models.generate_content(
                        model=model, contents=user, config=self._config(system)
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
                    code = self._code(exc)
                    transient = code in {429, 500, 502, 503, 504}
                    if transient and attempt < self.cfg.max_retries:
                        self._sleep(self.cfg.backoff_s * 2**attempt)
                        continue
                    break  # exhausted, not found, bad request: try the next model
        raise LLMError(f"Gemini call failed ({type(last).__name__}).") from last


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
        found = [int(n) for n in _CITE_RE.findall(text)]
        valid = {n for n in found if 1 <= n <= len(hits)}
        invalid = [n for n in found if n not in valid]
        if invalid:  # fabricated source tags are stripped, never shown
            text = _CITE_RE.sub(lambda m: m.group(0) if int(m.group(1)) in valid else "", text)
            text = re.sub(r"[ \t]{2,}", " ", text).strip()
            base.warnings.append(f"invalid_citations_stripped:{len(invalid)}")
            log_event(log, "invalid_citations_stripped", count=len(invalid))
        if not valid:  # never show uncited prose
            return self._fallback(question, hits, "no_valid_citation", base)
        base.text, base.cited = text, sorted(valid)
        return base
