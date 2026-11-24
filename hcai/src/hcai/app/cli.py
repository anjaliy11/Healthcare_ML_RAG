"""Command-line interface: `python -m hcai <command>`. The only place that prints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from hcai import DISCLAIMER
from hcai.app.service import Services, make_embedder
from hcai.config import Config, load_config
from hcai.errors import HcaiError
from hcai.logging import configure_logging
from hcai.ml.schema import CLINICAL_TEST_FIELDS, FEATURES, PatientInput
from hcai.rag.embeddings import HashingEmbedder
from hcai.rag.pipeline import Answer, build_index
from hcai.rag.rerank import CrossEncoderReranker

EXAMPLE_PATIENT = {
    "age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 250, "fbs": 0, "restecg": 0,
    "thalach": 120, "exang": 1, "oldpeak": 2.0, "slope": 2, "ca": 1, "thal": 7,
}  # fmt: skip


def _out(msg: str = "") -> None:
    sys.stdout.write(msg + "\n")


def format_answer(ans: Answer) -> str:
    lines = [ans.text]
    if ans.sources:
        lines += ["", "Sources:"]
        lines += [
            f"  [{s.cite}] {s.title} — {s.section} (score {s.score:.2f})\n       {s.url}  · {s.license}"
            for s in ans.sources
        ]
    if ans.warnings:
        lines += ["", f"(notes: {', '.join(ans.warnings)})"]
    lines += [
        "",
        f"[{ans.mode} · {ans.latency_ms:.0f} ms · request {ans.request_id}]",
        "",
        ans.disclaimer,
    ]
    return "\n".join(lines)


def cmd_data(cfg: Config, args: argparse.Namespace) -> int:
    from hcai.ml.data import load_dataset

    x, y, sha = load_dataset(cfg, args.source_file)
    _out(f"rows={len(x)} positives={int(y.sum())} ({y.mean():.1%}) sha256={sha[:12]}")
    _out(
        "missing cells: "
        + (json.dumps({c: int(n) for c, n in x.isna().sum().items() if n}) or "none")
    )
    return 0


def cmd_train(cfg: Config, args: argparse.Namespace) -> int:
    from hcai.ml.train import train

    res = train(cfg, args.source_file)
    t = res.report["test"]
    _out(f"selected: {res.selected}  (version {res.metadata['model_version']})")
    for name, rep in t.items():
        m, ci = rep["metrics"], rep["ci95"]["roc_auc"]
        _out(f"  {name:14s} acc {m['accuracy']:.3f}  prec {m['precision']:.3f}  rec {m['recall']:.3f}"
             f"  f1 {m['f1']:.3f}  auc {m['roc_auc']:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]")  # fmt: skip
    _out("")
    _out(res.report["explanation"])
    _out(f"\nReports: {cfg.paths.reports_dir}")
    if not res.report["sanity"]["floor_ok"]:
        _out(f"WARNING: selected model test ROC-AUC is below the {cfg.ml.auc_floor} floor.")
    return 0


def cmd_index(cfg: Config, args: argparse.Namespace) -> int:
    embedder = HashingEmbedder() if args.offline else make_embedder(cfg)
    if args.offline:
        _out("OFFLINE hashing embedder: lexical demo only. Use the default for real retrieval.")
    store = build_index(cfg, embedder)
    _out(f"indexed {len(store)} chunks with {embedder.name} -> {cfg.paths.index_dir}")
    return 0


def cmd_ask(cfg: Config, args: argparse.Namespace) -> int:
    _out(format_answer(Services(cfg, use_llm=not args.no_llm).ask(args.question)))
    return 0


def _patient_from_args(args: argparse.Namespace) -> dict[str, object]:
    if args.input:
        return json.loads(Path(args.input).read_text(encoding="utf-8"))
    data = {f: getattr(args, f) for f in FEATURES if getattr(args, f) is not None}
    return data


def cmd_assess(cfg: Config, args: argparse.Namespace) -> int:
    if args.example:
        _out(json.dumps(EXAMPLE_PATIENT, indent=2))
        return 0
    svc = Services(cfg, use_llm=not args.no_llm)
    res = svc.assess(_patient_from_args(args))
    r = res.risk
    _out(f"Estimated risk: {r.probability:.0%} ({r.band} band; demo cut-points, not clinical)")
    _out(f"Model: {r.model_version}")
    if r.top_factors:
        _out("Factors raising the estimate vs. a typical patient:")
        for f in r.top_factors:
            _out(f"  - {f.label} = {f.patient_value:g} (+{f.delta:.0%})")
    for n in res.notes:
        _out(f"Note: {n}")
    for e in res.explanations:
        _out(f"\n=== {e.factor.label} ===")
        _out(format_answer(e.answer) if e.answer else e.note)
    _out(f"\n{res.disclaimer}")
    return 0


def cmd_eval(cfg: Config, args: argparse.Namespace) -> int:
    from hcai.eval.rag_eval import load_golden, run_ablation, run_eval, write_report
    from hcai.monitoring.metrics import NullMetrics

    svc = Services(cfg, use_llm=not args.no_llm)
    pipe = svc.pipeline
    pipe.metrics = NullMetrics()
    items = load_golden()
    result = run_eval(pipe, items)
    reranker = CrossEncoderReranker(cfg.rag.reranker_model) if args.rerank else None
    ablation = run_ablation(pipe.retriever.embedder, pipe.retriever.store, items, cfg.rag, reranker)
    path = write_report(
        result,
        cfg.paths.reports_dir,
        pipe.retriever.embedder.name,
        pipe.generator.llm is not None,
        ablation,
    )
    _out(path.read_text(encoding="utf-8"))
    return 0


def cmd_monitor(cfg: Config, args: argparse.Namespace) -> int:
    from hcai.monitoring.drift import check_drift
    from hcai.monitoring.metrics import read_records, summarize

    _out(json.dumps(summarize(read_records(cfg.paths.metrics_log)), indent=2))
    if args.drift_csv:
        ref = pd.read_csv(cfg.paths.reference)
        _out(json.dumps(check_drift(ref, pd.read_csv(args.drift_csv)), indent=2))
    return 0


def cmd_status(cfg: Config, args: argparse.Namespace) -> int:
    checks = {
        "dataset cache": (cfg.paths.data_raw / "heart_cleveland.csv").is_file(),
        "document manifest": (cfg.paths.docs_dir / "manifest.json").is_file(),
        "trained model": cfg.paths.model.is_file(),
        "search index": (cfg.paths.index_dir / "index.faiss").is_file(),
        "LLM key set (GEMINI_API_KEY)": bool(
            os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        ),
    }
    for k, ok in checks.items():
        _out(f"[{'x' if ok else ' '}] {k}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="hcai", description="Healthcare ML + RAG prototype. " + DISCLAIMER
    )
    ap.add_argument("--config", type=Path, default=None)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data", help="download/cache and validate the UCI Cleveland dataset")
    p.add_argument("--source-file", type=Path, help="local processed.cleveland.data or CSV")
    p = sub.add_parser("train", help="train, evaluate and save the risk model")
    p.add_argument("--source-file", type=Path)
    p = sub.add_parser("index", help="chunk, embed and index the document corpus")
    p.add_argument("--offline", action="store_true", help="hashing embedder (no model download)")
    p = sub.add_parser("ask", help="ask a question answered from the documents")
    p.add_argument("question")
    p.add_argument("--no-llm", action="store_true", help="force extractive answers")
    p = sub.add_parser("assess", help="risk estimate + document-based explanation for one patient")
    p.add_argument("--input", type=Path, help="JSON file with the 13 fields")
    p.add_argument("--example", action="store_true", help="print an example input JSON")
    p.add_argument("--no-llm", action="store_true")
    for f in FEATURES:
        note = " (needs a clinical test)" if f in CLINICAL_TEST_FIELDS else ""
        p.add_argument(f"--{f}", type=float if f in {"trestbps", "chol", "thalach", "oldpeak"} else int,
                       help=(PatientInput.model_fields[f].description or "") + note)  # fmt: skip
    p = sub.add_parser("eval", help="evaluate retrieval/refusal on the golden set")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument(
        "--rerank", action="store_true", help="add a cross-encoder rerank row (downloads model)"
    )
    p = sub.add_parser("monitor", help="summarise request metrics; optional drift check")
    p.add_argument("--drift-csv", type=Path, help="CSV of incoming patient rows to compare")
    sub.add_parser("status", help="show which artifacts exist")
    return ap


COMMANDS = {
    "data": cmd_data, "train": cmd_train, "index": cmd_index, "ask": cmd_ask,
    "assess": cmd_assess, "eval": cmd_eval, "monitor": cmd_monitor, "status": cmd_status,
}  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(os.environ.get("HCAI_LOG_LEVEL", "WARNING"))
    try:
        return COMMANDS[args.command](load_config(args.config), args)
    except HcaiError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2
