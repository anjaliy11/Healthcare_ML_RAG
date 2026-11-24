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
from hcai.errors import HcaiError, PatientInputError
from hcai.logging import configure_logging
from hcai.ml.datasets import DATASETS, DEFAULT_DATASET, config_for
from hcai.ml.schema import CLINICAL_TEST_FIELDS, FEATURES, PatientInput
from hcai.rag.embeddings import HashingEmbedder
from hcai.rag.pipeline import Answer, build_index
from hcai.rag.rerank import CrossEncoderReranker
from hcai.textio import read_text

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
    from hcai.ml.datasets import INFO
    from hcai.ml.train import train

    res = train(cfg, args.source_file, dataset=args.dataset, tag=args.tag)
    t = res.report["test"]
    _out(f"dataset: {INFO[args.dataset].title}")
    _out(f"selected: {res.selected}  (version {res.metadata['model_version']}, run {res.run_id})")
    for name, rep in t.items():
        m, ci = rep["metrics"], rep["ci95"]["roc_auc"]
        _out(f"  {name:14s} acc {m['accuracy']:.3f}  prec {m['precision']:.3f}  rec {m['recall']:.3f}"
             f"  f1 {m['f1']:.3f}  auc {m['roc_auc']:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]")  # fmt: skip
    _out("")
    _out(res.report["explanation"])
    san, lk = res.report["sanity"], res.report["leakage"]
    _out(
        f"\nLeakage checks: {san['verdict'].upper()} · duplicates {lk['duplicates']['n_overlap']} · "
        f"strongest single feature {lk['single_feature_auc_max']['feature']} "
        f"({lk['single_feature_auc_max']['auc']:.2f}) · label-shuffle AUC "
        f"{lk['label_shuffle']['mean']:.2f} (expect ~0.5)"
    )
    for r in san["reasons"]:
        _out(f"NOTE: {r}")
    _out(f"\nReports: {cfg.paths.reports_dir}")
    if not san["floor_ok"]:
        _out(f"WARNING: selected model test ROC-AUC is below the {cfg.ml.auc_floor} floor.")
    return 0


def cmd_stability(cfg: Config, args: argparse.Namespace) -> int:
    from hcai.ml.datasets import load_named
    from hcai.ml.leakage import split_stability, stability_markdown

    x, y, _ = load_named(cfg, args.dataset, args.source_file)
    n = args.splits or cfg.ml.stability_splits
    _out(f"running {n} splits on {args.dataset} (about {max(1, n // 3)} s)...")
    res = split_stability(cfg, x, y, n)
    out = cfg.paths.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "stability.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    md = stability_markdown(res, args.dataset)
    (out / "stability.md").write_text(md, encoding="utf-8")
    _out(md)
    return 0


def cmd_runs(cfg: Config, args: argparse.Namespace) -> int:
    from hcai.ml.runs import import_legacy, list_runs

    if args.import_metrics:
        rid = import_legacy(
            cfg, args.import_metrics, args.dataset, args.tag, args.build or "imported"
        )
        _out(f"imported as run {rid}")
        return 0
    runs = list_runs(cfg)
    if not runs:
        _out("no runs yet. Train first: python -m hcai train")
        return 0
    _out(f"{'run id':52s} {'dataset':22s} {'build':8s} {'model':14s} test AUC [95% CI]   verdict")
    for r in runs:
        m = r["test"].get(r["selected"] or "", {})
        auc = (
            f"{m['roc_auc']:.3f} [{m['roc_auc_ci95'][0]:.2f}, {m['roc_auc_ci95'][1]:.2f}]"
            if m
            else "n/a"
        )
        _out(f"{r['run_id']:52s} {r['dataset']:22s} {r['build']:8s} {r['selected'] or '-':14s} {auc:20s} {r['verdict']}")  # fmt: skip
    return 0


def cmd_report(cfg: Config, args: argparse.Namespace) -> int:
    from hcai.report import publish

    readme, copied = publish(cfg, args.readme)
    _out(f"updated results block in {readme}")
    _out(f"copied {len(copied)} report files to {readme.parent / 'docs' / 'results'}")
    return 0


def cmd_compare(cfg: Config, args: argparse.Namespace) -> int:
    from hcai.ml.runs import resolve_run, write_comparison

    a, b = resolve_run(cfg, args.run_a), resolve_run(cfg, args.run_b)
    path = write_comparison(cfg, a, b)
    _out(path.read_text(encoding="utf-8"))
    _out(f"saved: {path}")
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
        path = Path(args.input)
        if not path.is_file():
            raise PatientInputError(f"Input file not found: {path}")
        try:
            data = json.loads(read_text(path))
        except json.JSONDecodeError as exc:
            raise PatientInputError(
                f"{path} is not valid JSON ({exc.msg}, line {exc.lineno})."
            ) from exc
        if not isinstance(data, dict):
            raise PatientInputError(f"{path} must contain a JSON object with the 13 fields.")
        return data
    data = {f: getattr(args, f) for f in FEATURES if getattr(args, f) is not None}
    return data


def cmd_assess(cfg: Config, args: argparse.Namespace) -> int:
    if args.example:
        text = json.dumps(EXAMPLE_PATIENT, indent=2)
        if args.out:  # writes UTF-8 itself: avoids PowerShell's UTF-16 `>` redirection
            Path(args.out).write_text(text + "\n", encoding="utf-8")
            _out(f"wrote example patient to {args.out}")
        else:
            _out(text)
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

    if not args.no_llm:  # stay under the free-tier requests-per-minute limit during the burst
        paced = cfg.llm.model_copy(update={"min_interval_s": max(cfg.llm.min_interval_s, 4.0)})
        cfg = cfg.model_copy(update={"llm": paced})
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
    import hcai

    _out(f"hcai {hcai.__version__} loaded from {Path(hcai.__file__).parent}")
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
    p.add_argument(
        "--source-file",
        type=Path,
        help="local data file (Cleveland .data/.csv, or Kaggle heart.csv)",
    )
    p.add_argument(
        "--dataset", choices=DATASETS, default=DEFAULT_DATASET, help="default: cleveland (primary)"
    )
    p.add_argument("--tag", default="", help="label for this run in the archive, e.g. build3")
    p = sub.add_parser("stability", help="how much does test ROC-AUC vary across random splits?")
    p.add_argument("--dataset", choices=DATASETS, default=DEFAULT_DATASET)
    p.add_argument("--source-file", type=Path)
    p.add_argument("--splits", type=int, help="number of random splits (default: config)")
    p = sub.add_parser(
        "runs", help="list archived training runs, or import an earlier metrics.json"
    )
    p.add_argument(
        "--import-metrics", type=Path, help="archive a metrics.json from an earlier build"
    )
    p.add_argument("--dataset", choices=DATASETS, default=DEFAULT_DATASET)
    p.add_argument("--tag", default="imported")
    p.add_argument("--build", help="label for the build that produced it, e.g. 0.2-gemini")
    p.add_argument(
        "--allow-current-build",
        action="store_true",
        help="import even if the file was written by build 0.3+ (normally a mistake)",
    )
    p.add_argument(
        "--assume-same-data",
        action="store_true",
        help="record this machine's cached dataset hash for the imported run (you assert it used the same file)",
    )
    p.add_argument("--source-file", type=Path)
    p = sub.add_parser("report", help="write measured results into README.md and docs/results/")
    p.add_argument("--readme", type=Path, help="default: README.md in the project root")
    p = sub.add_parser("compare", help="compare two archived runs (A = baseline, B = candidate)")
    p.add_argument("run_a")
    p.add_argument("run_b")
    p = sub.add_parser("index", help="chunk, embed and index the document corpus")
    p.add_argument("--offline", action="store_true", help="hashing embedder (no model download)")
    p = sub.add_parser("ask", help="ask a question answered from the documents")
    p.add_argument("question")
    p.add_argument("--no-llm", action="store_true", help="force extractive answers")
    p = sub.add_parser("assess", help="risk estimate + document-based explanation for one patient")
    p.add_argument("--input", type=Path, help="JSON file with the 13 fields")
    p.add_argument("--example", action="store_true", help="print an example input JSON")
    p.add_argument("--out", type=Path, help="with --example: write the JSON to this file (UTF-8)")
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
    "data": cmd_data, "train": cmd_train, "stability": cmd_stability, "runs": cmd_runs,
    "compare": cmd_compare, "report": cmd_report, "index": cmd_index, "ask": cmd_ask,
    "assess": cmd_assess, "eval": cmd_eval, "monitor": cmd_monitor, "status": cmd_status,
}  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(os.environ.get("HCAI_LOG_LEVEL", "WARNING"))
    try:
        cfg = load_config(args.config)
        if args.command in {"train", "stability"}:
            cfg = config_for(cfg, args.dataset)
        return COMMANDS[args.command](cfg, args)
    except HcaiError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2
