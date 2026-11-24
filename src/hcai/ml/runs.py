"""Run archive and run-to-run comparison.

Every `train` writes a compact, immutable summary under `artifacts/runs/<run_id>/`. Two runs can
then be compared (`python -m hcai compare A B`): old build vs new build on the same data, or one
dataset vs another. Only numbers that were actually produced are tabulated; nothing is invented."""

from __future__ import annotations

import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sklearn

from hcai import __version__
from hcai.config import Config
from hcai.errors import HcaiError
from hcai.textio import read_text

METRICS = ("roc_auc", "accuracy", "precision", "recall", "f1")


class RunNotFoundError(HcaiError):
    pass


def runs_dir(cfg: Config) -> Path:
    return cfg.paths.artifacts / "runs"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "run"


def _legacy_verdict(sanity: dict[str, Any]) -> str:
    """Earlier builds had only the fixed AUC ceiling rule and no `verdict` field."""
    if not sanity:
        return "n/a"
    return "passed (old ceiling rule)" if sanity.get("ceiling_ok") else "aborted (old ceiling rule)"


def summarise(
    report: dict[str, Any],
    *,
    dataset: str,
    tag: str,
    status: str,
    seed: int,
    data_sha: str,
    absent: list[str],
    git_sha: str = "unknown",
    build: str | None = None,
    trained_at: str | None = None,
) -> dict[str, Any]:
    """Reduce a full training report to the fields needed for comparison."""
    test = {}
    for name, rep in report.get("test", {}).items():
        m, ci = rep["metrics"], rep["ci95"]
        test[name] = {
            **{k: m[k] for k in METRICS if k in m},
            "roc_auc_ci95": list(ci["roc_auc"]),
            "recall_ci95": list(ci["recall"]) if "recall" in ci else None,
            "brier": rep.get("brier"),
        }
    sanity = report.get("sanity", {})
    leak = report.get("leakage", {})
    return {
        "dataset": dataset,
        "tag": tag,
        "status": status,
        "build": build or __version__,
        "trained_at": trained_at or datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha,
        "sklearn_version": sklearn.__version__,
        "seed": seed,
        "data_sha256": data_sha,
        "absent_features": absent,
        "n_train": report.get("n_train"),
        "n_test": report.get("n_test"),
        "positive_rate": report.get("class_balance_positive"),
        "selected": report.get("selected"),
        "cv": {k: {"mean": v["mean"], "std": v["std"]} for k, v in report.get("cv", {}).items()},
        "test": test,
        "nested_cv_auc": report.get("nested_cv_auc"),
        "calibration_applied": report.get("calibration", {}).get("applied"),
        "verdict": sanity.get("verdict") or _legacy_verdict(sanity),
        "verdict_reasons": sanity.get("reasons", []),
        "leakage_diagnostics": {
            k: leak.get(k)
            for k in ("duplicates", "single_feature_auc_max", "label_shuffle", "clean")
        }
        if leak
        else None,
    }


def archive_run(cfg: Config, summary: dict[str, Any], files: list[Path]) -> str:
    """Write summary.json (+ copies of the full reports) to a new run folder; return its id."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_id = _slug(
        f"{stamp}_{summary['dataset']}" + (f"_{summary['tag']}" if summary["tag"] else "")
    )
    folder = runs_dir(cfg) / run_id
    n = 1
    while folder.exists():  # two runs inside one second
        n += 1
        folder = runs_dir(cfg) / f"{run_id}-{n}"
    folder.mkdir(parents=True)
    run_id = folder.name
    (folder / "summary.json").write_text(
        json.dumps({"run_id": run_id, **summary}, indent=2), encoding="utf-8"
    )
    for f in files:
        if f.is_file():
            shutil.copyfile(f, folder / f.name)
    return run_id


def import_legacy(
    cfg: Config,
    metrics_json: Path,
    dataset: str,
    tag: str,
    build: str,
    data_sha: str = "unknown",
    allow_current_build: bool = False,
) -> str:
    """Archive a `metrics.json` produced by an earlier build (e.g. the build you already ran)."""
    if not metrics_json.is_file():
        raise RunNotFoundError(f"File not found: {metrics_json}")
    try:
        report = json.loads(read_text(metrics_json))
        if not isinstance(report, dict) or not report.get("test") or not report.get("selected"):
            raise KeyError("no 'selected' / 'test' sections")
        if not allow_current_build and (
            "leakage" in report or isinstance(report.get("dataset"), dict)
        ):
            raise RunNotFoundError(
                f"{metrics_json.name} was written by build 0.3 or later (it contains leakage "
                "checks), not by an older build. Importing it as an older build would compare a "
                "run with itself. Import the metrics.json saved BEFORE you trained with 0.3, or "
                "pass --allow-current-build if you really mean it."
            )
        stamp = datetime.fromtimestamp(metrics_json.stat().st_mtime, UTC).isoformat(
            timespec="seconds"
        )
        summary = summarise(
            report, dataset=dataset, tag=tag, status="imported", seed=int(report.get("seed", 42)),
            data_sha=data_sha, absent=[], build=build, trained_at=stamp,
        )  # fmt: skip
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RunNotFoundError(f"{metrics_json} is not a hcai metrics.json ({exc}).") from exc
    summary["source"] = f"imported from {metrics_json.name}"
    return archive_run(cfg, summary, [metrics_json.parent / "model_report.md"])


def list_runs(cfg: Config) -> list[dict[str, Any]]:
    out = []
    for p in sorted(runs_dir(cfg).glob("*/summary.json")) if runs_dir(cfg).is_dir() else []:
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def resolve_run(cfg: Config, ref: str) -> dict[str, Any]:
    """Run by exact id, or by unique substring of its id (tag, dataset, date)."""
    runs = list_runs(cfg)
    exact = [r for r in runs if r["run_id"] == ref]
    hits = exact or [r for r in runs if ref in r["run_id"]]
    if not hits:
        raise RunNotFoundError(f"No run matches '{ref}'. List them with: python -m hcai runs")
    if len(hits) > 1:
        ids = ", ".join(r["run_id"] for r in hits)
        raise RunNotFoundError(f"'{ref}' matches several runs ({ids}); use the full id.")
    return hits[0]


def _fmt_auc(m: dict[str, Any]) -> str:
    lo, hi = m["roc_auc_ci95"]
    return f"{m['roc_auc']:.3f} [{lo:.2f}, {hi:.2f}]"


def compare_runs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Structured comparison of two run summaries (A = baseline, B = candidate)."""
    sa, sb = a.get("selected"), b.get("selected")
    ma, mb = a["test"].get(sa or "", {}), b["test"].get(sb or "", {})
    same_data = a["data_sha256"] == b["data_sha256"] and a["data_sha256"] != "unknown"
    same_dataset = a["dataset"] == b["dataset"]
    overlap = None
    if ma and mb:
        (alo, ahi), (blo, bhi) = ma["roc_auc_ci95"], mb["roc_auc_ci95"]
        overlap = max(alo, blo) <= min(ahi, bhi)
    deltas = {k: (mb[k] - ma[k]) for k in METRICS if k in ma and k in mb}
    shared = sorted(set(a["test"]) & set(b["test"]))
    notes: list[str] = []
    if same_dataset and same_data and a.get("seed") == b.get("seed"):
        notes.append(
            "Same data and same seed: both runs used the same train/test split, so differences "
            "come from the build (models, tuning, calibration), not from the split."
        )
    elif same_dataset:
        notes.append(
            "Same dataset but the data hash or seed differs (or is unknown for an imported run): "
            "the split may differ, so small differences are within split noise."
        )
    else:
        notes.append(
            "Different datasets: populations, feature sets and test sets differ, so absolute "
            "numbers are not directly comparable. Compare the pattern (model ranking, CI width, "
            "calibration), not the decimals."
        )
        if set(a.get("absent_features", [])) != set(b.get("absent_features", [])):
            notes.append(
                f"Feature sets differ (absent in A: {a.get('absent_features') or 'none'}; "
                f"absent in B: {b.get('absent_features') or 'none'})."
            )
        notes.append(
            "The Kaggle Heart Failure data includes Cleveland-derived rows, so the two results "
            "are not fully independent confirmations."
        )
    if overlap is True:
        notes.append(
            "The 95% CIs of the selected models' ROC-AUC overlap: the difference is not "
            "statistically distinguishable at these test-set sizes."
        )
    elif overlap is False:
        notes.append("The 95% CIs of the selected models' ROC-AUC do not overlap.")
    best_cv = {
        r["run_id"]: max(r["cv"], key=lambda k: r["cv"][k]["mean"]) if r["cv"] else None
        for r in (a, b)
    }
    return {
        "a": a["run_id"], "b": b["run_id"], "same_dataset": same_dataset, "same_data": same_data,
        "selected": {"a": sa, "b": sb}, "delta_selected_b_minus_a": deltas,
        "ci_overlap": overlap, "shared_models": shared, "best_cv_model": best_cv, "notes": notes,
    }  # fmt: skip


def comparison_markdown(a: dict[str, Any], b: dict[str, Any], cmp: dict[str, Any]) -> str:
    def facts(r: dict[str, Any]) -> list[str]:
        lg = r.get("leakage_diagnostics") or {}
        shuffle = (lg.get("label_shuffle") or {}).get("mean")
        return [
            r["run_id"], r["build"], r["dataset"], f"{r['n_train']} / {r['n_test']}",
            ", ".join(r.get("absent_features") or []) or "none", r["selected"] or "n/a",
            r["verdict"], "n/a" if shuffle is None else f"{shuffle:.3f}",
        ]  # fmt: skip

    ra, rb = facts(a), facts(b)
    labels = ["run", "build", "dataset", "train / test rows", "absent features", "selected model",
              "leakage verdict", "label-shuffle AUC (expect ~0.5)"]  # fmt: skip
    lines = [
        f"# Run comparison: {a['run_id']}  vs  {b['run_id']}\n",
        "A = baseline, B = candidate. Generated by `python -m hcai compare`; do not edit by hand.\n",
        "| | A | B |", "|---|---|---|",
        *[f"| {lb} | {x} | {y} |" for lb, x, y in zip(labels, ra, rb, strict=True)],
        "",
        "## Selected model, held-out test (point [95% bootstrap CI])",
        "| metric | A | B | B − A |", "|---|---|---|---|",
    ]  # fmt: skip
    ta, tb = a["test"].get(a["selected"] or "", {}), b["test"].get(b["selected"] or "", {})
    if ta and tb:
        lines.append(f"| ROC-AUC | {_fmt_auc(ta)} | {_fmt_auc(tb)} | {cmp['delta_selected_b_minus_a']['roc_auc']:+.3f} |")  # fmt: skip
        for k in ("accuracy", "precision", "recall", "f1"):
            if k in ta and k in tb:
                lines.append(f"| {k} | {ta[k]:.3f} | {tb[k]:.3f} | {tb[k] - ta[k]:+.3f} |")
    lines += ["", "## Every model, test ROC-AUC [95% CI] and CV ROC-AUC (train split)",
              "| model | A test | B test | A CV | B CV |", "|---|---|---|---|---|"]  # fmt: skip
    for name in sorted(set(a["test"]) | set(b["test"])):

        def cell(r: dict[str, Any], key: str, name: str = name) -> str:
            return _fmt_auc(r["test"][name]) if key == "test" and name in r["test"] else (
                f"{r['cv'][name]['mean']:.3f}±{r['cv'][name]['std']:.3f}"
                if key == "cv" and name in r["cv"] else "—"
            )  # fmt: skip

        lines.append(f"| {name} | {cell(a, 'test')} | {cell(b, 'test')} | {cell(a, 'cv')} | {cell(b, 'cv')} |")  # fmt: skip
    na, nb = a.get("nested_cv_auc"), b.get("nested_cv_auc")
    if na or nb:
        f = lambda n: "n/a" if not n else f"{n['mean']:.3f} ± {n['std']:.3f}"  # noqa: E731
        lines += ["", f"Nested-CV ROC-AUC of the tuned winner (train split): A {f(na)} · B {f(nb)}"]
    lines += ["", "## How to read this", *[f"- {n}" for n in cmp["notes"]]]
    return "\n".join(lines) + "\n"


def write_comparison(cfg: Config, a: dict[str, Any], b: dict[str, Any]) -> Path:
    cmp = compare_runs(a, b)
    out = cfg.paths.artifacts / "reports"
    out.mkdir(parents=True, exist_ok=True)
    stem = _slug(f"comparison_{a['run_id']}__vs__{b['run_id']}")[:150]
    (out / f"{stem}.json").write_text(json.dumps(cmp, indent=2), encoding="utf-8")
    md = out / f"{stem}.md"
    md.write_text(comparison_markdown(a, b, cmp), encoding="utf-8")
    return md
