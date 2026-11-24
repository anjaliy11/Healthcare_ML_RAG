"""`hcai report`: README block is generated from artifacts only, idempotent, and never touches
the rest of the README."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hcai.config import Config
from hcai.errors import HcaiError
from hcai.ml.leakage import split_stability
from hcai.ml.train import train
from hcai.report import END, START, build_results_block, publish


def _readme(cfg: Config, newline: str = "\n") -> Path:
    """Written as bytes so the line-ending style is exactly what the test asks for on any OS."""
    p = cfg.paths.artifacts.parent / "README.md"
    body = f"# Title\nintro text\n{START}\nold\n{END}\n## After\nkeep me\n"
    p.write_bytes(body.replace("\n", newline).encode("utf-8"))
    return p


def test_no_results_is_a_friendly_error(cfg):
    with pytest.raises(HcaiError, match="No results"):
        build_results_block(cfg)


def test_missing_markers_explain_how_to_add_them(cfg):
    p = cfg.paths.artifacts.parent / "README.md"
    p.write_text("# no block\n", encoding="utf-8")
    train_dir = cfg.paths.reports_dir
    train_dir.mkdir(parents=True)
    (train_dir / "metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(HcaiError, match="no results block"):
        publish(cfg)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])  # Windows files use CRLF
def test_publish_uses_artifact_numbers_and_is_idempotent(cfg, synth_data, newline):
    res = train(cfg, data=synth_data)
    s = split_stability(cfg, synth_data[0], synth_data[1], 3, models=("logreg",))
    (cfg.paths.reports_dir / "stability.json").write_text(json.dumps(s), encoding="utf-8")
    readme = _readme(cfg, newline)

    out, copied = publish(cfg)
    raw = out.read_bytes().decode("utf-8")
    text = raw.replace("\r\n", "\n")
    auc = res.report["test"][res.selected]["metrics"]["roc_auc"]
    assert f"{auc:.3f}" in text and "Split stability" in text and "Leakage checks" in text
    assert text.startswith("# Title\nintro text\n") and text.endswith("## After\nkeep me\n")
    assert "\nold\n" not in text  # previous block replaced
    assert ("\r\n" in raw) == (newline == "\r\n")  # line-ending style preserved
    assert "\r\r" not in raw and "\n\n\n" not in text  # no doubled line breaks
    assert any(c.name == "model_report.md" for c in copied)
    assert (readme.parent / "docs" / "results" / "model_report.md").is_file()

    publish(cfg)
    publish(cfg)  # repeated runs must not grow the file
    again = readme.read_bytes().decode("utf-8")

    def strip(t: str) -> str:
        return "\n".join(x for x in t.splitlines() if not x.startswith("_Generated"))

    assert strip(raw) == strip(again) and len(again.splitlines()) == len(raw.splitlines())


def test_rag_block_is_rendered_from_rag_eval_json(cfg):
    cfg.paths.reports_dir.mkdir(parents=True)
    (cfg.paths.reports_dir / "rag_eval.json").write_text(
        json.dumps(
            {
                "n_items": 30, "threshold_used": 0.5,
                "current_rates": {"false_refusal_rate": 0.05, "false_answer_rate": 0.17},
                "retrieval": {"hit@1": 0.8, "hit@3": 0.9, "hit@5": 0.95, "mrr": 0.86},
                "refusal": {"precision": 1.0, "recall": 0.83, "false_refusal_rate": 0.05, "false_answer_rate": 0.17},
                "safety": {"emergency_recall": 1.0, "false_alarms": 0},
                "generation": {"llm_answers": 18, "extractive_answers": 2, "faithfulness_proxy_llm": 0.9,
                               "latency_ms": {"p50": 900, "p95": 4000}},
                "behaviour_failures": [], "ablation": [],
            }
        ),
        encoding="utf-8",
    )  # fmt: skip
    block = build_results_block(cfg)
    assert "0.80 / 0.90 / 0.95" in block and "| behaviour failures | 0 |" in block


def test_cli_report(cfg, synth_data, capsys):
    import yaml

    from hcai.app.cli import main

    train(cfg, data=synth_data)
    _readme(cfg)
    f = cfg.paths.artifacts.parent / "cfg.yaml"
    f.write_text(yaml.safe_dump(json.loads(cfg.model_dump_json())), encoding="utf-8")
    assert main(["--config", str(f), "report"]) == 0
    assert "updated results block" in capsys.readouterr().out
