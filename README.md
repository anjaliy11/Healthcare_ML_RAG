# hcai: healthcare ML risk classifier + grounded RAG QA

**Demonstration prototype, not a medical device.** It does not diagnose or advise treatment.

1. **ML**: heart-disease presence classifier on UCI Cleveland (LogReg / RandomForest / HistGradientBoosting and more), leakage-free pipeline, 5 metrics with bootstrap CIs; optional second dataset (Kaggle Heart Failure Prediction, 918 rows) for a comparison.
2. **RAG**: question answering over 9 public WHO / CDC / NIDDK pages. Answers cite sources, and say so when the documents don't contain the answer.
3. **Interface**: CLI and Streamlit app combining both (risk estimate + cited explanation of the top risk factors).

## Quickstart on Windows (VS Code + PowerShell)
Prerequisites: Python 3.12 or 3.13 (python.org; tick "Add to PATH"), VS Code with the *Python* and *Ruff* extensions.

```powershell
cd hcai
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.lock
pip install -e . --no-deps
```
In VS Code: Ctrl+Shift+P → *Python: Select Interpreter* → `.venv`.

**1. Verify the build (no internet or API key needed):**
```powershell
python -m pytest            # expect: 224 passed, ~93% coverage
python -m ruff check . ; python -m mypy
```
**2. Real data and models (needs internet):**
```powershell
python -m hcai data                    # downloads UCI Cleveland once, caches in data/raw
python -m hcai train                   # ~1.5-3 min; metrics + CIs + plots -> artifacts/reports/
python scripts/fetch_docs.py           # snapshots 9 documents + manifest.json
python -m hcai index                   # downloads bge-small (~130 MB) first time
python -m hcai eval                    # RAG eval + threshold recommendation
python -m hcai status                  # shows what exists + which hcai folder is loaded
```
If a document fails to download, save its text by hand (use `## ` for headings) and run
`python scripts/fetch_docs.py --add-local page.txt --doc-id ID --title T --url U --license L`.
If the UCI download fails, get `processed.cleveland.data` from the UCI page and run `python -m hcai data --source-file <path>`.

**3. Use it:**
```powershell
copy .env.example .env                 # optional: put GEMINI_API_KEY=... (free key: aistudio.google.com/apikey)
                                       # no key => extractive answers (still cited, no LLM)
python -m hcai ask "What are the risk factors for high blood pressure?"
python -m hcai assess --example --out patient.json     # writes UTF-8 (do not use > in PowerShell 5)
python -m hcai assess --input patient.json
python -m streamlit run src/hcai/app/streamlit_app.py
python -m hcai monitor                 # request metrics; add --drift-csv rows.csv for drift
```
`make <target>` works too where `make` exists (see Makefile). Offline demo without model download: `python -m hcai index --offline` (lexical only).
VS Code: *Run and Debug* has ready configs (`.vscode/launch.json`); the Testing panel runs pytest.

After `eval`, read the "Threshold trade-off" section: it shows the false-refusal / false-answer rates at the threshold in use and a suggestion that keeps false refusals ≤ 5%. Change `rag.score_threshold` only if you accept that trade, then re-run `eval`.

**4. Compare runs and datasets** (details and interpretation rules: [docs/RUN_COMPARISON.md](docs/RUN_COMPARISON.md)):
```powershell
python -m hcai runs --import-metrics <old folder>\artifacts\reports\metrics.json --tag build02 --build 0.2 --assume-same-data
python -m hcai train --tag build03
python -m hcai train --dataset kaggle_heart_failure --source-file <path>\heart.csv --tag kaggle   # Kaggle: Heart Failure Prediction Dataset
python -m hcai compare build02 build03
python -m hcai compare build03 kaggle
python -m hcai stability                # split-to-split AUC spread
```

## Troubleshooting (Windows)
| Symptom | Fix |
|---|---|
| `status` prints a `hcai ... loaded from` path that is not this folder | An older install is shadowing it. In this folder: `pip install -e . --no-deps`. |
| `train` stops with "Model NOT saved: …" | A leakage check failed (duplicates, a feature that alone predicts the target, or label-shuffle AUC far above 0.5). Read `artifacts\reports\model_report.md`; this is a real finding, not noise. A test AUC above 0.96 alone only prints a NOTE. |
| CDC page 403 | `fetch_docs.py` retries with alternates (NHLBI) automatically and says which URL it used; if both fail use `--add-local`. |
| `UnicodeDecodeError` reading patient.json | Fixed in 0.3 (UTF-16 handled). Prefer `--example --out patient.json`. |
| `ask` slow (> 25 s) or "notes: llm_fallback:llm_error" | Free-tier quota or network: the answer is still produced from the extractive fallback. Wait a minute, or run `ask --no-llm`. |
| Tests slow | `python -m pytest --durations=10` and look at the slowest. |

## Results
Not stored here by design: run `python -m hcai train` and paste the table from `artifacts/reports/model_report.md`
(test metrics with 95% CIs, CV ROC-AUC, calibration, ablations, leakage checks) and `rag_eval.md`.
Cleveland ROC-AUC is typically 0.88-0.93 but a 61-row test set moves by several points between splits, so a value above 0.96
is only a warning. Training stops (and saves nothing) when a **direct** leakage check fails: train/test duplicate rows, a single feature that alone
separates the classes, or a label-shuffle test AUC far from 0.5. Run `python -m hcai stability` to see the spread on your data.
After running everything, `python -m hcai report` fills the block below with the measured numbers (nothing is typed by hand) and copies the full reports to `docs/results/`.
Run comparisons (build vs build, Cleveland vs Kaggle): `python -m hcai compare`, see [docs/RUN_COMPARISON.md](docs/RUN_COMPARISON.md). What changed in 0.3 and why: [docs/BUILD_NOTES.md](docs/BUILD_NOTES.md).

<!-- RESULTS:START (generated by `python -m hcai report`; do not edit by hand) -->
_Generated 2026-11-24 07:51 UTC by hcai 0.3.0 from `artifacts/`. Test sets are small (Cleveland n≈61): read every number with its CI._

#### Primary model (UCI Cleveland)
303 rows · train 242 / test 61 · absent features: none · selected model: **logreg**

| model | accuracy | precision | recall | F1 | ROC-AUC [95% CI] | CV ROC-AUC (train) |
|---|---|---|---|---|---|---|
| logreg **(selected)** | 0.885 | 0.839 | 0.929 | 0.881 | 0.961 [0.90, 1.00] | 0.899 ± 0.040 |
| logreg_fe | 0.885 | 0.839 | 0.929 | 0.881 | 0.969 [0.92, 1.00] | 0.898 ± 0.037 |
| svm_rbf | 0.869 | 0.812 | 0.929 | 0.867 | 0.960 [0.90, 1.00] | 0.897 ± 0.042 |
| random_forest | 0.869 | 0.812 | 0.929 | 0.867 | 0.949 [0.89, 0.99] | 0.894 ± 0.039 |
| hist_gb | 0.918 | 0.871 | 0.964 | 0.915 | 0.965 [0.92, 0.99] | 0.871 ± 0.041 |
| voting | 0.869 | 0.812 | 0.929 | 0.867 | 0.964 [0.91, 1.00] | 0.901 ± 0.037 |

Nested-CV ROC-AUC of the tuned winner (train split): 0.896 ± 0.044

Leakage checks: **WARN** · duplicate test rows 0 · strongest single feature `thal` AUC 0.78 · label-shuffle AUC 0.51 (chance = 0.5)
- test ROC-AUC 0.969 is above the usual ceiling 0.96, but every direct leakage check is clean (selected-model 95% CI [0.90, 1.00]). With a small test set this is within split noise; run `python -m hcai stability` to measure how often that happens.

#### Second dataset (Kaggle Heart Failure Prediction)
918 rows · train 734 / test 184 · absent features: ca, thal · selected model: **logreg**

| model | accuracy | precision | recall | F1 | ROC-AUC [95% CI] | CV ROC-AUC (train) |
|---|---|---|---|---|---|---|
| logreg **(selected)** | 0.902 | 0.896 | 0.931 | 0.913 | 0.936 [0.90, 0.97] | 0.927 ± 0.028 |
| logreg_fe | 0.897 | 0.895 | 0.922 | 0.908 | 0.935 [0.90, 0.97] | 0.927 ± 0.027 |
| svm_rbf | 0.897 | 0.888 | 0.931 | 0.909 | 0.944 [0.90, 0.98] | 0.924 ± 0.024 |
| random_forest | 0.886 | 0.872 | 0.931 | 0.900 | 0.932 [0.89, 0.97] | 0.931 ± 0.025 |
| hist_gb | 0.902 | 0.896 | 0.931 | 0.913 | 0.934 [0.89, 0.97] | 0.931 ± 0.026 |
| voting | 0.891 | 0.880 | 0.931 | 0.905 | 0.940 [0.90, 0.97] | 0.931 ± 0.024 |

Nested-CV ROC-AUC of the tuned winner (train split): 0.924 ± 0.017

Leakage checks: **PASS** · duplicate test rows 0 · strongest single feature `slope` AUC 0.82 · label-shuffle AUC 0.53 (chance = 0.5)

#### Split stability
100 random stratified splits, test n≈61, default hyper-parameters:

| model | mean AUC | std | 5th pct | 95th pct | max | splits > ceiling |
|---|---|---|---|---|---|---|
| logreg | 0.908 | 0.031 | 0.862 | 0.958 | 0.984 | 5% |
| random_forest | 0.897 | 0.033 | 0.843 | 0.945 | 0.969 | 1% |
| hist_gb | 0.869 | 0.035 | 0.807 | 0.927 | 0.948 | 0% |

#### Archived training runs

| run | build | dataset | model | test ROC-AUC [95% CI] | leakage verdict |
|---|---|---|---|---|---|
| 20261124-072319_cleveland_build03 | 0.3.0 | cleveland | logreg | 0.961 [0.90, 1.00] | warn |
| 20261124-073615_kaggle_heart_failure_kaggle | 0.3.0 | kaggle_heart_failure | logreg | 0.936 [0.90, 0.97] | pass |
| 20261124-073737_cleveland_build02 | 0.2 | cleveland | logreg | 0.961 [0.90, 1.00] | warn |

Comparisons (full tables in `docs/results/`):
- **20261124-072319_cleveland_build03** vs **20261124-073615_kaggle_heart_failure_kaggle**: Different datasets: populations, feature sets and test sets differ, so absolute numbers are not directly comparable. Compare the pattern (model ranking, CI width, calibration), not the decimals. 95% CIs overlap.
- **20261124-073737_cleveland_build02** vs **20261124-072319_cleveland_build03**: Same dataset but the data hash or seed differs (or is unknown for an imported run): the split may differ, so small differences are within split noise. 95% CIs overlap.

#### RAG evaluation (golden set)
30 items · threshold in use 0.500 (false refusal 0.00, false answer 0.62)

| metric | value |
|---|---|
| retrieval hit@1 / hit@3 / hit@5 | 0.95 / 1.00 / 1.00 |
| retrieval MRR | 0.98 |
| refusal precision / recall | 0.88 / 0.88 |
| false-refusal / false-answer rate | 0.05 / 0.12 |
| emergency recall · false alarms | 1.00 · 0 |
| answers (LLM / extractive) | 21 / 0 |
| faithfulness proxy (lexical, LLM answers) | 0.98 |
| latency p50 / p95 (ms) | 3956 / 8908 |
| behaviour failures | 2 |

| retrieval variant | hit@1 | hit@3 | MRR |
|---|---|---|---|
| dense | 1.00 | 1.00 | 1.00 |
| bm25 | 0.71 | 0.95 | 0.82 |
| hybrid (RRF) | 0.95 | 1.00 | 0.98 |
| hybrid + diversity cap | 0.95 | 1.00 | 0.98 |

<!-- RESULTS:END -->

## Architecture
```
UCI data -> validate -> Pipeline(impute+encode+scale+model) -> joblib -> predict_risk()  ─┐
docs -> clean -> section-aware chunks -> bge-small -> FAISS (+JSON) -> retrieve(top-5, threshold)
question -> safety gate -> retrieve -> LLM (cited) | extractive fallback | refusal        ─┴-> app.service.assess() -> CLI / Streamlit
```
Grounding: answers must carry `[C#]` tags; tags not in the retrieved set are stripped and logged; uncited LLM prose is never shown
(falls back to extractive passages); retrieval score below threshold or `NOT_IN_DOCUMENTS` gives the fixed refusal.
Retrieved text is fenced and treated as data. Emergency phrases (chest pain, stroke signs, fainting) get an urgent-care message before any retrieval.
Patient inputs are validated with clinical ranges (rejected, never clamped) and never logged.

## Model selection (what `train` does and why)
Six candidates, each tuned by grid search inside 5x3 repeated stratified CV on the **training split only**:
logistic regression (baseline), logistic regression + 2 engineered clinical features (`hr_pct_max`, `oldpeak_x_exang`),
RBF-SVM (Platt-calibrated), random forest, histogram gradient boosting, and a soft-voting ensemble.
The winner is the highest CV ROC-AUC, **but the baseline is kept unless another model beats it by more than 0.01**
(`ml.selection_margin`): with 242 training rows, fold-to-fold noise is about ±0.03, so smaller gaps are not evidence.
Then, on train only: sigmoid calibration is applied to the winner if it lowers out-of-fold Brier score by more than 0.005,
a recall-oriented threshold is chosen from out-of-fold predictions, and a nested-CV estimate of "tune, then score" is computed.
The test set is touched once. `model_report.md` shows every candidate's test metrics with bootstrap CIs plus a **paired bootstrap
of each candidate's ROC-AUC difference from the baseline**; if no interval excludes 0, the report says none is demonstrably better.
In my development check (a re-encoded copy of Cleveland, not your UCI download, not a reported result) all candidates tied
within noise and logistic regression was kept; expect the same, and say so plainly in your write-up rather than claiming a gain.

## Retrieval strategy and ablation
Default: **hybrid** = dense (bge-small) + BM25, merged by reciprocal-rank fusion, then a per-document diversity cap (max 3 chunks per document).
The refusal gate still uses the dense cosine score, so fusion never changes what counts as "answerable".
Optional cross-encoder reranking (`rag.rerank: true`, `BAAI/bge-reranker-base`) is off by default.
`python -m hcai eval` prints an ablation table (dense / bm25 / hybrid / hybrid + cap) of hit@1/3/5, MRR, document diversity and latency;
`python -m hcai eval --rerank` adds the reranker row. **Keep a variant only if it beats `dense` on your real corpus**, then set
`rag.retrieval_mode`, `rag.max_per_doc` and `rag.rerank` in `configs/default.yaml` accordingly and paste the table into this README.

## Corpus (snapshotted once, see `data/raw/docs/manifest.json`)
WHO fact sheets (hypertension, CVDs, diabetes, healthy diet, physical activity, tobacco; CC BY-NC-SA 3.0 IGO, verify on page),
NIDDK diabetes symptoms & causes, CDC about high blood pressure, CDC high blood pressure risk factors (US government works; verify per page).
URLs are in `src/hcai/rag/fetch.py`. Do not commit text whose license you have not confirmed.

## External services, libraries, models (disclosure)
- **Dataset**: UCI Heart Disease (Cleveland), Detrano et al.; DOI 10.24432/C52P4X; fetched via `ucimlrepo`.
- **Embedding model**: `BAAI/bge-small-en-v1.5` (Hugging Face, via sentence-transformers, runs locally on CPU).
- **LLM / API**: Google Gemini API, free tier, default `gemini-3.5-flash-lite` with automatic fallback to `gemini-3.1-flash-lite` then `gemini-3.6-flash` when a model is rate-limited (needs `GEMINI_API_KEY`; temperature 0). Free-tier terms: Google may use prompts/responses to improve its products, so **never type real patient information**; the app sends only the question and retrieved document passages, never the 13 patient fields. Optional paid provider: Anthropic (`llm.provider: anthropic`). Without a key, answers are extractive.
- **Libraries**: numpy, pandas, scikit-learn, scipy, joblib, pydantic, PyYAML, matplotlib, ucimlrepo, threadpoolctl, faiss-cpu, sentence-transformers (PyTorch), google-genai, streamlit, requests, beautifulsoup4; optional: anthropic; dev: pytest, pytest-cov, ruff, mypy.
- Built with AI assistance (Claude); be ready to explain every module, see `docs/DESIGN.md` for the reasoning.

## Testing
`python -m pytest` runs 224 hermetic tests: fake/hashing embedder, scripted LLM, synthetic patient data (tests only, never used for reported metrics), no network.
Covers validation edge cases, the leakage gate (duplicates, target proxy, label shuffle, warn-vs-abort), the Kaggle adapter and run comparison, Gemini deadline/retry behaviour, grouped citations, UTF-16 input, artifact version/checksum checks, chunking invariants, ingest errors, index corruption,
citation stripping, prompt injection (question and poisoned document), emergency gate, degradation (model missing / index missing / LLM down),
privacy of logs, CLI and a Streamlit smoke test. `eval` (real embedder, optional real LLM) is run on demand.

## Limitations
See [docs/LIMITATIONS.md](docs/LIMITATIONS.md) and [docs/MODEL_CARD.md](docs/MODEL_CARD.md). Notable: 303-row single-site 1980s data, ~61-row test set, no external validation,
several inputs need cardiology tests, probabilities not clinically calibrated, English-only corpus, crude keyword safety gate.
`joblib` files are pickles: only load models you trained.

## Layout
`src/hcai/{ml,rag,monitoring,app,eval}` · `report.py` · `tests/` · `scripts/fetch_docs.py` · `configs/default.yaml` · `docs/` · `artifacts/` (gitignored)
