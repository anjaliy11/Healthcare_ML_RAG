# hcai: healthcare ML risk classifier + grounded RAG QA

**Demonstration prototype, not a medical device.** It does not diagnose or advise treatment.

1. **ML**: heart-disease presence classifier on UCI Cleveland (LogReg / RandomForest / HistGradientBoosting), leakage-free pipeline, 5 metrics with bootstrap CIs.
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
python -m pytest            # expect: 152 passed, ~92% coverage
python -m ruff check . ; python -m mypy
```
**2. Real data and models (needs internet):**
```powershell
python -m hcai data                    # downloads UCI Cleveland once, caches in data/raw
python -m hcai train                   # ~1.5-3 min; metrics + CIs + plots -> artifacts/reports/
python scripts/fetch_docs.py           # snapshots 9 documents + manifest.json
python -m hcai index                   # downloads bge-small (~130 MB) first time
python -m hcai eval                    # RAG eval + threshold recommendation
python -m hcai status                  # shows what exists
```
If a document fails to download, save its text by hand (use `## ` for headings) and run
`python scripts/fetch_docs.py --add-local page.txt --doc-id ID --title T --url U --license L`.
If the UCI download fails, get `processed.cleveland.data` from the UCI page and run `python -m hcai data --source-file <path>`.

**3. Use it:**
```powershell
copy .env.example .env                 # optional: put GEMINI_API_KEY=... (free key: aistudio.google.com/apikey)
                                       # no key => extractive answers (still cited, no LLM)
python -m hcai ask "What are the risk factors for high blood pressure?"
python -m hcai assess --example > patient.json
python -m hcai assess --input patient.json
python -m streamlit run src/hcai/app/streamlit_app.py
python -m hcai monitor                 # request metrics; add --drift-csv rows.csv for drift
```
`make <target>` works too where `make` exists (see Makefile). Offline demo without model download: `python -m hcai index --offline` (lexical only).
VS Code: *Run and Debug* has ready configs (`.vscode/launch.json`); the Testing panel runs pytest.

After `eval`, copy the recommended threshold into `rag.score_threshold` in `configs/default.yaml`, re-run `index`/`eval`.

## Results
Not stored here by design: run `python -m hcai train` and paste the table from `artifacts/reports/model_report.md`
(test metrics with 95% CIs, CV ROC-AUC, calibration, ablations) and `rag_eval.md`. Sanity: Cleveland ROC-AUC is expected around 0.88-0.93; above 0.96 the training aborts as suspected leakage.

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
- **Libraries**: numpy, pandas, scikit-learn, scipy, joblib, pydantic, PyYAML, matplotlib, ucimlrepo, faiss-cpu, sentence-transformers (PyTorch), google-genai, streamlit, requests, beautifulsoup4; optional: anthropic; dev: pytest, pytest-cov, ruff, mypy.
- Built with AI assistance (Claude); be ready to explain every module, see `docs/DESIGN.md` for the reasoning.

## Testing
`python -m pytest` runs 152 hermetic tests: fake/hashing embedder, scripted LLM, synthetic patient data (tests only, never used for reported metrics), no network.
Covers validation edge cases, leakage alarm, artifact version/checksum checks, chunking invariants, ingest errors, index corruption,
citation stripping, prompt injection (question and poisoned document), emergency gate, degradation (model missing / index missing / LLM down),
privacy of logs, CLI and a Streamlit smoke test. `eval` (real embedder, optional real LLM) is run on demand.

## Limitations
See [docs/LIMITATIONS.md](docs/LIMITATIONS.md) and [docs/MODEL_CARD.md](docs/MODEL_CARD.md). Notable: 303-row single-site 1980s data, ~61-row test set, no external validation,
several inputs need cardiology tests, probabilities not clinically calibrated, English-only corpus, crude keyword safety gate.
`joblib` files are pickles: only load models you trained.

## Layout
`src/hcai/{ml,rag,monitoring,app,eval}` · `tests/` · `scripts/fetch_docs.py` · `configs/default.yaml` · `docs/` · `artifacts/` (gitignored)
