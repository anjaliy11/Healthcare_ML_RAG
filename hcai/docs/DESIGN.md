# DESIGN.md — Healthcare ML + RAG prototype

## 1. What the assignment actually grades
The brief is small on purpose ("small dataset, lightweight models, demonstration prototype, be prepared to explain your code"). Reviewers will look for:

| Signal | How we show it |
|---|---|
| Correct ML hygiene | Stratified split, leakage-free pipeline, 5 required metrics, CIs, model comparison with a reason |
| Honest RAG | Visible sources, explicit "not in documents" behavior, no hallucinated citations |
| Engineering | Tests, README that runs from a clean clone, pinned deps, config, logging |
| Judgment | Limitations section that is specific, not generic; clear scope vs. medical advice |
| Explainability of your own work | Small codebase you can walk through; DESIGN.md records why |

Explicit requirement checklist (every row must be traceable to a file):

| Req | Where |
|---|---|
| Clean/preprocess, missing values, train/test split | `ml/data.py`, `ml/features.py` |
| ≥2 classifiers; accuracy, precision, recall, F1, ROC-AUC | `ml/train.py`, `ml/evaluate.py` → `artifacts/reports/metrics.json` |
| Explain best model | `docs/MODEL_CARD.md` |
| Save model + prediction function | `ml/registry.py`, `ml/predict.py` |
| 5–10 public docs, clean, chunk, embed, vector DB | `rag/ingest.py`, `chunking.py`, `embeddings.py`, `store.py` |
| Retrieve, LLM answer, show sources, say when unavailable | `rag/retriever.py`, `generator.py`, `pipeline.py` |
| Interface (CLI/Streamlit) | `app/cli.py`, `app/streamlit_app.py` |
| README, dataset link, metrics, screenshots, limitations, external APIs listed | `README.md`, `docs/LIMITATIONS.md` |

## 2. Research findings

### 2.1 Dataset
Candidates considered:

| Dataset | Rows | Real missing values | Fetch | Fit to brief |
|---|---|---|---|---|
| **UCI Heart Disease (Cleveland)** | 303, 13 features | Yes: `ca` 4, `thal` 2 (6 rows affected) | `ucimlrepo` id=45, programmatic | Strong: real NaNs to handle, small, links to hypertension/CVD docs |
| Pima Indians Diabetes | 768, 8 features | Hidden: zeros in glucose/BP/insulin/BMI mean "missing" | OpenML/Kaggle, provenance murky | Matches "diabetes symptoms" example, but the missing-value trap is a known gotcha and licensing is unclear |
| Statlog (Heart) | 270 | None | UCI id=145 (CC BY 4.0) | Too clean; nothing to show for the missing-value requirement |
| Combined 4-site heart data | 920 | Heavy (up to ~60% in some columns) | UCI | More rows but large site shift; scope creep |

**Decision: Cleveland.** The target `num` is 0–4; published work collapses it to presence (1–4) vs absence (0). Class balance is ~54% / 46%, so no resampling is needed (and we should say why not). Verify the license on the UCI page before submission and cite the DOI 10.24432/C52P4X.

Mismatch to own: the brief's example RAG questions mention diabetes. The corpus therefore covers CVD, hypertension, diabetes, diet, activity, tobacco. The ML model predicts heart disease only. State this plainly.

Known weaknesses to disclose: 303 patients, 1980s single-hospital cohort, `ca`/`thal`/`oldpeak` need cardiology tests (a lay patient cannot enter them), sex imbalance (~68% male), no external validation.

### 2.2 Corpus (RAG)
Target 8 documents, all freely published, snapshotted locally:

1. WHO fact sheet: Hypertension
2. WHO fact sheet: Cardiovascular diseases (CVDs)
3. WHO fact sheet: Diabetes
4. WHO fact sheet: Healthy diet
5. WHO fact sheet: Physical activity
6. WHO fact sheet: Tobacco (CVD risk section is relevant)
7. CDC or NIDDK: diabetes symptoms / when to see a doctor
8. NHLBI/CDC: high blood pressure risk factors and lifestyle changes

Licensing: WHO publications are generally CC BY-NC-SA 3.0 IGO (non-commercial, attribution, share-alike), fine for an assessment; record license per document in `manifest.json` and attribute in README. US government (CDC/NIH) pages are generally public domain but confirm per page. Do not redistribute anything with an unclear license; if unsure, commit the fetch script + manifest and not the text.

Constraint: build sandbox cannot reach who.int / cdc.gov, so `scripts/fetch_docs.py` runs on your machine (requests + trafilatura/BeautifulSoup), writes `data/raw/docs/*.txt` + `manifest.json` with sha256. Tests use small hand-written fixture docs, never the real corpus.

Corpus gap is a feature: the eval set includes questions the corpus cannot answer (drug dosing, pregnancy hypertension specifics if not ingested), so refusal behavior is measured, not assumed.

## 3. Architecture

```
          ┌──────────── ML path ────────────┐        ┌──────────────── RAG path ────────────────┐
UCI data → validate → Pipeline(impute+scale/  → joblib  docs → clean → section-aware chunk → bge-small
           stratified split   encode + model)   artifact         embeddings → FAISS(+metadata JSON)
                                   │                                            │
 patient input → pydantic validate → predict_risk() ──► risk + band    question → safety gate → retrieve(top-k, threshold)
                                   └────────────┬───────────────────────────────┘                     │
                                                ▼                                  LLM (grounded, cited) / refusal
                                      app.service.assess() → CLI / Streamlit
                                                │
                                   monitoring: JSONL metrics, drift check
```

Integration of the two paths (Part 3): ML returns a risk band and top contributing factors (from LR coefficients or permutation importance, whichever model wins). The service turns the top factors into retrieval queries ("how does high resting blood pressure affect heart disease risk") and the RAG layer returns cited explanations. The LLM never sees the raw probability as a clinical claim; it explains factors from documents.

## 4. Design decisions and tradeoffs

### 4.1 ML
| Decision | Why | Tradeoff / alternative |
|---|---|---|
| Single sklearn `Pipeline` + `ColumnTransformer` | Eliminates leakage by construction; one artifact to ship | Slightly more boilerplate than ad-hoc pandas |
| Median / most-frequent imputation + `add_indicator=True` | Only 6 missing cells; indicator keeps "was missing" signal | Dropping 6 rows is simpler but loses 2% of 303 |
| One-hot for `cp, restecg, slope, thal`; scale numerics for LR only | Right tool per model; trees need no scaling | Slightly different pipelines per model, handled by a factory |
| Models: LR, RF, HistGradientBoosting | Interpretable baseline, bagging, boosting; all CPU-trivial. HGB handles NaN natively, a useful cross-check on imputation | XGBoost/LightGBM add deps for no gain at n=303 |
| Selection by 5×2 repeated stratified CV ROC-AUC, then **one** test evaluation | n=303 makes a single split noisy | Nested CV is purer but overkill; document that tuning is a small grid inside CV only |
| Bootstrap 95% CIs on test metrics | 61 test rows → ±8–10 pts on accuracy | Cost: a few lines. Benefit: prevents over-claiming |
| Threshold: default 0.5 for headline metrics; also report recall-oriented threshold | Screening favors recall; false negatives cost more | Threshold chosen on CV out-of-fold predictions, never on test |
| Calibration check (reliability + Brier) | A "risk %" shown to users must mean something | Don't recalibrate unless clearly miscalibrated; n too small |
| Model choice explanation | Likely outcome: LR ≈ RF ≈ HGB within CI. Say so. Prefer LR if indistinguishable (interpretable, calibrated) | Do not pretend a 1-point difference is real |
| Artifact: joblib + `metadata.json` (sklearn version, schema, metrics, data sha256, git sha) | Load-time version/schema check | joblib is pickle: only load trusted artifacts (note in README) |

Prediction function contract: `predict_risk(PatientInput) -> RiskResult{probability, band, top_factors, model_version, disclaimer}`. Bands (low/moderate/high) cut-points are defined in config and explained as demo-only.

### 4.2 RAG
| Decision | Why | Tradeoff / alternative |
|---|---|---|
| Section-aware chunking (split on headings, then recursive ~350–450 tokens, ~60 token overlap) | Fact sheets are heading-structured; keeps symptoms/risk-factor lists intact | Fixed-size splits are simpler but cut lists mid-way |
| Chunk metadata: `doc_id, title, url, section, chunk_idx, license` | Needed for citations and attribution | Slightly larger index; trivial at this size |
| `bge-small-en-v1.5` (33M params, 384-d) with its query instruction prefix | Strong retrieval for size, CPU-fast, free, no API | `all-MiniLM-L6-v2` is lighter but weaker; OpenAI embeddings add an API dependency and cost |
| FAISS `IndexFlatIP` on normalized vectors (exact cosine) | < 1k chunks: exact search is instant and deterministic; no tuning | Chroma gives built-in metadata filtering and persistence; for this scale we use a JSON sidecar and a `VectorStore` protocol so swapping is one file |
| Dense retrieval, top-k=5, then score threshold | Simple, measurable | Hybrid BM25+RRF is the first stretch ablation: add only if it measurably improves hit@k on the golden set |
| "Unavailable" handled in **two layers**: retrieval-score threshold, then prompt-level `NOT_IN_DOCUMENTS` | Threshold alone misses plausible-but-irrelevant chunks; prompt alone trusts the LLM | Threshold calibrated on golden answerable vs unanswerable scores; report the tradeoff curve |
| Citation protocol: LLM emits `[C1]`-style tags mapped to chunk ids; post-validated | Prevents fabricated sources | Slight prompt complexity; large trust gain |
| Temperature 0, max tokens capped, fenced context | Determinism, cost, injection resistance | Less fluent prose; acceptable |
| LLM behind `LLMClient`; `ExtractiveFallback` returns top passages verbatim with sources | Demo works with no key/offline; tests are hermetic | Not a "generated" answer, so it's labeled as such |
| Safety gate before retrieval | Red-flag symptoms must not wait on an LLM | Keyword list is crude; documented as a limitation |

### 4.3 What we deliberately do not build
LangChain/LlamaIndex (more code to explain than it removes), a reranker (measure first), fine-tuning, Docker/K8s (a Dockerfile is optional if time remains), a database, auth.

## 5. Edge cases (each maps to a test)

**ML input:** out-of-range values (age 0/200, BP 0, chol 5000), wrong types, missing required field, unknown category code, extra fields, NaN for `ca`/`thal` (allowed, imputed, flagged), boundary values, empty batch, artifact missing, artifact trained on a different sklearn version or schema (fail loudly with remediation message), deterministic output across runs.
**ML training:** target collapse correct (`num>0`), stratification preserved in split, no test rows in CV folds, no imputer fit on test, metrics keys complete, ROC-AUC sanity ceiling.
**RAG ingest:** empty document, doc with no headings, very long line, non-UTF8/HTML residue, duplicate doc (sha256 dedupe), boilerplate stripping (nav/footers), manifest/file mismatch.
**RAG query:** empty/whitespace query, very long query (truncate and warn), non-English query (respond that only English docs are indexed or answer if embedder handles it, but state sources are English), out-of-domain ("who won the world cup"), medical-but-uncovered ("metformin dose"), emergency phrasing, prompt injection in the question and in a poisoned fixture chunk, conflicting passages, query that matches a chunk below threshold, index missing/corrupt, LLM timeout/rate-limit/empty reply (retry with backoff, then fall back to extractive), LLM returns a citation that wasn't retrieved (stripped), LLM ignores citation format (fallback to refusal or extractive, never uncited prose).
**Combined:** high-risk prediction with no relevant doc retrieved (show risk, say explanation unavailable), ML passes but RAG fails (degrade gracefully, and vice versa).

## 6. Testing strategy
Pyramid, all hermetic (fake embedder = deterministic hash-based vectors; fake LLM = scripted replies):

- **Unit** (~60%): schema validation, imputation behavior, chunker invariants (no empty chunks, overlap, max length, metadata preserved), store add/search/save/load roundtrip, safety gate, citation validator, prompt builder (context fencing), threshold logic, config loading.
- **Contract:** `predict_risk` output schema and monotonic sanity (e.g., adding the same patient twice yields identical output), model artifact loads and metadata matches, metrics floor (ROC-AUC ≥ 0.80 on test; fail CI if the model is broken) and ceiling (≤ 0.96, leakage alarm).
- **Integration:** tiny 3-document fixture corpus → ingest → index → ask answerable/unanswerable/injection questions with the fake LLM → assert sources and refusals.
- **Evaluation (not unit tests, run on demand):** `make eval` with the real embedder and, if a key exists, the real LLM.
- **Tooling gates:** `ruff`, `mypy`, coverage target ≥ 80% on `src/`, GitHub Actions running lint+tests on push.

## 7. Evaluation plan

**ML:** table of 3 models × (accuracy, precision, recall, F1, ROC-AUC) on test with 95% bootstrap CIs, plus CV mean±std; confusion matrix; ROC and calibration plots; threshold analysis from out-of-fold predictions; permutation importance; one ablation (no indicator, or drop-rows imputation).

**RAG golden set** (`eval/golden.jsonl`, ~30 items, written by you after reading the ingested docs):
- ~20 answerable (symptoms, risk factors, lifestyle changes, when to see a doctor), each labeled with the expected `doc_id` (+ section when possible)
- ~6 unanswerable / out-of-scope (dosing, unrelated topics, things the corpus lacks)
- ~4 adversarial (injection, emergency, vague, misspelled)

Metrics: retrieval hit@1/3/5 and MRR against expected doc; refusal precision/recall; citation validity rate (every cited id ∈ retrieved ids); faithfulness proxy (answer sentences supported by cited chunk via token-overlap/NLI-lite; optional LLM-judge flagged as such); latency p50/p95. Report the threshold chosen and the false-refusal vs false-answer tradeoff. Keep one failure-analysis paragraph: pick 3 real failures and say what you'd fix.

## 8. Monitoring (right-sized)
- Structured JSON logs with request id; no patient inputs logged.
- `artifacts/metrics.jsonl`: per request, latency (retrieve/generate), top-1 score, refusal flag, fallback used, token counts, error type.
- `make monitor` reads it and prints: request count, refusal rate, p50/p95 latency, retrieval-score distribution, fallback rate.
- Input drift: compare incoming ML inputs (ranges only, aggregated) to the training distribution using KS test for numerics and chi-square for categoricals; flag when p < 0.01 across a rolling window. Out-of-range rate as a data-quality metric.
- Index/model provenance shown in UI footer: model version, index hash, embedding model.
- Explicitly out of scope: Prometheus/Grafana (mention as a future improvement).

## 9. Limitations to write (specific, not generic)
Small single-site 1980s cohort; no external validation; features requiring cardiology tests; sex imbalance; probabilities not clinically calibrated; ML explains heart disease while corpus also covers diabetes; English-only retrieval; fact sheets are general-population guidance, not individualized; LLM can still paraphrase poorly even with citations; keyword safety gate is crude; no clinician review; not a medical device.
Future: external validation on the other three UCI sites, calibration with more data, hybrid retrieval + reranker, clinician-reviewed golden set, LLM-as-judge faithfulness with human spot checks, multilingual, auth/audit logging, Docker + proper observability.

## 10. Schedule for today (realistic ~10–11 h of work)
| Block | Hours | Output / gate |
|---|---|---|
| 0. Scaffold, config, logging, CI, Makefile | 0.75 | `make lint test` green |
| 1. ML data + pipeline + tests | 1.5 | tests green |
| 2. Train + evaluate + model card | 1.25 | metrics.json, plots, ceiling check |
| 3. Predict + registry + contract tests | 0.75 | |
| 4. Fetch docs, ingest, chunk, embed, store + tests | 1.75 | index builds |
| 5. Retriever, safety, generator, pipeline + integration tests | 1.75 | fakes then real |
| 6. Golden set + RAG eval + threshold | 1.25 | report |
| 7. Monitoring | 0.5 | `make monitor` |
| 8. CLI + Streamlit + screenshots | 1.0 | demo works |
| 9. README, LIMITATIONS, clean-clone run, polish | 1.0 | `make all` from fresh clone |

Cut line if behind schedule (in order): Streamlit polish → drift check → hybrid retrieval → bootstrap CIs on RAG. Never cut: leakage-free pipeline, citations, refusal behavior, tests, limitations.

## 11. Risks
| Risk | Mitigation |
|---|---|
| Docs can't be fetched/parsed cleanly | Fetch script + manual fallback (save page text by hand), manifest tracks it |
| No LLM key / rate limits | Extractive fallback; disclose in README |
| Over-trusting 0.9 AUC on 61 rows | CIs, repeated CV, plain-language caveat |
| Time overrun | Gates + cut line above |
| "Is this your own work?" | Small codebase, DESIGN.md decisions, be ready to explain each module; cite AI assistance if the brief or company expects it |

## 12. README outline
Overview + disclaimer → quickstart (3 commands) → dataset + citation → results table → RAG corpus table with licenses → architecture diagram → how to run CLI/Streamlit → testing/eval → external APIs/libraries/models (explicit list, as the brief demands) → limitations & future work → repo layout.

## 12b. LLM provider decision (changed from the original Anthropic default)
Reason: no paid API access. Chosen: Gemini free tier, `gemini-3.5-flash-lite` (cheap, fast, enough for short
grounded answers with citation tags at temperature 0), falling back to `gemini-3.1-flash-lite` and `gemini-3.6-flash`
because free quotas are per model and unpublished. Alternatives considered: Gemini 3.8 Flash (stronger but its
paid price doubles on 1 Jan 2027 and free limits are tighter), Gemma 4 (free, but weaker at following the
citation format). Risks: model IDs and quotas change; free-tier data may be used by Google (so no patient fields are ever sent);
the extractive fallback keeps the app working if every model fails. Model names are config, not code.

## 12c. Retrieval upgrades (implemented, evaluated by ablation)
BM25 is implemented in-repo (`rag/bm25.py`, no dependency); dense and lexical rankings are merged with RRF (k=60) in `rag/retriever.py`.
A per-document cap trades a little recall for broader, multi-source answers. A cross-encoder reranker is optional (`rag/rerank.py`).
The threshold gate stays on the dense cosine because it is the only calibrated score scale. Decision rule: ship a variant only if
the ablation in `artifacts/reports/rag_eval.md` shows it beats dense on the real corpus; with ~25 labelled items, differences under
~0.05 are noise. Not done (ideas): small-to-big chunk retrieval, query rewriting, NLI sentence-level faithfulness.

## 12d. ML model improvements (implemented, evaluated, outcome recorded in generated reports)
Added: domain features (heart-rate reserve share, ST-depression x exertional angina), RBF-SVM, soft-voting ensemble, conditional sigmoid
calibration, nested CV, paired-bootstrap comparison with the baseline, and a wider C grid for the baseline. Rationale for restraint: n=303,
test n=61, CV noise about +-0.03 AUC, so the selection rule keeps the interpretable baseline unless a rival wins by > 0.01 and the report
states whether any difference is statistically distinguishable. Rejected: SMOTE/resampling (classes are ~54/46), deep models, stacking meta-learners
(more variance than data supports), tuning on the test set.

## 13. Implementation addendum (dependency justification and deviations)
| Dependency | Why it is needed |
|---|---|
| numpy, pandas, scikit-learn, scipy, joblib | Data, Pipeline/models/CV/metrics, KS and chi-square drift tests, model artifact |
| pydantic, pyyaml | Boundary validation with range checks; typed YAML config |
| matplotlib | ROC and calibration plots (Agg backend, no display) |
| ucimlrepo | Programmatic dataset download (settled decision) |
| faiss-cpu, sentence-transformers | Vector index and `bge-small-en-v1.5` embeddings (settled decisions) |
| google-genai | Default LLM client (Gemini free tier) behind `LLMClient`; `anthropic` is an optional extra |
| streamlit | Optional UI (Part 3) |
| requests, beautifulsoup4 | One-time corpus snapshot script and HTML cleaning in ingest |
Dev: pytest, pytest-cov, ruff, mypy (+ type stubs).

Deviations from the layout: `rag/fetch.py` (fetch logic, so it is unit-tested; `scripts/fetch_docs.py` is a thin wrapper);
`HashingEmbedder` in `rag/embeddings.py` (lexical, offline; used for tests and `index --offline`);
`python -m hcai` replaces `make` on Windows. The occlusion-based factor attribution is model-agnostic so it works
for whichever model is selected.
