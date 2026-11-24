# Limitations and future work

This is a demonstration prototype. It is **not a medical device** and must not be used to make
decisions about anyone's care.

## ML model
- **Small, old, single-site data.** UCI Cleveland: 303 patients from one 1980s hospital. The test
  split is ~61 rows, so every metric carries a wide confidence interval (reported in
  `artifacts/reports/`). Differences of a few points between models are noise.
- **No external validation.** The other three UCI sites (Hungary, Switzerland, VA Long Beach) were not used.
  The optional Kaggle Heart Failure dataset is a *combination that includes Cleveland rows*, so it is not independent validation either.
- **Leakage checks are necessary, not sufficient.** They catch duplicates, target proxies and label leaks through the pipeline;
  they cannot detect a dataset that is itself biased or mislabelled.
- **Inputs a lay user cannot supply.** `ca` (fluoroscopy vessel count), `thal`, `oldpeak`,
  `restecg` and `thalach` need clinical tests. `ca`/`thal` may be left blank (median/mode
  imputed and flagged); the UI says which fields need a test.
- **Sex imbalance** (~68% male) means performance for women is less certain.
- **Probabilities are not clinically calibrated.** Brier score and a reliability plot are reported,
  but n is too small to recalibrate. Low/moderate/high cut-points are demo values from config.
- **Per-patient "factors"** are one-at-a-time occlusion against typical values. They show what the
  model reacts to, not causes, and ignore interactions.
- **Scope mismatch.** The model predicts heart-disease presence only, while the document corpus
  also covers diabetes, diet, activity and tobacco.

## RAG
- **English-only** corpus and retrieval. Other languages are mostly refused.
- **General-population fact sheets**, not individualized guidance. Nothing about drug dosing or
  pregnancy-specific care is indexed, so those questions are refused by design.
- **Keyword safety gate** is crude: it can miss novel phrasings and over-triggers on personal or
  present-tense wording by design (a false alarm is cheaper than a miss). General questions
  ("what are the symptoms of a stroke?", "can high blood pressure lead to a stroke?") pass through. Tested on a
  short phrase list, not on real user traffic; English only.
- **Score threshold** depends on the embedder; it must be re-calibrated (`python -m hcai eval`)
  if the model or corpus changes. The offline hashing embedder is lexical, for tests/demos only.
- **An LLM can still paraphrase poorly** even with validated citations. Citation validation proves a
  cited chunk was retrieved, not that the sentence is entailed by it. The faithfulness number in the
  eval is a lexical proxy; spot-check by hand.
- **Retrieval variants are unproven until you run the ablation** on the real corpus; the unit tests prove the mechanics (fusion, cap, rerank), not that hybrid wins on WHO/CDC text.
- **Golden set** labels were written from general knowledge of the fact sheets and must be reviewed
  against the ingested text before trusting the metrics. No clinician reviewed anything.
- **Snapshots go stale.** Pages are saved once; the manifest records retrieval date and checksum.

## Platform
- No authentication, audit trail, multi-user storage or observability stack (by design).
- `joblib` artifacts are pickles: load only models you trained yourself.

## Future work (in rough priority order)
1. External validation on the other UCI sites; calibration with more data.
2. Hybrid BM25 + dense retrieval with reciprocal-rank fusion, then a reranker, each adopted only if
   it improves hit@k on the golden set.
3. Clinician-reviewed golden set; LLM-as-judge faithfulness with human spot checks.
4. Multilingual retrieval; richer safety triage than keywords.
5. Docker, CI release of the index, auth + audit logging, proper observability.
