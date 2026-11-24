# Model card: heart-disease risk classifier (`hcai`)

**Numbers are not stored here.** `python -m hcai train` writes the measured metrics, confidence
intervals, the "which model is best and why" paragraph, calibration and ablations to
`artifacts/reports/model_report.md` and `metrics.json`. Copy results from there into the README.

## Intended use
Educational demonstration of a tabular classifier and its evaluation hygiene. Not for screening,
triage, diagnosis or any clinical decision.

## Data
UCI Heart Disease, Cleveland subset (`ucimlrepo`, id 45; DOI 10.24432/C52P4X), 303 rows, 13
features. Target `num` (0-4) is collapsed to presence (`num > 0`), about 46% positive, so no
resampling is used. Real missing values: `ca` (4) and `thal` (2).

## Method
- Stratified 80/20 split, seed 42. All imputation, encoding and scaling live inside one
  scikit-learn `Pipeline`, fit on the training split only (no leakage by construction).
- Median imputation for numerics, most-frequent for categoricals, one-hot encoding for
  `cp, restecg, slope, thal`, standard scaling for logistic regression only, plus missing-indicator
  flags for `ca`/`thal`.
- Candidates: logistic regression (baseline), random forest, histogram gradient boosting. Small grid
  search inside 5x2 repeated stratified CV on the training split, scored by ROC-AUC.
- Selection: highest CV ROC-AUC, but the interpretable baseline is kept unless another model beats it
  by more than `ml.selection_margin` (config).
- The test set is used once: accuracy, precision, recall, F1, ROC-AUC with 95% bootstrap CIs, Brier score.
- A recall-oriented threshold is chosen on out-of-fold training predictions, never on test.
- Guards: ROC-AUC above `ml.auc_ceiling` (0.96) aborts and refuses to save the model (leakage alarm);
  contract tests check a floor of 0.80 when real data is used.

## Artifact
`artifacts/model.joblib` + `model_metadata.json` (sklearn version, feature schema, data sha256,
git sha, training ranges, test metrics, model checksum). Loading fails loudly on a version, schema
or checksum mismatch.

## Known limits
See [LIMITATIONS.md](LIMITATIONS.md). Small single-site cohort, no external validation, features that
need cardiology tests, uncalibrated probabilities.
