# Comparing training runs

Two comparisons are supported, both produced by the tool from archived run summaries (no hand-typed numbers):

1. **Build 0.2 vs build 0.3 on Cleveland**: what changed after the fixes.
2. **Cleveland vs Kaggle Heart Failure**: does the pipeline behave the same on a second, larger dataset.

## Workflow (PowerShell, from the project folder, venv active)

```powershell
# 0. Get the Kaggle file once (free Kaggle account): open
#    https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction  -> Download -> unzip -> heart.csv

# 1. Cache the Cleveland data in the NEW folder first
python -m hcai data

# 2. Archive the run you already have (build 0.2). Its metrics.json was written even though training aborted.
python -m hcai runs --import-metrics C:\path\to\OLD_FOLDER\artifacts\reports\metrics.json --tag build02 --build 0.2 --assume-same-data

# 2b. New build, primary dataset
python -m hcai train --tag build03

# 3. New build, second dataset (written to artifacts\datasets\kaggle_heart_failure\, never over the Cleveland model)
python -m hcai train --dataset kaggle_heart_failure --source-file C:\path\to\heart.csv --tag kaggle

# 4. See and compare
python -m hcai runs
python -m hcai compare build02 build03       # comparison 1
python -m hcai compare build03 kaggle        # comparison 2
python -m hcai stability                     # how much does test AUC move between splits? (~1 min)
```
Outputs: `artifacts\reports\comparison_*.md/.json`, `artifacts\reports\stability.md`, and per-run folders in `artifacts\runs\`.
`--assume-same-data` records the hash of the Cleveland file cached on this machine for the imported run; use it only if the old run used the same download (it does if you ran `python -m hcai data` once). Without it the comparison says it cannot confirm the split was identical.

## How to read a comparison
- **Same data + same seed** (comparison 1): both runs share the same train/test split, so differences come from the build. Even so, with 61 test rows the 95% CIs are about ±0.06 AUC wide; the report says whether they overlap.
- **Different datasets** (comparison 2): different patients, different test sets, and Kaggle has no `ca`/`thal`, so the models there work from fewer inputs. Compare the *pattern*: which model family wins, how wide the CIs are, calibration, leakage-check results. Do not claim one dataset "scores higher". The Kaggle file also contains Cleveland-derived rows, so it is not an independent confirmation.
- A **model change of < 0.02 AUC is noise** at these sizes (CV std ≈ 0.03).

## Stability: why the old fixed ceiling was removed
`python -m hcai stability` refits the leak-free pipeline (default hyper-parameters) on many random stratified 80/20 splits and reports the spread of test ROC-AUC.

Development measurement (re-encoded copy of Cleveland, 303 rows, test n = 61; **not your download**; re-run on yours):

| model | mean AUC | std | 5th pct | 95th pct | max | splits > 0.96 |
|---|---|---|---|---|---|---|
| logreg | 0.914 | 0.032 | 0.862 | 0.963 | 0.975 | 7% |
| random_forest | 0.903 | 0.032 | 0.856 | 0.954 | 0.965 | 2% |
| hist_gb | 0.877 | 0.036 | 0.818 | 0.934 | 0.953 | 0% |

An earlier 120-split run of the same experiment (different seeds) gave 9% for "best of 3 models > 0.96" and 0% above 0.99. So a clean pipeline occasionally scores above 0.96 on a 61-row test set; your 0.965 is inside that range, and the fixed ceiling was a false alarm. The new gate still stops real leakage: see `docs/BUILD_NOTES.md` #1.

## What goes in your report
Paste the two `comparison_*.md` tables and the stability table, then write the interpretation yourself from them: which numbers moved, whether CIs overlap, what the leakage checks said, and the caveats above. Do not copy numbers from this file; they are development measurements.
