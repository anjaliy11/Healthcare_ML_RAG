# Split-stability study (cleveland)

100 random stratified splits, test n≈61, default hyper-parameters, same leak-free pipeline.

| model | mean AUC | std | 5th pct | 95th pct | max | share of splits > ceiling |
|---|---|---|---|---|---|---|
| logreg | 0.908 | 0.031 | 0.862 | 0.958 | 0.984 | 5% |
| random_forest | 0.897 | 0.033 | 0.843 | 0.945 | 0.969 | 1% |
| hist_gb | 0.869 | 0.035 | 0.807 | 0.927 | 0.948 | 0% |

Share of splits where the best of these models exceeds the 0.96 ceiling: **5%**; above the hard limit: 0%.

Reading: if clean splits regularly exceed the ceiling, a single test AUC above it is not evidence of leakage by itself. That is why the gate relies on direct diagnostics.
