# SPTnet training-speed benchmark (train_seconds)

Steady-state per-epoch `train_seconds` (warmup epochs dropped: 1; 10000 hierarchical bootstrap resamples). train_seconds = compute only; epoch_total_seconds = full epoch incl. plotting.

| Config | Runs | Epochs | Mean s/epoch | 95% CI |
|---|---|---|---|---|
| new_fresh_good_runs | 8 | 24 | 102.81 | [102.18, 103.33] |
| old_good_8_runs | 8 | 24 | 445.55 | [441.02, 450.54] |

## Headline speedup
**old_good_8_runs / new_fresh_good_runs = 4.33x** (95% CI [4.28, 4.39]).
