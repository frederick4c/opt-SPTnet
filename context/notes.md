# Project Notes

Use this file for durable notes that future coding agents should know but that
do not belong in the report or source comments.

## Notes

- 2026-06-09: Diffusion/Hurst ablations used generated CSD3 binned data under
  `./diff_bins/<condition>/*.h5`, covering diffusion bins across roughly
  `D=0.0-0.5`. The existing pretrained model was under
  `./perf_exp/trained_model`, with experiment outputs under `./perf_exp` and
  diffusion-evaluation summaries under `diff_evals/`.

- 2026-06-09: Current/pre-ablation diffusion evaluation showed reasonable
  mid-range performance but compression at high D. In
  `diff_evals/current_model_diffusion_eval_summary.csv`, the current model
  predicted mean D values of about `0.0808`, `0.1718`, `0.2594`, `0.2698`,
  `0.2474`, `0.3209`, and `0.3612` for target D values `0.05`, `0.15`,
  `0.25 +/- 0.01`, `0.25 +/- 0.05`, `0.25 +/- 0.15`, `0.35`, and `0.45`.
  The strongest visible issue was underprediction at high D, especially
  target `D=0.45`, where the mean prediction was `0.3612` and bias was
  `-0.0888`.

- 2026-06-09: Fine-tuning ablation results are in
  `diff_evals/finetune/comparison_plots/`. Ranking by mean condition MAE:
  `BCE logits` `0.026648`, `Baseline` `0.026780`, `H/D off match` `0.027027`,
  `H/D up` `0.027211`, and `Relative D` `0.029162`. The differences among the
  first four fine-tune variants were very small; the short fine-tune sweep is
  useful for screening but should not be overinterpreted as a strong
  statistical result.

- 2026-06-09: Scratch ablation results are in
  `diff_evals/scratch/comparison_plots/`. Ranking by mean condition MAE:
  `H/D off match` `0.025790`, `Log H/D` `0.032165`, `BCE logits` `0.087909`,
  and `Baseline` `0.091283`. Weighted MAE values were `0.023002`, `0.024177`,
  `0.118418`, and `0.128964` respectively. The scratch comparison is much more
  decisive than the fine-tune comparison.

- 2026-06-09: Interpretation of the scratch results: the scratch baseline and
  scratch BCE-logits-only models did not learn tracking/objectness well enough
  for meaningful D/H prediction. Their diffusion outputs were close to a
  mean-like constant, so they should not be read as clean tests of the
  diffusion loss alone. The useful scratch signal is that both `Log H/D` and
  `H/D off match` improved diffusion matching substantially once training
  avoided the weakest assignment behavior.

- 2026-06-09: The best scratch run was `H/D off match`, meaning H/D were
  removed from the Hungarian matching cost but still contributed to the final
  loss. This is the strongest practical model-improvement finding so far. The
  likely explanation is that noisy H/D predictions should not determine query
  assignment, especially early in training; objectness and coordinate quality
  are more reliable for matching, while D/H can still learn after assignment.

- 2026-06-09: `Log H/D` from scratch was second best. Its prediction slope was
  close to ideal (`0.9949`) with correlation `0.9648`, but its MAE and high-D
  MAE were worse than `H/D off match`. It remains a candidate for a combined
  follow-up run, but current evidence favors H/D-off matching alone.

- 2026-06-09: Scratch `Baseline` and `BCE logits` performed poorly for D in the
  saved evaluation. Both had near-zero prediction slope/correlation and far
  fewer confident tracks than `H/D off match` and `Log H/D`. This may indicate
  objectness/detection collapse, optimization instability, or thresholding
  effects, not just a diffusion-regression problem.

- 2026-06-09: Final model plan from the ablations: train one full model on all
  available data using all three useful conditions together:
  BCE-with-logits objectness, log-style D/H losses, and H/D removed from
  Hungarian matching while retained in the final scalar loss. Rationale:
  fine-tuning suggested BCE-with-logits helps when starting from a competent
  model, while scratch training showed log H/D and H/D-off matching improve
  diffusion behavior. The final run should then be evaluated on diffusion,
  detection, and tracking before being presented as the thesis model.

- 2026-06-09: Relative diffusion loss was not promising in the fine-tune sweep
  (`mean_condition_mae=0.029162`, worst of the fine-tune variants). It may be
  conceptually better as a scratch objective than as a fine-tune objective
  because a model pretrained with absolute loss has already shaped its output
  calibration. The scratch follow-up used log-style D/H instead of simple
  relative D.

- 2026-06-09: Significance testing is only partly justified with the current
  results. The fine-tune ablation differences are small enough that repeated
  seeds or bootstrap confidence intervals would be needed for a strong claim.
  The scratch `H/D off match` improvement over scratch baseline is large enough
  for an engineering conclusion, but formal inference should still use
  resampling over videos/conditions or repeated seeds.

- 2026-06-09: Plotting helper
  `notebooks/finetune_diffusion_plots.py` now handles both fine-tune and
  scratch/final evaluation layouts via `--results-root` and `--plot-prefix`.
  The final cell of `notebooks/testing.ipynb` currently runs the final
  full-model comparison and embeds the ranking table plus plots.

- 2026-06-09: Slurm entry points for these experiments are
  `slurm/finetune_perf_ablation_csd3.slurm` and
  `slurm/train_scratch_ablation_csd3.slurm`. The scratch script was configured
  for 30 epochs on the binned data with variants for baseline, log D/H,
  BCE-with-logits objectness, and H/D-off matching. Fine-tune runs should use a
  fresh optimizer (`--no-resume-optimizer`) when comparing variants from the
  same pretrained weights.

- 2026-06-09: Useful generated artifacts:
  `diff_evals/finetune/comparison_plots/finetune_model_ranking.csv`,
  `diff_evals/scratch/comparison_plots/scratch_model_ranking.csv`,
  `diff_evals/scratch/comparison_plots/scratch_overall_ranking.png`,
  `diff_evals/scratch/comparison_plots/scratch_prediction_calibration.png`,
  `diff_evals/scratch/comparison_plots/scratch_condition_mae_heatmap.png`,
  `diff_evals/scratch/comparison_plots/scratch_condition_bias_heatmap.png`,
  `diff_evals/scratch/comparison_plots/scratch_mean_sweep_comparison.png`, and
  `diff_evals/scratch/comparison_plots/scratch_range_sweep_comparison.png`.

- 2026-06-10: Final full-model diffusion-evaluation results are under
  `diff_evals/final/diff_evals/final_full_model/`, with plots/CSVs written to
  `diff_evals/final/comparison_plots/` and displayed in the final cell of
  `notebooks/testing.ipynb`. The final model collected `2910` confident tracks
  across the diffusion-bin evaluation. Mean condition MAE was `0.065615`,
  weighted MAE was `0.059732`, high-D MAE was `0.084907`, prediction slope was
  `0.709405`, and prediction correlation was `0.771360`.

- 2026-06-10: The final full model did not reproduce the strong binned
  ablation diffusion performance. It is roughly comparable to the previous
  current model by mean condition MAE (`~0.065`) and somewhat better by
  weighted MAE, but substantially worse than the best binned fine-tune/scratch
  variants (`~0.026` mean condition MAE). The main residual failure is high-D
  compression: target `D=0.45` had mean predicted D `0.340646` with bias
  `-0.109354` and within-range fraction `0.190955`.

- 2026-06-10: Next experiment decision: run a targeted fine-tune from the final
  full model on the balanced binned data in `./diff_bins/*/*.h5`. This is
  worthwhile because the binned fine-tune/scratch results show the model and
  loss can learn much better diffusion calibration, while the final full model
  likely learned broader tracking/detection but still underweights high-D
  calibration. Use the final combined settings (BCE-with-logits objectness,
  log-style D/H losses, H/D removed from matching but retained in the final
  loss), a low learning rate such as `1e-5`, validation every epoch, patience
  `5`, and a fresh optimizer via `--no-resume-optimizer`.

- 2026-06-10: The key criterion for the targeted binned fine-tune is not just
  lower D MAE. It should improve high-D calibration while retaining the final
  full model's tracking/detection quality. Compare against the current model,
  final full model, best binned fine-tune ablation, and best scratch binned
  model before selecting the thesis model.

- 2026-06-10: The diffusion-teacher model/experiment is now considered a side
  experiment rather than a main thesis result. It should likely be left out of
  the report unless it provides clear additional value, such as a measurable
  improvement over the direct binned fine-tune path or a useful explanatory
  comparison.

## 2026-06-10 audit: evaluation metric is not per-track accuracy

- The diffusion evaluation in `notebooks/diffusion_eval.py` and the ablation
  plotting in `notebooks/finetune_diffusion_plots.py` do NOT measure per-track
  diffusion accuracy. Confirmed by reading the code and inspecting result files:
  - Result `.h5` files under `diff_evals/**/inference_results/` contain only
    predictions (`estimation_C`, `estimation_H`, `estimation_xy`,
    `obj_estimation`) and a `source_file` attr. The ground-truth per-particle
    D/H labels live in the source videos and are never read by the eval.
  - `collect_predictions` scores each confident prediction against the condition
    nominal mean: `abs_error_to_target = abs(predicted_D - target_mean)`. There
    is no prediction-to-particle matching.
  - `compute_ranking` regresses pooled `predicted_D` against `target_D` (the
    condition mean, only 5-7 discrete values), so the reported slope/correlation
    are calibration-of-means, not per-track calibration.

- The generator samples each particle's D uniformly within `[d_min, d_max]`
  (`data_generator.py`, `_rand_range` is uniform, line ~634). So in e.g.
  `D_mean_0p45_pm_0p05`, true D is spread over [0.40, 0.50] but every prediction
  is scored against 0.45. The metric is therefore minimised by predicting the
  condition mean and penalises a correct per-track predictor.

- Per-track oracle floor (mean abs deviation of a uniform range) that even a
  perfect model would score under this metric:
  - +/-0.05 conditions: ~0.025
  - +/-0.01 condition: ~0.005
  - +/-0.15 condition: ~0.075
  - mean over the seven conditions: ~0.029
  The best binned ablation reports `mean_condition_mae = 0.0257`, BELOW the
  ~0.029 oracle floor. That is only possible by under-dispersing (shrinking
  predictions toward the conditional mean). So lower scores partly reward
  variance shrinkage, not accuracy, and rankings may change once fixed.

- `within_range` (fraction inside [mean +/- hw]) is a coverage metric, not
  accuracy, and is near-impossible to satisfy for the +/-0.01 condition given
  prediction noise. Both headline metrics (mae_to_target, within_range) are
  maximised by collapsing to the per-condition mean.

- The scratch baseline/BCE-logits runs still scored badly (~0.09) only because
  they collapsed to a single GLOBAL constant (~0.26, the fit intercept) across
  all conditions, which is far from the 0.05 and 0.45 targets. The metric
  penalises global collapse but rewards per-condition mean-tracking with low
  variance, so it cannot distinguish a well-calibrated narrow predictor from a
  truly per-track-accurate one.

- Detection errors are invisible: predictions are pooled and unmatched, so false
  positives, duplicate query slots, and missed particles are never penalised.
  Confident-track counts swing from 874 to 2976 across models, meaning the
  per-condition averages are computed over different self-selected subsets.

- Sampling is unequal across conditions: low-D conditions have 100 result files
  each, the rest have 20 (confirmed by file counts). The pooled slope/correlation
  and `weighted_mae` are dominated ~5:1 by the two low-D conditions where the
  model is strongest, so global calibration metrics look optimistic.
  `mean_condition_mae` averages per condition and is more robust, but is still
  built on the flawed target.

- The "current model" baseline numbers were computed with the
  `diffusion_eval.run_diffusion_eval` default `max_files=20`, while the
  scratch/finetune/final plots use `max_files=None` (all files). So the
  "final ~= current ~= 0.065 mean condition MAE" comparison mixes different
  sample sizes.

## 2026-06-10 audit: confounds, variance, and reproduction

- The final-vs-binned comparison is confounded by training data: ablations were
  trained on binned `diff_bins/*/*.h5`, the final model on the original sparse
  distribution, and ALL are evaluated on binned-style generated conditions. The
  binned models have a home-distribution advantage. The conclusion "final didn't
  reproduce the binned performance" cannot be attributed to the objective choices
  as designed.

- Single seed only: training fixes `RANDOM_SEED = 68` (`cli.py`), so each variant
  is one run. The top four fine-tune variants span 0.02665-0.02721 mean MAE,
  inside plausible seed noise. Need >=3 seeds or bootstrap CIs before presenting
  any ranking as a result.

- The high-D compression is real and was reproduced from the original: in
  `diff_evals/model_comparison_*`, target D=0.45 gives mean predicted 0.361
  (reproduced) vs 0.370 (Original ti2microscope). Both share the bias, so it is
  inherent to SPTnet, not introduced by the refactor. This is a genuine
  reproduction finding for the report's "Comparison to OG model" section.

- Possible mechanism for high-D compression worth testing: the D loss is divided
  by a CRLB ratio (`losses.py`, ~line 196). If CRLB grows with D, this
  systematically down-weights high-D errors during training, which could cause or
  worsen the compression. A loss-side fix may beat or complement data rebalancing.

- The planned bias fine-tune has a metric trap: because the eval scores against
  the condition mean, a fine-tune that merely inflates high-D predictions toward
  0.45 will improve every headline metric with no per-track gain. Rebuild the
  metric before running it, and judge success on per-track high-D error plus
  retained detection/tracking.

- No quantitative real-data result exists yet (Aim 3): `RealData/` has tiles, a
  TrackMate XML, and segmentation manifests, but no stitched-track output or
  metrics. No runtime-optimisation timing table exists yet either (Aim 2),
  despite the AMP/TF32/vectorisation claims; SLURM `*_metrics.txt` files hold the
  wall times needed to build one.

## 2026-06-10: matched per-track evaluation built and run

- Ground-truth label files were copied from CSD3 to
  `diff_evals/ground_truths/<condition>/trainingvideos_N_gt.h5` (labels only, no
  video pixels). Schema is the MATLAB cell-array layout read by
  `TransformerMatDataset`: `Clabel`/`Hlabel`/`traceposition`/`duration`/
  `moleculeid` are `(10 particles, 100 videos)` HDF5-reference arrays. `Clabel`
  is real D (not normalised), `traceposition` is `(2, 30)` centered pixels with
  NaN on inactive frames. Counts match the result files exactly (100/100/20x5).

- Inference used `SPT_HDF5_CLIP_INDEX=0`, so each `result_trainingvideos_N.h5`
  corresponds to video column 0 of `trainingvideos_N_gt.h5`. Only 1 of every 100
  generated videos per file was actually evaluated.

- Coordinate frames: predicted `estimation_xy` is normalised to [-1, 1];
  GT positions are centered pixels and are normalised for matching by dividing by
  `image_size/2` (=32 for 64px), matching the training convention
  (`cli.py`: `position_label / (image_size/2)`).

- New module `notebooks/diffusion_eval_matched.py` does the correct evaluation:
  loads GT, Hungarian-matches confident predicted queries to true particles by
  mean trajectory distance over overlapping active frames (gate default 0.25
  normalised ~= 8px, min overlap 3 frames), and reports per-track `|pred_D -
  true_D|`, signed bias, per-track H error, localisation RMSE, detection
  precision/recall/false-positives/missed, true-D calibration slope/corr, the
  per-track oracle floor, and the OLD mae-to-target for comparison. Writes
  `*_matched_{tracks,summary,ranking}.csv` next to the old plots.

- Corrected results (per-track MAE_D; old metric in parentheses; gate 0.25):
  - Scratch: H/D off match 0.0365 (0.0257), Log H/D 0.0397 (0.0322),
    Baseline 0.1006 (0.0913), BCE logits 0.1019 (0.0879). The winning order is
    preserved, but the old "best" 0.0257 was BELOW the oracle floor (~0.029) and
    is now correctly above it once shrinkage is removed. Baseline/BCE "poor D" is
    really detection collapse: precision/recall ~0.42/0.12 and ~0.27/0.10 vs
    ~0.93/0.93 for the good runs.
  - Finetune: all five variants land at 0.0329-0.0342 per-track MAE_D with
    precision/recall ~0.92-0.95 and the oracle floor at ~0.029. The ranking
    REORDERS versus the old metric (old best BCE logits; new best Baseline by
    <0.001). This is direct proof the fine-tune ablation differences are
    single-seed noise and must not be presented as a result.
  - Final full model: per-track MAE_D 0.0648 but precision/recall only ~0.29 and
    loc_rmse ~0.153 (~4.9px) versus ~0.045 (~1.4px) for the binned models. It
    matched only 872 true tracks; ~2000 of its "confident" predictions in the old
    eval were false positives the old metric silently scored against the mean.

- The corrected final-model finding reframes the project narrative: the final
  full model's gap on the binned-style evaluation is dominated by a
  detection/localisation failure, not purely diffusion-head miscalibration. This
  is the training-distribution confound made concrete (sparse-trained model vs a
  10-particle binned-style eval that the binned models were trained on). High-D
  compression is still real and gate-robust (signed high-D bias ~-0.08 across
  gates 0.15-0.60), but it is secondary to the detection gap.

- Implication for the bias fine-tune: fine-tuning the final model on binned data
  will likely improve detection AND reduce high-D bias together, but most of the
  gain will be distribution adaptation, not evidence about the loss objective.
  Report it as such, and evaluate with the matched metric.

- Three-model matched comparison (final cell of `notebooks/testing.ipynb`,
  `M.collect_models` over Original ti2 / Dense-sparse `inference` / Final full
  model): all three are statistically indistinguishable on true per-track error
  (per-track MAE_D 0.0644 / 0.0658 / 0.0648), all collapse to ~0.27-0.29
  precision/recall and ~4.9px localisation on the dense (10-particle) eval, and
  all show high-D compression (signed bias -0.060 / -0.065 / -0.083). The new
  "final full model" is marginally the WORST on high-D bias and calibration slope
  (0.71 vs 0.78 for ti2), i.e. the combined-loss "final" model is not an
  improvement. The `inference` dir under `diff_evals/generated` is the
  dense_sparse model (`../SPTnet/Trained_models/dense_sparse/trained_model`);
  `inference_ti2microscope` is the original ti2 model.

## 2026-06-10: project pivot to speed + packaging headline

- Decision: stop experimenting; consolidate. The report headline is now (1) the
  ~12x training-speed improvement of `opt-SPTnet` over `../SPTnet`, and (2) the
  refactor into an installable, tested, documented, MATLAB-free package with
  CLIs. The diffusion work is a secondary critical-evaluation chapter (faithful
  reproduction of the inherent high-D bias + the corrected per-track evaluation
  that exposed the earlier "improvements" as metric artifacts). The binned bias
  fine-tune is shelved. Rationale: writing is the bottleneck with ~3 weeks left,
  and the speed/usability contribution plus an honest evaluation is a stronger,
  more defensible MPhil than chasing a confounded diffusion win.

## 2026-06-11: binned fine-tune re-run correctly — strong positive result

- The earlier fine-tune that "did not improve" was a spurious early stop: the
  early-stopping baseline `min_v_loss` is inherited from the resumed loss-history
  CSV (`cli.py:419-420`), which held the final model's best val (0.154) from its
  original (different-distribution) training. The binned fine-tune val loss
  (0.186 -> 0.160, still descending) never beat that stale cross-distribution
  baseline, so patience 5 tripped while the model was actively improving.
  Re-running with a fresh loss history fixed this.

- Results live in `diff_evals/final/diff_evals/final_full_model_ft/` and are now
  the 4th model in the final cell of `notebooks/testing.ipynb`
  (`diff_evals/final/comparison_plots/ft_matched_*`). Matched per-track metrics
  (gate 0.25), vs the others:
  - Final model FT: per-track MAE_D 0.0373, mean bias -0.012, high-D MAE 0.047,
    high-D bias -0.034, precision 0.95, recall 0.92, loc_rmse 0.035, slope 0.84,
    corr 0.95, 2781 matched tracks.
  - Final full model: 0.0648 / -0.011 / 0.091 / -0.115 / 0.29 / 0.28 / 0.153 /
    0.71 / 0.79 / 872 tracks.
  - Original ti2: 0.0644 / ... / high-D bias -0.089 / recall 0.23.
  - At D=0.45: FT predicts mean 0.401 (bias -0.048, recall 0.92) vs final 0.342
    (bias -0.115, recall 0.23) and ti2 0.368 (bias -0.089, recall 0.23).

- Interpretation: the fine-tune roughly HALVED per-track diffusion error, cut
  high-D compression by ~60-70%, and fixed the detection/localisation collapse
  (recall 0.28 -> 0.92, loc_rmse 0.153 -> 0.035). BUT the eval distribution IS
  the fine-tune's training distribution (binned, 10 particles), so most of the
  detection/localisation jump is distribution adaptation, not evidence about the
  loss objective. The KEY UNTESTED question is catastrophic forgetting: a matched
  eval on the general/sparse distribution is still needed before claiming the FT
  model is a better general tracker. As-is it is a strong "targeted adaptation
  recovers target-distribution performance" result, not a free lunch.

- This reopens whether the fine-tune belongs in the report. It is a genuine
  positive extension result (diagnose -> fix), so it is worth an appendix or a
  short subsection IF the forgetting check is done; otherwise present it as a
  distribution-adaptation demonstration with that caveat explicit.

## 2026-06-11: forgetting test (held-out sparse general distribution)

- Data: `diff_evals/forget/` with GT in `diff_evals/forget/gt/gt/trainingvideos_N.h5`
  (flat layout, no `_gt` suffix), 28 files x 100 videos, seed 70123, sparse
  1-10 particles (mean 5.77/video). Inference clip 0 -> 28 videos, ~161 GT
  particles. Two models: `final_full/inference` (pre-FT final) and
  `final_full_ft/inference` (the binned fine-tune). Matched eval, gate 0.25.
- Results on the sparse set (paired on the same 28 videos):
  - Final model FT: recall 0.95, precision 0.94, loc_rmse 0.023 (~0.7px),
    per-track MAE_D 0.096, slope 0.27, corr 0.49, MAE_H 0.13, 153 matched.
  - Final full model: recall 0.22, precision 0.22, loc_rmse 0.143 (~4.6px),
    per-track MAE_D 0.072, slope 0.63, corr 0.69, MAE_H 0.17, only 36 matched.
- Interpretation (two clear, opposite signals):
  1. DETECTION/LOCALISATION: NOT forgotten -- the FT model is far better on the
     sparse set too (recall 0.95 vs 0.22, 0.7px vs 4.6px). The big detection gap
     appears on BOTH binned and sparse evals for all the "full" models
     (ti2/dense-sparse/final all ~0.22-0.28 recall), so it is not a
     particle-density effect; the fine-tune broadly improved detection. (Open
     puzzle: why binned fine-tuning improves sparse detection so much -- likely
     the final model was undertrained on detection, or a train/eval
     generator-domain effect. Note, do not over-claim a mechanism.)
  2. DIFFUSION CALIBRATION: specialised to the binned distribution and does NOT
     transfer -- FT slope collapses to 0.27 on sparse (vs 0.84 on binned),
     i.e. near-flat D predictions. So the high-D bias "fix" is
     distribution-specific overfitting of the regression head, the real
     forgetting signal.
- Caveats: small/low-power -- final_full matched only 36 particles, so its
  sparse slope/MAE rest on a tiny self-selected subset; and the per-track MAE
  comparison (0.096 FT vs 0.072 final) is confounded by detection selection bias
  (FT scores over 153 incl. hard particles, final over 36 easy ones). The
  cleaner, less-confounded signal is the calibration slope. A fairer follow-up:
  compute D error only on GT particles BOTH models detect (requires storing the
  GT particle index per match in `match_video`).
- Report framing: the fine-tune improved detection/localisation broadly but
  specialised diffusion calibration to the binned distribution (clear trade-off
  on the regression head). Honest "targeted adaptation with a transfer cost",
  not a uniformly better model.

- [SUPERSEDED 2026-06-15 — see the "benchmark audit + rebuilt harness" entry
  below: the "12x" is an artifact (~4.2x is the honest per-iteration figure), and
  DataLoader workers are NOT a differentiator since they exist in both repos.]
  The 12x speed claim currently has NO committed in-repo evidence. The benchmark
  harness exists (`slurm/train_sptnet_benchmark_csd3.slurm`): it runs
  `sptnet-train` under `/usr/bin/time -v`, parses the script's "Training takes N
  s" line into `training_seconds`, and records total wall time and a
  startup+data-loading estimate to `<model_dir>/slurm_benchmark_*_metrics.txt`
  (on CSD3, not committed). To defend the headline: run the ORIGINAL `../SPTnet`
  training and `opt-SPTnet` on the SAME data/epochs/hardware, commit a small
  benchmark table (suggest `experiments/benchmarks/`), and decompose the speedup
  into its sources (vectorised per-sample normalisation, AMP/TF32, DataLoader
  workers/pinned memory/persistent workers, removed per-epoch plotting/debug,
  single-pass (T,H,W) inference, flat ConcatDataset). Repeat a few times and
  report mean/spread; state the exact comparison conditions in Methods.

## 2026-06-15: benchmark audit + rebuilt statistically-backed harness

- AUDIT of the committed runs under `experiments/benchmarks/`: the "12x" was an
  artifact and is not defensible.
  - `standard_new` (opt) DIVERGED to NaN: no grad-clip in that run, first NaN at
    batch 199 of epoch 1, 3297 skipped batches, training-loss components 0.0 from
    epoch 2 and validation frozen at 1.9208. Early-stopping then quit at 8 epochs.
    It is not a valid trained model and must not be used as evidence.
  - `standard_old` is healthy (22 epochs, v_loss 1.61 -> 1.17).
  - The "12x" = training_seconds 10585/852, but that divides totals over DIFFERENT
    epoch counts (22 vs 8) caused by the broken new run early-stopping. Not
    like-for-like.
  - Honest signal = per-iteration throughput: both healthy optimized runs sit at
    ~4.9 it/s vs old ~1.15 it/s => ~4.2x. `sptnet_final_30293619.out` (healthy,
    full TrainData=100k samples, grad-clip 1.0, 4.88 it/s) corroborates this but
    is NOT a clean head-to-head with standard_old (10x more data, different loss
    config), so it supports the per-iteration figure only.
  - DataLoader workers/pin_memory/persistent_workers are ALREADY in BOTH repos
    (`src/sptnet/data/loaders.py`; `../SPTnet/SPTnet_toolbox.py`), so the speedup
    isolates compute (AMP/TF32/cudnn.benchmark) + removed per-epoch plotting, not
    data loading.

- REBUILT HARNESS (committed this repo):
  - `src/sptnet/training/cli.py`: writes per-run `epoch_timing.csv`
    (`epoch,train_seconds,val_seconds,n_train_batches,n_val_batches,amp,tf32,cudnn_benchmark`,
    timed with `torch.cuda.synchronize()` around train and val passes), plus
    env-gated toggles `SPT_DISABLE_AMP`, `SPT_DISABLE_TF32`, `SPT_CUDNN_BENCHMARK`
    (defaults preserve normal behaviour). Verified: TF32 toggle flips the torch
    backend at import; `pytest` green.
  - `slurm/train_sptnet_benchmark_csd3.slurm`: `SPT_SYSTEM=old|new` switch,
    `#SBATCH --array` per-run dirs (`<RUN_NAME>/run_NN/`), toggle pass-through,
    benchmark defaults (max-epochs 4, patience 99, grad-clip 1.0, max_files 100).
  - `experiments/benchmarks/analyze_benchmarks.py`: hierarchical bootstrap
    (resample runs -> epochs, drop epoch 1 warmup), emits `results/summary.csv`,
    `results/summary.md` (headline old/new speedup + 95% CI + decomposition), and
    `results/per_epoch_times.png`. Smoke-tested on synthetic data.
  - `experiments/benchmarks/README.md`: full protocol, run matrix, submit
    commands, and the old-script patch instructions.

- REMAINING (on CSD3): apply the same per-epoch `epoch_timing.csv` emitter +
  `SPT_DISABLE_PLOT` toggle to `../SPTnet/SPTnet_training_old_cli.py` (the old
  repo and part of the data live on CSD3; a Codex prompt with the exact edits was
  handed off). Then submit K=10 headline arrays (`new_full`, `old`) and K=3
  decomposition arrays (`new_no_amp`, `new_no_tf32`, `new_no_cudnn_bench`,
  `old_no_plot`), run the analyzer, and commit `results/` + per-run
  `epoch_timing.csv`/`*_metrics.txt`. Point BOTH systems at the SAME data files
  (absolute paths) since data is split across the two repos.

## 2026-06-15: benchmark RESULTS (4.33x confirmed) + decomposition dropped

- Ran the headline arms on CSD3 (8 runs each, not 10 — variance is tiny, 8 is
  ample). Results committed under `experiments/benchmarks/new_fresh_good_runs/`
  and `old_good_8_runs/` (per-run `run_NN/epoch_timing.csv` + `*_metrics.txt` +
  logs), analysed by `analyze_benchmarks.py` into
  `experiments/benchmarks/results/summary_{train_seconds,epoch_total_seconds}.{csv,md}`
  and `per_epoch_*.png`. 4 epochs/run, epoch 1 dropped as warmup => 24 steady
  epochs/system; hierarchical bootstrap, 10k resamples.
- Headline (steady-state per-epoch TRAINING time): new 102.8 s/epoch vs old
  445.6 => **4.33x, 95% CI [4.28, 4.39]**.
- End-to-end (`epoch_total_seconds`, incl. plotting/logging/checkpointing):
  new 116.3 vs old 511.7 => 4.40x [4.34, 4.46]. Validation loop: 5.35x.
  Whole-loop `training_seconds` from metrics corroborates (~475 vs ~2022).
- Very tight per-run spread: new train per-run means [101.1, 103.8]s, old
  [437.8, 456.9]s. Warmup epoch 1 modestly slower (new 107.6 vs 102.8 steady;
  old 449.4 vs 445.5), confirming the drop.
- DECISION: decomposition arms (new_no_amp/new_no_tf32/new_no_cudnn_bench/
  old_no_plot) NOT run — judged low value for the GPU time. The 4.33x is a TOTAL
  speedup; not attributed to AMP vs TF32 vs cudnn individually. Harness/toggles
  remain if ever revisited.
- Gotcha: the old `*_metrics.txt` echoes `disable_amp=0, cudnn_benchmark=1` —
  those are SLURM env DEFAULTS, not the old script's behaviour. The
  `epoch_timing.csv` columns are authoritative and correctly record
  amp=0/tf32=0/cudnn_benchmark=0 for old.

## 2026-06-15: Results section outline (Runtime Optimisation + the rest)

Outline agreed for the report Results chapter (4 sections already stubbed in
`report/main.tex`):

1. Runtime Optimisation (centrepiece):
   - Restate the benchmark protocol in one line (cross-ref Methods
     `sec:benchmark-protocol`); state the fixed workload and that it's wall-time,
     not convergence.
   - Headline: 4.33x per-epoch training speedup [4.28, 4.39], table of new vs old
     mean s/epoch (train, val, full-epoch) with CIs.
   - Figure: `per_epoch_train_seconds.png` (per-config distribution) showing tight
     within/between-run spread => the estimate is stable on 8 runs.
   - Sentence on end-to-end 4.40x and validation 5.35x; note decomposition not run
     so the number is total (attribute qualitatively to AMP/TF32/cudnn + removed
     plotting, with workers already in both).
   - Separate short paragraph: the input-normalization fix (train/inference
     mismatch) => new converges to lower loss; explicitly NOT folded into the
     speed number.
2. Comparison to OG model (reproduction): three-model matched per-track
   comparison (Original ti2 / dense-sparse / final), equivalence + shared high-D
   compression; quantify agreement. (Source: diff_evals/final matched CSVs.)
3. Mean/Variance sweeps: matched per-track D calibration across the mean sweep
   (0.05-0.45) and range sweep (+/-0.01,0.05,0.15); slope/corr, high-D bias.
4. (Optional) Stitching/real-data qualitative demo OR explicit out-of-scope note.
