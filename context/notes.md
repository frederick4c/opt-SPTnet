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
  scratch ablation layouts via `--results-root` and `--plot-prefix`. The final
  cell of `notebooks/testing.ipynb` runs the scratch comparison and embeds the
  ranking table plus plots.

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
