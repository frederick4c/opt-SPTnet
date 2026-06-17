# Report Artifacts Checklist

Tracks the data/logs and figures the report depends on, and where each lives.

## Repo context

Two repositories:

- **Public GitHub** (`opt-SPTnet`): the usable tool. Data, results, notebooks,
  and `report/` are deliberately `.gitignore`d (`*.csv`, `*.h5`, `*.pt`,
  `RealData/`, `diff_evals/`, `notebooks/`, `report/`). Do NOT relax this just to
  store result artifacts — it should stay a clean codebase.
- **GitLab submission**: includes the report source AND the required data/logs so
  the thesis results are reproducible by the assessor.

So the lists below are about what must be present in the **submission repo**, not
about un-ignoring files on GitHub. Items already on CSD3 must be copied down.
Large raw inputs (movies, full GT/inference `.h5`, per-track CSVs) can be cited by
path in Methods rather than shipped, unless the assessor needs to re-run.

Status legend: ✅ present locally · ⬇️ on CSD3, copy down · 🔧 needs
(re)generating · 📁 large, decide whether to ship or cite.

## 1. Data / logs by report section

### A. Runtime benchmark (Results: Runtime optimisation — headline)
- ✅ **Repeated benchmark runs (8× each, old & new)** — DONE 2026-06-15. Evidence
  in `experiments/benchmarks/{new_fresh_good_runs,old_good_8_runs}/` with analyzer
  outputs in `experiments/benchmarks/results/` (`summary_*.{csv,md}`,
  `per_epoch_*.png`). Harness: `experiments/benchmarks/analyze_benchmarks.py`.
- ✅ Per-epoch `epoch_timing.csv` per run (train/val/epoch_total seconds + env
  toggles) — the authoritative timing source.
- ❌ Decomposition / ablation arms (no-AMP / no-TF32 / no-cudnn / old-no-plot)
  DELIBERATELY NOT RUN (2026-06-15 decision) — speedup reported as a TOTAL. Do not
  reintroduce as a to-do.

Settled numbers (for `tab:runtime-benchmark`, already in `main.tex`):
steady-state per-epoch training 445.5 s → 102.8 s = **4.33× [4.28, 4.39]**
(hierarchical bootstrap); validation 5.35×; full epoch 511.7 s → 116.3 s = 4.40×
[4.34, 4.46]. The old "12.4×"/"9.8×"/"4.2×" figures are SUPERSEDED — see
`[[benchmark-speedup-corrected]]`, `context/plan.md` step 1, and the 2026-06-15
notes. The earlier `standard_{new,old}` single runs are provisional/`standard_new`
is broken (NaN); do not cite them.

### B. Reproduction (Results: Reproduction & inherent high-D bias)
- ✅ `diff_evals/final/comparison_plots/three_model_matched_{ranking,summary}.csv`.
- 📁 GT labels + inference results these were computed from
  (`diff_evals/ground_truths/`, `diff_evals/final/.../inference_results/`) —
  large; cite path in Methods unless re-run is required.

### C. Corrected evaluation (Results: Critical evaluation of diffusion estimation)
- ✅ `diff_evals/scratch/comparison_plots/scratch_matched_{ranking,summary}.csv`.
- ✅ `diff_evals/finetune/comparison_plots/finetune_matched_{ranking,summary}.csv`.

### D. Fine-tune + forgetting (Results: Targeted fine-tuning & transfer cost)
- ✅ `diff_evals/final/comparison_plots/ft_matched_{ranking,summary}.csv`.
- 🔧 **Forgetting-test matched CSV** — the sparse-set numbers (recall 0.95 vs
  0.22, slope 0.27 vs 0.84) currently exist ONLY in `context/notes.md`. Regenerate
  the matched ranking/summary on `diff_evals/forget/` and save a CSV so the
  fine-tune transfer-cost claim has a committed source.

### E. Real data (Results: Real microscopy data — qualitative)
- ✅ `RealData/full_model_ft/stitched_tracks.csv`.
- ✅ `RealData/**/*_segmentation_manifest.csv`.
- 📁 `RealData/full_realdata.{h5,tif}` (~4.5 GB) — too large to ship; reference
  only, or include a small clip/preview.

### F. Code that generates the results (reproducibility)
These live under `notebooks/` (GitHub-ignored) and must be in the submission so
results are reproducible:
- `notebooks/diffusion_eval_matched.py` — the corrected matched metric.
- `notebooks/diffusion_eval.py`, `notebooks/finetune_diffusion_plots.py`.
- `notebooks/testing.ipynb` — final three/four-model comparison (or export to a
  script).
- `report/` — the LaTeX source itself (GitHub-ignored).

## 2. Initial figure list

To be saved into `report/figures/` (the rough plots under
`diff_evals/**/comparison_plots/` are diagnostics, not report figures — redraw).

### Methods
- **M1 Pipeline overview** — ✅ DONE:
  generate → train → infer → segment/stitch.
- **M2 SPTnet architecture** — ✅ DONE: (may need checking at end)
  backbone + dual 2D/3D transformer branches + query heads.
- **M3 Synthetic data reproduction** — ✅ DONE:
  `report/figures/data_generation_comparison.pdf` (MATLAB original vs Python
  render from the same tracks). Source: last cell of `notebooks/figs.ipynb`.
  Optional extra: a frame with GT trajectories overlaid (PSF + Perlin + Poisson).

### Results
- **R1 Runtime benchmark** — ✅ DONE: per-epoch training-time distribution
  (`report/figures/benchmark_per_epoch_train.pdf`), pairs with
  `tab:runtime-benchmark`. No decomposition panel (not run).
- ⚠️ **EVAL BUG (2026-06-17): R2 and R4 below are INVALID pending re-eval.** The
  matched CSVs scored the (y,x) models (Original, Dense/sparse, Full) with
  transposed coords; their recall/loc-RMSE/bias are artifacts. See
  `context/rerun_matched_evals_csd3.md` + 2026-06-17 notes. Do NOT use R2/R4 until
  the CSVs are regenerated; then rebuild both figures.
- **R2 Reproduction bias** — ⚠️ REDO (Original/Full boxes invalid; FT box ok):
  `report/figures/reproduction_bias.pdf` (binned mean±SEM signed bias D̂−D vs true D
  + zero line; slope+recall in legend). REPORT MODELS = Original (`Original ti2`),
  Full model (`Final full model`), Full model fine-tuned (`Final model FT`) — NOT
  the old Original/Dense-sparse/Final trio. Source: `ft_matched_{tracks,ranking}.csv`
  (has the FT model), cell in `notebooks/figs.ipynb`. NB: Original≈Full (repro), FT
  much better ON ITS BINNED DISTRIBUTION (recall 0.92 vs ~0.28) — text must carry
  the forgetting caveat (FT forgets on the general sparse distribution).
- **R3 Metric artifact** — ❌ DROPPED as a figure (2026-06-16). The scatter was
  weak/decision-irrelevant: on converged models only 3/28 points dip below the
  oracle floor (25/28 above → it argues the OLD metric is mostly fine). The dramatic
  below-floor cases were the collapsed scratch runs, now shelved. The metric
  correction is already a METHODS point (\cref{sec:eval-protocol} defines the oracle
  floor); make it in PROSE, optionally a tiny inline number/table, NOT a figure.
- **R4 Detection bottleneck** — ⚠️ INVALID, REDO (Original/Full recall+loc-RMSE are
  xy/yx artifacts; corrected, all models ~0.92 recall → the "bottleneck" disappears):
  `report/figures/detection_bottleneck.pdf`
  (2-panel: detection recall + localisation RMSE vs target D, mean sweep, 3 report
  models). STRONG/decision-supporting: Original & Full overlap at recall ~0.3 /
  loc_rmse ~0.15; Finetuned model recall ~0.92 / loc_rmse ~0.03. Shows the limit is
  detection/localisation under distribution shift, not the diffusion head. Source:
  `ft_matched_summary.csv` (half_width==0.05), cell in `figs.ipynb`.
- **Fine-tune ablations** — ❌ DROPPED (2026-06-16). The 5 fine-tune variants are
  near-indistinguishable (MAE_D ~0.033, recall ~0.92), so the bar chart showed 5
  equal bars — no information. Make the "objective choice barely matters from a
  pretrained start" point in PROSE if needed, not a figure.
- **Variance sweep** — ✅ DONE: `report/figures/diffusion_variance.pdf` (single
  panel, fine-tuned model at D=0.25: true vs predicted spread across half-widths
  0.01/0.05/0.15; predicted spread ~fixed → over-disperses narrow, under-disperses
  wide → follows the MEAN but not the VARIANCE). Mean-following deliberately NOT
  re-plotted (already in `reproduction_bias.pdf`). Source: `ft_matched_summary.csv`,
  cell in `figs.ipynb`.
- **R5 Fine-tune & transfer cost** — 🔧 BLOCKED on data: needs the matched eval
  REGENERATED on `diff_evals/forget/` (has `final_full`, `final_full_ft`, `gt` but
  NO `*_matched_*` CSV). Not a plot-only task — must run the matched metric first.
- **R6 Real-data** — ✅ MOSTLY DONE (annotated-frame demo + aggregate, NOT a
  track-density map). Real-data section = (1) SUCCESS: `figures/ft_success.png`
  (FT works on bright real data, returns H/D TrackMate doesn't); (2) LIMITATION on
  dim KDEL: `figures/ft_fail.png` (illustrative single frame, FT boxes on flat
  noise) + `figures/realdata_overdetection.pdf` (QUANTITATIVE backbone: FT median
  806 detections/frame vs Dense/sparse 3 = ~270x across the whole KDEL movie; cell
  in `figs.ipynb`). `pre_ft_real.png` is the non-FT/sparse model on KDEL (looser).
  FRAMING: SNR/objectness-robustness limitation (FT over-detects high-confidence on
  low-contrast data, not threshold-fixable), NOT forgetting; qualitative, no GT.
  NO matched single FT-vs-nonFT frame exists (checked) — the ~270x AGGREGATE is the
  cherry-pick-proof claim instead. Screenshots still need: scale bar, box-colour
  legend, label red × as TrackMate. See [[realdata-ft-model-distribution-shift]].

Tables scaffolded for the draft: benchmark (R1), three-model (R2), fine-tune (R5).
