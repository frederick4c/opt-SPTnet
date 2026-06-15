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
- ✅ `experiments/benchmarks/standard_{new,old}/slurm_benchmark_*_metrics.txt`,
  `*.log`, `trained_modeltraining_log.txt` — already in repo (tracked).
- ✅ `experiments/benchmarks/standard_new/trained_modelloss_history.csv`.
- ⬇️ **Repeated benchmark runs (2–3× each, old & new)** — needed for mean/spread
  on the headline claim. Currently single-run only. (User uploading.)
- 🔧 Optional ablation runs (AMP off, workers=0, plotting on) to decompose the
  speedup into named sources.

Current single-run numbers (provisional, for `tab:benchmark`):
training 10585 s → 852 s (12.4×); total wall 10781 s → 1105 s (9.8×); fixed
startup + data-loading ≈ 200–250 s. NOTE: reconcile `context/context.md` which
still says 4.2×.

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
- **M1 Pipeline overview** — generate → train → infer → segment/stitch.
  Fills TODO at `report/main.tex` pipeline-overview section.
- **M2 SPTnet architecture** — backbone + dual 2D/3D transformer branches +
  query heads. Fills TODO in the architecture section.
- **M3 (optional) Synthetic data example** — a frame with GT trajectories
  overlaid; illustrates PSF + Perlin background + Poisson noise.

### Results
- **R1 Runtime benchmark** — bar chart, training vs total-wall, original vs
  opt-SPTnet (+ optional decomposition). Pairs with `tab:benchmark`. Awaits
  repeated-run upload.
- **R2 Reproduction calibration** — predicted vs true per-track D for the three
  models; agreement + shared high-D compression. Source: `three_model_matched_*`.
- **R3 Metric artifact** — old mean-target score vs oracle floor vs corrected
  per-track error; the visual core of the honesty argument.
- **R4 Detection bottleneck** — precision/recall + localisation RMSE across
  models (why the gap isn't the diffusion head).
- **R5 Fine-tune & transfer cost** — before/after on binned distribution +
  calibration-slope collapse on sparse (forgetting). Source: `ft_matched_*` and
  the forgetting CSV (item D).
- **R6 Real-data overlay** — stitched tracks on the experimental movie
  (qualitative end-to-end demo). Source: `RealData/full_model_ft/stitched_tracks.csv`.

Tables scaffolded for the draft: benchmark (R1), three-model (R2), fine-tune (R5).
