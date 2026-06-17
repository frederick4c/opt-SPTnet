# Project Plan

Use this file for next steps, active tasks, and handoff plans for future coding
agents.

## Status and focus (set 2026-06-10)

The project is OUT of experimentation and INTO consolidation. No new model
experiments. The goal is a watertight, cohesive, reproducible project and a
finished 7000-word report in `report/` (due 1 July 2026).

Report headline (primary): the ~4x training-speed improvement of `opt-SPTnet`
over the original `../SPTnet`, and the refactor into an installable, tested,
documented, MATLAB-free Python package with CLIs. This is the
reproducibility-and-usability contribution.

Report secondary: a critical-evaluation chapter on the diffusion work. Its value
is honesty, not a performance win: SPTnet's high-D compression is inherent
(reproduced from the original ti2 model), and a corrected per-track evaluation
(built 2026-06-10, see `notebooks/diffusion_eval_matched.py`) showed the earlier
diffusion-ablation "improvements" were largely metric artifacts, with the true
bottleneck being detection/localisation under distribution shift.

Shelved: the binned bias fine-tune. It was judged not worth the time given the
writing workload; the diagnosis stands on its own. If revisited, it would need
the matched metric and a cross-distribution (binned + general) eval to test
catastrophic forgetting, and would be appendix-only. Do not start it without an
explicit decision to.

## Next steps (consolidation, ordered)

1. [DONE 2026-06-15] Speed claim substantiated with a committed, controlled,
   BOOTSTRAPPED benchmark. Headline result: **4.33x** steady-state per-epoch
   TRAINING speedup, 95% CI **[4.28, 4.39]** (old 445.6 vs new 102.8 s/epoch).
   End-to-end (full epoch incl. plotting/logging/ckpt) **4.40x [4.34, 4.46]**;
   validation loop 5.35x. This replaces the bogus "12x" for good.
   - Evidence: `experiments/benchmarks/new_fresh_good_runs/` and
     `old_good_8_runs/` (8 runs each, not 10 — variance is tiny so 8 is ample;
     per-run new train means [101.1, 103.8]s, old [437.8, 456.9]s). Analyzer
     outputs in `experiments/benchmarks/results/summary_{train_seconds,
     epoch_total_seconds}.{csv,md}` + `per_epoch_*.png`.
   - Protocol: steady-state per-epoch training time, epoch 1 dropped as warmup,
     4 epochs, identical fixed workload (100-file subset, 500 train/125 val
     batches, batch 16, query 20, lr 1e-4, max_dc 0.5), CSD3 Ampere, workers=2,
     new with default objective matching the original loss. Hierarchical bootstrap
     (resample runs -> epochs), 10k resamples. Methods benchmark-protocol section
     of `report/main.tex` already describes this.
   - DECISION (2026-06-15): the decomposition arms (new_no_amp / new_no_tf32 /
     new_no_cudnn_bench / old_no_plot) are NOT being run — judged low value for
     the extra GPU time. So the ~4.33x is reported as a TOTAL speedup, not
     attributed to AMP vs TF32 vs cudnn individually. The toggles/harness remain
     in place if ever revisited. DataLoader workers are already in BOTH repos, so
     the gain is compute + removed plotting/overhead, not data loading.
   - Tooling (committed): instrumented `src/sptnet/training/cli.py` (per-epoch
     `epoch_timing.csv` with train/val/epoch_total seconds + env toggles),
     generalised `slurm/train_sptnet_benchmark_csd3.slurm`,
     `experiments/benchmarks/analyze_benchmarks.py`, `README.md`.
   - Caveats: the old `*_metrics.txt` env echo (disable_amp=0, cudnn_benchmark=1)
     reflects SLURM defaults, NOT the old script — the `epoch_timing.csv` columns
     (amp=0,tf32=0,cudnn_benchmark=0 for old) are authoritative. The old
     `standard_new` run is BROKEN (NaN) and must not be used. The new model also
     converges to a lower loss (input-normalization fix, see
     `[[train-inference-normalization-fix]]`) — irrelevant to timing; report
     separately, never as "Nx faster AND lower loss".
   - REMAINING: write the Results "Runtime Optimisation" section around 4.33x
     (table + per-epoch distribution figure); see the results outline in the
     2026-06-15 notes entry.

2. Write the report around the headline. Fill, in order:
   - Methods: keep the drafted pipeline/architecture/loss; ADD an evaluation-
     protocol subsection (the D_mean conditions, the obj>=0.5 / >=3-frame
     confident filter, the matched per-track metric) and a benchmark-protocol
     subsection (data, epochs, hardware, what is timed).
   - Results "Runtime Optimisation": the benchmark table/figure + the speedup
     decomposition. This is the centrepiece.
   - Results "Comparison to OG model": reproduction. Use the matched
     three-model comparison (Original ti2 vs Dense/sparse vs Final;
     `diff_evals/final/comparison_plots/three_model_matched_*`), showing the
     models are equivalent per-track and all share the inherent high-D
     compression. Quantify agreement with the original.
   - Results critical-evaluation: the metric flaw and its correction (old
     mae-to-target rewards mean-shrinkage; one old "best" sat below the oracle
     floor), and the detection-bottleneck finding under distribution shift.
   - Abstract and Conclusion (currently empty/placeholder).
   - Resolve Methods placeholders: "TODO/SPARE" section, overview figure,
     architecture figure.

3. Make the repo reproducible and cohesive (supports the usability claim and is
   itself evidence for the report):
   - Ensure `pip install -e .` + the documented CLIs reproduce each workflow
     stage from a clean checkout; fix any drift.
   - Confirm the test suite passes locally (`pytest`) and the docs build.
   - Tidy `experiments/` vs core `src/sptnet`: keep the package clean; clearly
     mark one-off experiment scripts and SLURM variants as such.
   - Make sure every report figure/table maps to a committed source artifact or
     a notebook cell. Current sources: the new matched CSVs/plots under each
     `diff_evals/*/comparison_plots/`, the three-model matched comparison in the
     final cell of `notebooks/testing.ipynb`, and the (to-be-committed)
     benchmark table.

4. Scope the remaining aims honestly:
   - Real data (Aim 3): only `RealData/` tiles, a TrackMate XML, and
     segmentation manifests exist; no quantitative result. Either add a short
     qualitative real-data demonstration of the end-to-end package (segment ->
     infer -> stitch on the real movie) to support the usability story, or state
     explicitly that real-data evaluation is out of scope and why.
   - Diffusion-teacher: keep as a side experiment, omit from the main narrative
     unless it earns a place; an appendix mention is the most it merits.

5. Polish: verify forward-dated refs are real (`fidltrack2026`, `fan2025dblur`,
   `anchoredbm2026`, `anomalousnet2025`); fix the dangling "T" and "aend" typos
   in `main.tex`; confirm the title-page date.

- MAIN TASKS NOW, IN ORDER: (1) [DONE] benchmark run + analysed (4.33x, see
  above); now WRITE the Results "Runtime Optimisation" section around it,
  (2) WRITE THE REST OF THE REPORT AROUND SPEED + PACKAGING WITH THE
  REPRODUCTION AND CORRECTED-EVALUATION AS THE SECONDARY CHAPTER, (3) MAKE THE
  REPO REPRODUCIBLE/COHESIVE FROM A CLEAN CHECKOUT, (4) SCOPE REAL-DATA AND THE
  TEACHER HONESTLY, (5) POLISH.

## Figure and appendix plan (set 2026-06-15)

Planned figures/media (author's list): training-data screenshots, inference
results, stitching output, an SPTnet architecture diagram, mean/variance sweep
plots, real-data results, and possibly attached videos in the appendix.

FONT: report figures should use Computer Modern Roman to match the report font
(LaTeX default). In matplotlib set `text.usetex=True` with
`font.serif=["Computer Modern Roman"]` (needs a LaTeX install; the working
`figs.ipynb` data-generation cell does this), or fall back to the bundled `cmr10`
font (`mathtext.fontset="cm"`) if usetex is unavailable. See
`[[figures-computer-modern-font]]`.

Mapping each report slot to a figure and its source artifact (every figure must
trace to a committed artifact or notebook cell; PLOTS as vector PDF per
`[[figures-as-pdf]]`, screenshots/videos raster is fine):

METHODS FIGURES DONE 2026-06-16 (all three in `main.tex` + build clean):
- Methods §Pipeline overview — DONE: `figures/pipeline_overview.pdf` (TikZ,
  Computer Modern), horizontal generate→train→infer→segment/stitch with real
  thumbnails (train_eg / example_curve / tile_overview / stitched_overview).
- Methods §SPTnet architecture — DONE: `figures/architecture.png` (AI-generated,
  accuracy-verified against the code; caption says "unchanged from Bi et al.").
  Caveat: PNG raster + sans-serif labels (NOT Computer Modern) — accepted
  trade-off. Old TikZ source `figures/architecture.tex` is now unused but kept.
- Methods §Synthetic data generation — DONE: `figures/data_generation_comparison.pdf`
  (MATLAB vs Python from same tracks; last cell of `notebooks/figs.ipynb`).
  Optional extra still open: a frame with GT trajectories overlaid.
- Results §Runtime Optimisation — DONE: benchmark table + per-epoch distribution
  figure (`figures/benchmark_per_epoch_train.pdf`).
RESULTS SECTION ORDER (reset 2026-06-16): (1) Runtime Optimisation, (2) Diffusion
estimation: ablations, improvements and sweeps, (3) Comparison to the original
model, (4) Real-data and stitching. Sweeps/ablations now come BEFORE the
comparison.
- Results §Diffusion estimation (ablations/improvements + sweeps) — DONE so far:
  `figures/reproduction_bias.pdf` (signed per-track D bias box plot, three REPORT
  models, source `ft_matched_*`); `figures/ablation_scratch_calibration.pdf`
  (from-scratch ablations pred-vs-true D: Baseline & BCE-with-logits collapse to the
  mean slope~0, Log H/D & H/D-off-matching recover slope~0.93; source
  `scratch_matched_*`). SHELVED 2026-06-16: `ablation_scratch_calibration.pdf`
  must NOT be used as-is — Baseline & BCE were CONVERGENCE FAILURES (v_loss flat
  ~2.0 epoch 1->30), so their collapse/poor detection is dead-run artifact, not a
  finding. See the 2026-06-16 notes entries. Use the FINETUNE ablations (all
  converged) for the loss-config story instead. Still needed: mean/variance SWEEP
  figure + table, metric-artifact panel (R3), detection-bottleneck (R4). CAVEAT in
  text: FT gains are in-distribution; forgetting on the general sparse set.
- Results §Comparison to original model — reproduction (Original ≈ Full model);
  the equivalence is already visible in `reproduction_bias.pdf`. Still needed: an
  agreement table quantifying closeness to the original.
- Results §Real-data and stitching — inference results, stitching output, real
  data results, optional videos (author's list); shorter section.

NEXT UP (Results figures, in suggested order — all need redrawing as vector PDF
with CM font from the matched CSVs; the `diff_evals/**/comparison_plots/*.png` are
diagnostics, not report figures):
- DONE: R2 bias box plot `figures/reproduction_bias.pdf` (now serves the §Diffusion
  estimation section; three report models Original/Full/Full-FT from `ft_matched_*`).
1. Sweep figure + table for §Diffusion estimation: per-track D calibration/bias
   across the mean sweep (0.05-0.45) and range sweep (±0.01,0.05,0.15). Source:
   `final_mean_sweep_comparison`/`final_range_sweep_comparison` data.
2. R3 metric-artifact + R4 detection-bottleneck (§Diffusion estimation) — share the
   matched-eval data; R4 turns the detection-bottleneck finding into a figure.
3. Agreement table for §Comparison to original model (Original ≈ Full model).
4. R5 fine-tune & transfer cost (needs the forgetting CSV regenerated, artifacts
   item D) and R6 real-data overlay last.

Other figures to CONSIDER adding (not yet on the author's list):
- Detection precision/recall vs condition (for the sweeps / critical-evaluation
  section): directly visualises the detection-bottleneck-under-distribution-shift
  finding, which is currently argued in prose only. Source data exists in the
  matched-eval CSVs.
- A metric-comparison panel for the corrected-evaluation point (mean-target vs
  matched per-track, with the oracle floor marked): makes the "old metric rewarded
  shrinkage / one best sat below the floor" argument visual.
- Loss/convergence curve (new vs old) IF the normalisation-fix lower-loss claim is
  made in Results — gives it a figure rather than a bare sentence. Keep separate
  from the speed figure.
- Videos belong in the Appendix (Additional Figures); reference them from the
  real-data section. Confirm submission format allows attached media; if not,
  use still frames + a repo link.

## Open questions (for the write-up, not new experiments)

- What exactly accounts for the 4x speedup, and in what proportions? The
  benchmark decomposition (step 1) must answer this so the claim is defensible.
- Under what conditions is the 4x measured (data size, epochs, hardware, GPU vs
  CPU, with/without AMP)? Pin these down; report a like-for-like comparison.
- For the reproduction: how close are `opt-SPTnet` predictions to the original
  on identical inputs? Quantify with the matched metric, not just visual/means.
- How should the corrected-evaluation finding be framed so it strengthens rather
  than undermines the thesis? Frame as scientific rigor: a faithful reproduction
  plus an honest evaluation that corrects a misleading metric and locates the
  real limitation. This is a contribution, not a failure.
- Is any statistical testing needed? For the speed claim this is now handled: the
  harness repeats each config K=10 times and `analyze_benchmarks.py` reports the
  speedup ratio with a 95% hierarchical-bootstrap CI (resample runs -> epochs).
  The diffusion rankings are single-seed and must not be presented as significant.
