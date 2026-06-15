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

1. Substantiate the ~4.2x speed claim with a committed, controlled, BOOTSTRAPPED
   benchmark. The harness was rebuilt 2026-06-15 (see the 2026-06-15 notes entry)
   and is in place; what remains is running it on CSD3 and committing results.
   - Metric: steady-state TRAINING seconds per epoch (epoch 1 dropped as warmup;
     validation timed separately and excluded). This removes the early-stopping /
     epoch-count confound that produced the bogus "12x".
   - Fixed identical workload for old and new: same data subset, batch 16,
     query 20, lr 1e-4, max_dc 0.5, `--max-epochs 4` (old hardcodes patience 6 so
     4 epochs always run; new uses `--patience 99 --grad-clip 1.0`). CSD3 Ampere,
     1 GPU, `SPT_NUM_WORKERS=2`.
   - Repeats: K=10 independent SLURM array tasks per system → hierarchical
     bootstrap (resample runs -> epochs) for the speedup ratio + 95% CI.
   - Decomposition: K=3 each with one optimization disabled — `new_no_amp`
     (`SPT_DISABLE_AMP=1`), `new_no_tf32` (`SPT_DISABLE_TF32=1`),
     `new_no_cudnn_bench` (`SPT_CUDNN_BENCHMARK=0`), and `old_no_plot`
     (`SPT_DISABLE_PLOT=1`). NOTE: DataLoader workers are already in BOTH repos,
     so this isolates compute (AMP/TF32/cudnn) + removed plotting, not data
     loading; report data-loading/startup gains separately from
     `startup_plus_data_loading_seconds`.
   - Tooling (committed): instrumented `src/sptnet/training/cli.py` (per-epoch
     `epoch_timing.csv` + the env toggles), generalised
     `slurm/train_sptnet_benchmark_csd3.slurm` (`SPT_SYSTEM=old|new`, job array,
     per-run dirs), `experiments/benchmarks/analyze_benchmarks.py` (bootstrap +
     tables + plot), and `experiments/benchmarks/README.md` (full protocol +
     submit commands).
   - REMAINING: (a) apply the matching instrumentation to
     `../SPTnet/SPTnet_training_old_cli.py` ON CSD3 (per-epoch `epoch_timing.csv`
     + `SPT_DISABLE_PLOT` toggle; Codex prompt + steps in the benchmark README);
     (b) submit the K=10 headline arrays and K=3 decomposition arrays; (c) run
     the analyzer; (d) commit `results/` + per-run `epoch_timing.csv` /
     `*_metrics.txt`. State the exact comparison conditions in Methods.
   - The existing `experiments/benchmarks/standard_new` run is BROKEN (NaN
     divergence, no grad-clip) and must not be used as evidence; `standard_old`
     and `sptnet_final_30293619.out` are healthy and corroborate ~4.9 vs ~1.15
     it/s but differ in data size/epochs so are not a clean head-to-head.

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

- MAIN TASKS NOW, IN ORDER: (1) RUN THE REBUILT BENCHMARK HARNESS ON CSD3 AND
  COMMIT THE BOOTSTRAPPED ~4.2x RESULT + DECOMPOSITION (harness is built; remaining
  work is the old-script patch on CSD3, the SLURM arrays, and the analysis),
  (2) WRITE THE REPORT AROUND SPEED + PACKAGING WITH THE
  REPRODUCTION AND CORRECTED-EVALUATION AS THE SECONDARY CHAPTER, (3) MAKE THE
  REPO REPRODUCIBLE/COHESIVE FROM A CLEAN CHECKOUT, (4) SCOPE REAL-DATA AND THE
  TEACHER HONESTLY, (5) POLISH.

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
