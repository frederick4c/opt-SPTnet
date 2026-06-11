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

1. Substantiate the 4x speed claim with a committed, controlled benchmark. This
   is the report's headline and currently has NO in-repo evidence. Run the
   original `../SPTnet` training and `opt-SPTnet` (`slurm/train_sptnet_benchmark_csd3.slurm`)
   on the SAME data, SAME epochs, SAME hardware (CSD3 Ampere), capture
   `training_seconds` / total wall time, and commit a small results table
   (e.g. `experiments/benchmarks/` CSV + the SLURM `*_metrics.txt`). Decompose
   where the speedup comes from (vectorised normalisation, AMP/TF32, DataLoader
   workers, removed per-epoch plotting/debug, single-pass dimension inference,
   flat ConcatDataset) so the number is explained, not just asserted. State the
   exact comparison conditions in Methods so it is reproducible.

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

- MAIN TASKS NOW, IN ORDER: (1) COMMIT A CONTROLLED 4x BENCHMARK WITH A
  DECOMPOSITION, (2) WRITE THE REPORT AROUND SPEED + PACKAGING WITH THE
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
- Is any statistical testing needed? For the speed claim, repeat the benchmark a
  few times and report mean/spread. The diffusion rankings are single-seed and
  must not be presented as significant.
