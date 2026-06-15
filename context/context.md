# Project Context

This repository is an optimized and packaged SPTnet project for an MPhil thesis.
The work is both a reproducibility effort and an extension of the original
SPTnet codebase.

The thesis report is a 7000 word report in `report/`, with the main LaTeX file
at `report/main.tex`. Code changes should be made with the report context in
mind: the repository supports reproducible training, inference, evaluation,
data generation, and extensions of SPTnet.

## Report focus (decided 2026-06-10; benchmark corrected 2026-06-15)

The MAIN contribution of the report is now (1) the ~4.2x training-speed
improvement of `opt-SPTnet` over the original `../SPTnet`, and (2) the
refactoring of the original MATLAB-dependent script collection into an
installable, tested, documented Python package with command-line interfaces.
Together these are a reproducibility-and-usability story: the system is faster,
license-free (MATLAB/DIPimage removed), and usable by independent researchers.

This replaces the earlier "~12x", which was an artifact: it divided total
`training_seconds` across two runs with different epoch counts (the new run had
diverged to NaN and early-stopped). The defensible, like-for-like figure is now
CONFIRMED (2026-06-15): a **4.33x** steady-state per-epoch training speedup,
95% CI [4.28, 4.39] (8 runs each, hierarchical bootstrap; end-to-end 4.40x).
Evidence in `experiments/benchmarks/{new_fresh_good_runs,old_good_8_runs}/` and
`results/`. The per-optimization decomposition was deliberately NOT run. See
`context/plan.md` step 1 and the 2026-06-15 notes entries.

The diffusion-loss experiments are NOT the headline. They become a secondary
"critical evaluation" chapter whose value is methodological honesty: the
reproduction (the high-D compression bias is inherent to SPTnet, present in the
original ti2 model too) and the corrected per-track evaluation, which showed the
earlier diffusion-ablation "improvements" were largely metric artifacts and that
the real bottleneck on the dense binned-style eval is detection/localisation
under distribution shift, not the diffusion head.

The project is moving OUT of an experimentation phase and INTO a consolidation
phase: no new model experiments (the binned bias fine-tune is shelved; see
`context/plan.md`). The focus is making the repository and report watertight,
cohesive, and reproducible around the speed and packaging contributions.

The sibling repository `../SPTnet` is the original/reference codebase used for
comparison. This repository, `opt-SPTnet`, contains the optimized package form.
See `context/changes.md` for a summary of package-level changes from the
original repository.

Agents should preserve the distinction between core package functionality and
experiment-specific scripts. When documenting or modifying project context,
focus on reusable functionality, reproducibility, optimization, and thesis
relevance unless the user explicitly asks for experiment-specific notes.

The diffusion-teacher model/experiment should be treated as a side experiment,
not part of the main thesis path. It is likely to be omitted from the report
unless it provides clear, report-worthy value beyond the direct loss/training
changes and binned-data fine-tuning results.

## Diffusion experiments (secondary chapter; experimentation now closed)

This is background for the secondary critical-evaluation chapter, not the report
headline. No further diffusion experiments are planned.

Earlier work focused on improving diffusion (`D`) and Hurst exponent (`H`)
prediction accuracy while preserving particle detection and tracking quality.
The key experiment family uses binned synthetic sparse videos from
`./diff_bins/<condition>/*.h5`, with diffusion ranges across approximately
`D=0.0-0.5`.

Three experiment stages have been run:

- Fine-tuning from `./perf_exp/trained_model` for short controlled comparisons.
- Training from scratch for 30 epochs on the binned diffusion data.
- Training a final full model on the original sparse training distribution with
  the combined settings suggested by the ablations.

The scratch baseline and BCE-logits-only runs did not learn tracking well
enough for useful diffusion regression; their D predictions collapsed close to
the mean. In contrast, log-space D/H losses and removing H/D terms from
Hungarian matching both noticeably improved diffusion matching. This suggests
that noisy early H/D predictions should not dominate query assignment, while
log-style D/H losses can improve diffusion calibration once assignments are
reasonable.

Fine-tuning showed a different but compatible signal: BCE-with-logits helped
slightly when starting from an already useful pretrained model. The final full
model therefore used the three useful conditions together: BCE-with-logits
objectness, log-style D/H losses, and H/D removed from matching while retained
in the final loss.

The final full model has now been evaluated under
`diff_evals/final/diff_evals/final_full_model/`, with plots and CSVs in
`diff_evals/final/comparison_plots/`. It remained compressed at high D and did
not match the binned ablation performance.

IMPORTANT (2026-06-10 audit): the existing diffusion evaluation does not measure
per-track accuracy. It scores predictions against the nominal condition mean,
never reads the ground-truth per-particle labels, and does no
prediction-to-particle matching, so it rewards predicting the condition mean and
penalises correct per-track predictions (the best ablation even scores below the
per-track oracle floor). Before any further diffusion experiment or report claim,
the evaluation must be rebuilt to read GT labels, match predictions to particles,
report per-track error and detection metrics, and the existing runs re-scored.
This blocks the bias fine-tune. See `context/plan.md` and the 2026-06-10 audit
entries in `context/notes.md`.

Once the metric is fixed, the next experiment is a targeted fine-tune from the
final full model on the balanced binned data, using the combined settings and a
fresh optimizer, judged on per-track high-D error and retained detection/tracking
rather than mean-bias. The training-distribution confound (binned ablations vs
sparse-trained final model, both evaluated on binned-style conditions) must also
be resolved before attributing performance differences to the objective choices.

The final cell of `notebooks/testing.ipynb` currently regenerates and displays
the final full-model diffusion evaluation. Earlier scratch and fine-tune
comparison artifacts are still available in `diff_evals/scratch/comparison_plots/`
and `diff_evals/finetune/comparison_plots/`.
