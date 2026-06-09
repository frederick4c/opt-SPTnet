# Project Context

This repository is an optimized and packaged SPTnet project for an MPhil thesis.
The work is both a reproducibility effort and an extension of the original
SPTnet codebase.

The thesis report is a 7000 word report in `report/`, with the main LaTeX file
at `report/main.tex`. Code changes should be made with the report context in
mind: the repository supports reproducible training, inference, evaluation,
data generation, and extensions of SPTnet.

The sibling repository `../SPTnet` is the original/reference codebase used for
comparison. This repository, `opt-SPTnet`, contains the optimized package form.
See `context/changes.md` for a summary of package-level changes from the
original repository.

Agents should preserve the distinction between core package functionality and
experiment-specific scripts. When documenting or modifying project context,
focus on reusable functionality, reproducibility, optimization, and thesis
relevance unless the user explicitly asks for experiment-specific notes.

## Current Experimental Focus

Recent work has focused on improving diffusion (`D`) and Hurst exponent (`H`)
prediction accuracy while preserving particle detection and tracking quality.
The key experiment family uses binned synthetic sparse videos from
`./diff_bins/<condition>/*.h5`, with diffusion ranges across approximately
`D=0.0-0.5`.

Two ablation styles have been run:

- Fine-tuning from `./perf_exp/trained_model` for short controlled comparisons.
- Training from scratch for 30 epochs on the binned diffusion data.

The scratch baseline and BCE-logits-only runs did not learn tracking well
enough for useful diffusion regression; their D predictions collapsed close to
the mean. In contrast, log-space D/H losses and removing H/D terms from
Hungarian matching both noticeably improved diffusion matching. This suggests
that noisy early H/D predictions should not dominate query assignment, while
log-style D/H losses can improve diffusion calibration once assignments are
reasonable.

Fine-tuning showed a different but compatible signal: BCE-with-logits helped
slightly when starting from an already useful pretrained model. The sensible
final experiment is therefore to train a full model on all available training
data using the three useful conditions together: BCE-with-logits objectness,
log-style D/H losses, and H/D removed from matching while retained in the final
loss.

The latest scratch comparison plots and CSVs are in
`diff_evals/scratch/comparison_plots/`. The final cell of
`notebooks/testing.ipynb` regenerates and displays those scratch comparisons.
