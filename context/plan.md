# Project Plan

Use this file for next steps, active tasks, and handoff plans for future coding
agents.

## Next Steps

- Train the final full model on all available training data using the combined
  settings supported by the ablations:
  BCE-with-logits objectness, log-style D/H losses, and H/D removed from
  Hungarian matching while retained in the final loss.
- After final training, run the same diffusion evaluation used for the ablation
  plots and compare against the current model, scratch ablations, and
  fine-tuning ablations. Save ranking CSVs and plots in `diff_evals/`.
- Collect and write the results needed for the report:
  diffusion calibration/MAE plots, high-D behavior, detection/tracking quality,
  qualitative prediction examples, training curves, and runtime/reproducibility
  details.
- Make sure every report figure/table has a clear source artifact or notebook
  cell. Current useful sources include `diff_evals/scratch/comparison_plots/`,
  `diff_evals/finetune/comparison_plots/`, and the final cell of
  `notebooks/testing.ipynb`.
- Write the report in `report/`, using `context/changes.md` for package
  improvements and `context/notes.md` for experiment findings.
- For claims in the report, frame the scratch baseline/BCE failure as a
  tracking/objectness learning issue leading to mean-like D predictions, not as
  evidence that BCE is generally bad. In fine-tuning, BCE-with-logits helped
  slightly from a pretrained model.

- MAIN TASKS ARE NOW: TRAIN THE FINAL MODEL, COLLECT/VERIFY RESULTS, AND WRITE
  THE REPORT IN report/

## Open Questions

- Has the final combined model finished training and been evaluated against the
  same diffusion bins as the ablations?
- Do the final model detection/tracking metrics improve enough that D/H
  evaluation is trustworthy, especially compared with the scratch baseline and
  scratch BCE-logits runs?
- Is statistical testing worth including in the report? Current single-run
  ablations are enough for a practical engineering conclusion, but not enough
  for a strong inferential claim without resampling or repeated seeds.
