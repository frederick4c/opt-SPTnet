# [OBSOLETE 2026-06-17 — DO NOT USE]

This was superseded: the bug is READ-side and all inference h5 are available
locally, so it was fixed locally via `notebooks/reeval_matched_fixed.py` (per-model
auto-detected x/y swap, re-run on the existing h5). No CSD3 rerun or rsync was
needed. Kept only as a record of the diagnosis. See the 2026-06-17 RESOLVED entry
in `context/notes.md`.

---

# Task for Codex (CSD3): re-run the contaminated matched diffusion evals

You are running on CSD3, where this project's data and code are **split across two
repositories / directories** and the layout differs from a laptop checkout. Treat
all paths below as *logical names*, not literal paths — **locate the real files
yourself** (the matched-eval script, the per-model inference outputs, the
ground-truth files, and where the comparison CSVs are written). Do not assume the
local-laptop tree.

## 1. What is wrong (root cause)

The matched per-track diffusion evaluation loads each model's predicted
coordinates **without accounting for a per-model coordinate-convention split**, so
several models were scored with their x and y axes transposed. This silently
wrecks the Hungarian matching (predictions land in the wrong place), producing
spuriously low recall/precision, huge localisation error, and garbage
prediction-to-particle pairings (so per-track D/H errors, slope and correlation are
also wrong) — **for the affected models only**.

- The evaluator is the **matched** diffusion eval script (on the laptop it is
  `notebooks/diffusion_eval_matched.py`; find its equivalent on CSD3). The bug is
  in its prediction loader (`load_prediction`), which reads the `estimation_xy`
  dataset from each `result_*.h5` as-is.
- Inference result files store, per query: `obj_estimation`, `estimation_xy`,
  `estimation_C` (diffusion, ×max_diffusion), `estimation_H`. Ground-truth files
  store `Hlabel`, `Clabel`, `traceposition` (cell-array refs).
- The coordinate convention is **a property of the model/inference lineage**, not
  the data: newer fine-tuned-lineage inference stores **(x,y)**; the original
  family stores **(y,x)**. (This same split is already handled in the stitched
  *visualisation* via an `xy_order="auto"` auto-detect — but it was never applied
  to the matched eval.)

## 2. Per-model convention (verified empirically)

Probed by computing recall both ways on `D_mean_0p25_pm_0p05`: the **correct**
orientation gives recall ≈ 0.93; the wrong one ≈ 0.25.

| model | stored convention | needs x/y swap? |
|-------|-------------------|-----------------|
| Original ti2 | (y,x) | **YES** |
| Dense/sparse | (y,x) | **YES** |
| Final full model (pre-finetune) | (y,x) | **YES** |
| Final model FT (fine-tuned) | (x,y) | no |
| scratch ablations (baseline/bce/log_hd/hd-down-match) | (x,y) | no |
| finetune ablations | UNVERIFIED — probe before trusting |

Do NOT blindly swap everything: the FT and scratch models are already correct;
swapping them would *introduce* the bug.

## 3. How to confirm on CSD3 before changing anything

For one model and one condition, count matched true positives with and without the
swap (`pred_xy = pred_xy[..., ::-1]`). The correct orientation yields dramatically
higher recall. Reuse the existing eval functions (`load_prediction`,
`load_ground_truth`, `match_video`) rather than rewriting matching. Eval params to
keep fixed: `obj_threshold=0.5`, `min_valid_frames=3`, `gate=0.25`,
`min_overlap=3`, `image_size=64`, `max_diffusion=0.5`, `clip_index=0`.

Expected, once fixed: every model that detects at all reaches recall ≈ 0.9 on the
binned conditions (the real differences between models are then small, NOT the
0.3-vs-0.9 gap currently in the CSVs).

## 4. The fix

Make the prediction loader use the correct convention per model. Either is
acceptable; prefer the robust one:

- **Recommended (auto-detect, model-agnostic):** for each `result_*.h5` (or each
  model directory), determine the orientation that aligns predictions to ground
  truth — e.g. run the match in both orientations on a few videos and keep the one
  with more matched tracks / lower mean matched distance — then apply it to all of
  that model's files. This needs no hard-coded per-model table and is robust to new
  models.
- **Minimal:** add a per-model `xy_order` flag and set it from the table in §2.

Whichever you choose, it must leave the (x,y) models untouched (no double-swap) and
swap only the (y,x) models.

## 5. What to regenerate (and what NOT to)

Re-run the matched eval and overwrite the comparison CSVs (ranking + summary +
tracks) for every grouping that includes a (y,x) model:

- the **four-model** comparison (Original ti2 / Dense-sparse / Final full model /
  Final model FT) — feeds the report's reproduction + detection figures.
- the **three-model** reproduction comparison (Original / Dense-sparse / Full).
- the **final** single-model (Full) summary.
- the **forgetting** eval (Full model vs FT on the general/sparse `trainingvideos_*`
  held-out set; GT + both models' inference live under a `forget` grouping). Note
  this set is NOT the `D_mean_*` conditions — it is flat per-video files, so the
  condition-globbing CLI may not fit; drive the lower-level functions directly if
  needed.

Do **NOT** re-run / do not need to change:
- the **scratch** ablation eval — already convention-correct (its low Baseline/BCE
  recall is a real convergence failure, not a convention artifact: the swap makes
  it *lower*, confirming (x,y)).
- anything based on raw stitched detection **counts** (e.g. real-data
  over-detection) — counts don't depend on x/y order.

## 6. Verification after the rerun

- On the binned conditions, Original ti2, Dense/sparse and Final full model recall
  jump from ~0.25 to ~0.9 (similar to FT). If a model is still ~0.25, its
  convention is still wrong.
- Localisation RMSE for the corrected models drops from ~0.15 (normalised) to
  ~0.03.
- On the forgetting set, Full model and FT BOTH reach recall ≈ 0.95 (i.e. the old
  "recall 0.95 vs 0.22" gap was the artifact and should vanish).
- Sanity: corrected per-track D slope/correlation for Original/Full become sensible
  (matches are now to the right particles).

## 7. Deliverables back to the laptop repo

Commit/return the regenerated ranking/summary/tracks CSVs for each affected
grouping (same filenames the figure notebook reads), plus a short note of: the fix
applied, the final per-model convention used, and the corrected headline numbers
(per-model recall/precision/loc-RMSE/slope on the binned conditions, and Full-vs-FT
recall on the forgetting set). The laptop side will regenerate the figures
(reproduction bias, detection comparison) from the corrected CSVs.

## 8. Why this matters (context, do not skip)

Two of the report's current findings are artifacts of this bug and will likely
disappear once fixed: (a) a claimed "detection bottleneck" where Original/Full
supposedly fail to detect (~0.3 recall) while FT detects (~0.9) — corrected, all
detect ~0.9; (b) a "catastrophic forgetting" result (FT 0.95 vs Full 0.22 recall on
the general set) — corrected, both ~0.95. The faithful-reproduction conclusion
(Original ≈ Full) should *strengthen*. Report the corrected numbers honestly even
though they remove two results.
