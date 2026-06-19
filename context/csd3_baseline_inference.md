# CSD3 Task: Baseline Model Inference + Data Transfer

## Context

This file tells you (Codex on CSD3) exactly what to run and what to send back.
All paths below are **relative to the opt-SPTnet repo root on CSD3** unless
stated otherwise. The sibling `../SPTnet/` repo is the original codebase.

### Why

The thesis compares model quality between:
- **Baseline model**: trained with `SPTnet_training_old_cli.py` (old script, same
  100k training videos, epoch-17 checkpoint)
- **Final full model**: trained with `sptnet-train` (opt-SPTnet, same 100k
  videos, epoch-17 checkpoint)

The baseline replaces the Original ti2 model in the matched diffusion evaluation.
Ti2 was trained on 200k+ different videos — an unfair comparison. The baseline
uses identical training data so any quality difference is attributable to the
training approach, not the data.

### What needs to come back locally

1. Inference result h5 files for the **baseline model** on:
   - A. 7 diffusion conditions (matching existing `diff_evals/final/diff_evals/final_full_model/` structure)
   - B. Forget/general-sparse set (matching `diff_evals/forget/final_full/`)
2. Loss-curve data for **both** models:
   - Baseline: `training_log.txt` → convert to CSV (see Step 3)
   - Final full model: `loss_history.csv` (already CSV, just copy)

---

## Step 0: Discover paths

Run these before anything else. Record the real paths.

```bash
# 1. Baseline model checkpoint (old script, 100k data, epoch-17 best checkpoint)
#    Likely named trained_model.pt or trained_model in a run directory from
#    SPTnet_training_old_cli.py. Check ../SPTnet/ and any shared scratch space.
find . ../SPTnet -name "trained_model*" -newer /dev/null 2>/dev/null | grep -v ".git" | head -30

# 2. Final full model checkpoint (opt-SPTnet, epoch-17 best checkpoint)
#    Should be in perf_exp/trained_model or wherever sptnet-train wrote it.
find . -name "trained_model*" -path "*/perf_exp/*" 2>/dev/null | head -10

# 3. Source binned condition videos (used for Final full model inference)
#    These are the h5 files with trainingvideos_N.h5 pattern, one dir per condition.
#    The 7 conditions are listed in Step 1 below.
find . -name "trainingvideos_1.h5" -path "*D_mean*" 2>/dev/null | head -10

# 4. Forget-set source videos (28 files, seed 70123, sparse 1-10 particles/video)
#    Check diff_evals/forget/ and wherever the sparse test data was generated.
#    These need to contain video pixels (timelapsedata dataset), not just labels.
find . -name "trainingvideos_*.h5" -path "*forget*" 2>/dev/null | head -10

# Verify a forget file has video pixels (not labels-only):
python3 -c "
import h5py, sys
f = h5py.File('<PATH_TO_FORGET_trainingvideos_1.h5>', 'r')
print(list(f.keys()))
"
# If 'timelapsedata' is not in the keys, the source videos are elsewhere — search:
find . -name "trainingvideos_*.h5" | xargs -I{} python3 -c "
import h5py, sys
try:
    f = h5py.File('{}', 'r')
    if 'timelapsedata' in f:
        print('HAS VIDEO:', '{}')
except: pass
" 2>/dev/null | grep "forget\|sparse\|70123" | head -10
```

**Fill in before proceeding:**
- `BASELINE_MODEL=<path to baseline epoch-17 trained_model file>`
- `FINAL_FULL_MODEL=<path to Final full model epoch-17 trained_model file>`
- `CONDITION_SRC_ROOT=<directory whose subdirs contain trainingvideos_*.h5 per condition>`
- `FORGET_SRC=<directory containing the 28 forget-set trainingvideos_*.h5 files>`
- `BASELINE_LOSS_LOG=<path to training_log.txt for the baseline training run>`
- `FINAL_FULL_LOSS_CSV=<path to loss_history.csv for the Final full model training run>`

---

## Step 1: Inference — 7 diffusion conditions

Set up a clean output root for the baseline and symlink the source videos into it,
then run the existing inference slurm script.

```bash
cd <opt-SPTnet repo root>

CONDITIONS=(
    D_mean_0p05_pm_0p05
    D_mean_0p15_pm_0p05
    D_mean_0p25_pm_0p01
    D_mean_0p25_pm_0p05
    D_mean_0p25_pm_0p15
    D_mean_0p35_pm_0p05
    D_mean_0p45_pm_0p05
)

BASELINE_INPUT_ROOT="diff_evals/baseline"
mkdir -p "$BASELINE_INPUT_ROOT"

# Symlink source videos into the baseline input tree
for cond in "${CONDITIONS[@]}"; do
    mkdir -p "$BASELINE_INPUT_ROOT/$cond"
    for f in "$CONDITION_SRC_ROOT/$cond"/trainingvideos_*.h5; do
        ln -sfn "$(realpath "$f")" "$BASELINE_INPUT_ROOT/$cond/$(basename "$f")"
    done
    echo "  $cond: $(ls "$BASELINE_INPUT_ROOT/$cond"/trainingvideos_*.h5 2>/dev/null | wc -l) files linked"
done

# Submit inference job
SPT_MODEL_PATH="$BASELINE_MODEL" \
SPT_DIFF_EVAL_INPUT_ROOT="$BASELINE_INPUT_ROOT" \
SPT_HDF5_CLIP_INDEX=0 \
SPT_CONDITIONS=all \
  sbatch slurm/inference_diff_eval_csd3.slurm
```

Results will be written to:
`diff_evals/baseline/<condition>/inference/inference_results/result_trainingvideos_N.h5`

Wait for the job to complete before Step 4.

---

## Step 2: Inference — forget/general-sparse set

```bash
FORGET_OUTPUT="diff_evals/forget/baseline"
mkdir -p "$FORGET_OUTPUT/inference"
ln -sfn "$(realpath "$BASELINE_MODEL")" "$FORGET_OUTPUT/inference/trained_model"

SPT_MODEL_PATH="$FORGET_OUTPUT/inference/trained_model" \
SPT_INFER_DATA="$FORGET_SRC/trainingvideos_*.h5" \
SPT_HDF5_CLIP_INDEX=0 \
  sbatch slurm/inference_sptnet_csd3.slurm
```

Results will be written to:
`<FORGET_SRC>/inference_results/result_trainingvideos_N.h5`

After the job completes, move the results to the expected location:
```bash
mkdir -p "$FORGET_OUTPUT/inference/inference_results"
mv "$FORGET_SRC/inference_results/"result_trainingvideos_*.h5 \
   "$FORGET_OUTPUT/inference/inference_results/"
```

---

## Step 3: Loss-curve CSV for the baseline model

The old script writes `training_log.txt` (text), not a CSV. Convert it to the same
format as opt-SPTnet's `loss_history.csv` so both can be plotted together.

```python
# Run as: python3 convert_loss_log.py <path/to/training_log.txt> <output.csv>
import re, csv, sys

log_path, out_path = sys.argv[1], sys.argv[2]
with open(log_path) as f:
    text = f.read()

rows = []
for m in re.finditer(
    r'epoch\s+(\d+),\s*t_loss:\s*([\d.]+),\s*v_loss:\s*([\d.]+)'
    r'(?:.*?t_cls_loss:\s*([\d.]+),\s*v_cls_loss:\s*([\d.]+))?'
    r'(?:.*?t_coor_loss:\s*([\d.]+),\s*v_coor_loss:\s*([\d.]+))?'
    r'(?:.*?t_hurst_loss:\s*([\d.]+),\s*v_hurst_loss:\s*([\d.]+))?'
    r'(?:.*?t_diff_loss:\s*([\d.]+),\s*v_diff_loss:\s*([\d.]+))?'
    r'(?:.*?t_bg_loss:\s*([\d.]+),\s*v_bg_loss:\s*([\d.]+))?',
    text, re.DOTALL
):
    g = m.groups()
    rows.append({
        'epoch': g[0], 't_loss': g[1], 'v_loss': g[2],
        't_cls': g[3] or '', 'v_cls': g[4] or '',
        't_coor': g[5] or '', 'v_coor': g[6] or '',
        't_hurst': g[7] or '', 'v_hurst': g[8] or '',
        't_diff': g[9] or '', 'v_diff': g[10] or '',
        't_bg': g[11] or '', 'v_bg': g[12] or '',
    })

with open(out_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {len(rows)} epochs to {out_path}")
```

Save this script as `convert_loss_log.py` and run:
```bash
python3 convert_loss_log.py "$BASELINE_LOSS_LOG" diff_evals/baseline/loss_history.csv
```

Also copy the Final full model loss CSV:
```bash
cp "$FINAL_FULL_LOSS_CSV" diff_evals/final/loss_history.csv
```

---

## Step 4: Package and transfer results

Wait for both inference jobs (Steps 1 and 2) to complete, then rsync to the
local machine. The local user is `fredlawrence` (email: frederick4c@gmail.com).

```bash
# Check jobs have finished and result files exist
echo "Condition results:"
for cond in "${CONDITIONS[@]}"; do
    n=$(ls diff_evals/baseline/$cond/inference/inference_results/*.h5 2>/dev/null | wc -l)
    echo "  $cond: $n files"
done

echo "Forget results:"
ls diff_evals/forget/baseline/inference/inference_results/*.h5 2>/dev/null | wc -l

# rsync to local machine
# Replace LOCAL_HOST with the laptop's address (e.g. fred-macbook.local or IP).
# Run from CSD3 if it can reach the laptop; otherwise run these rsync commands
# FROM THE LOCAL MACHINE (pulling from CSD3).

LOCAL_HOST="<laptop hostname or IP>"
LOCAL_REPO="<local path to opt-SPTnet repo>"

# Condition inference results (baseline model)
rsync -avz --progress \
    diff_evals/baseline/ \
    "${LOCAL_HOST}:${LOCAL_REPO}/diff_evals/baseline/"

# Forget-set inference results (baseline model)
rsync -avz --progress \
    diff_evals/forget/baseline/ \
    "${LOCAL_HOST}:${LOCAL_REPO}/diff_evals/forget/baseline/"

# Loss CSVs
rsync -avz --progress \
    diff_evals/baseline/loss_history.csv \
    diff_evals/final/loss_history.csv \
    "${LOCAL_HOST}:${LOCAL_REPO}/diff_evals/"
```

**Alternatively**, if pushing from CSD3 is not possible, provide the user with
this command to run on the local machine:

```bash
# Run locally, replacing CSD3_USER and CSD3_HOST appropriately:
CSD3_USER="<username>"
CSD3_HOST="login.hpc.cam.ac.uk"
CSD3_REPO="<path to opt-SPTnet repo on CSD3>"
LOCAL_REPO="/Users/fredlawrence/DiS/opt-SPTnet"

rsync -avz --progress \
    "${CSD3_USER}@${CSD3_HOST}:${CSD3_REPO}/diff_evals/baseline/" \
    "${LOCAL_REPO}/diff_evals/baseline/"

rsync -avz --progress \
    "${CSD3_USER}@${CSD3_HOST}:${CSD3_REPO}/diff_evals/forget/baseline/" \
    "${LOCAL_REPO}/diff_evals/forget/baseline/"

rsync -avz --progress \
    "${CSD3_USER}@${CSD3_HOST}:${CSD3_REPO}/diff_evals/baseline/loss_history.csv" \
    "${CSD3_USER}@${CSD3_HOST}:${CSD3_REPO}/diff_evals/final/loss_history.csv" \
    "${LOCAL_REPO}/diff_evals/"
```

---

## Step 5: Verify completeness before signing off

```bash
# Expected file counts (matching Final full model):
# D_mean_0p05_pm_0p05 and D_mean_0p15_pm_0p05: ~100 result files each
# All other 5 conditions: ~20 result files each
# Forget set: ~28 result files

for cond in "${CONDITIONS[@]}"; do
    n=$(ls diff_evals/baseline/$cond/inference/inference_results/*.h5 2>/dev/null | wc -l)
    ref=$(ls diff_evals/final/diff_evals/final_full_model/$cond/inference/inference_results/*.h5 2>/dev/null | wc -l)
    status=$( [ "$n" -eq "$ref" ] && echo "OK" || echo "MISMATCH (got $n, expected $ref)" )
    echo "$cond: $status"
done

n_forget=$(ls diff_evals/forget/baseline/inference/inference_results/*.h5 2>/dev/null | wc -l)
echo "Forget set: $n_forget files (expected ~28)"

echo "Loss CSVs:"
[ -f diff_evals/baseline/loss_history.csv ] && wc -l diff_evals/baseline/loss_history.csv || echo "  MISSING baseline loss CSV"
[ -f diff_evals/final/loss_history.csv ] && wc -l diff_evals/final/loss_history.csv || echo "  MISSING final full model loss CSV"
```

---

## What happens next (on the local machine, after transfer)

Once all files arrive:

1. Run `notebooks/reeval_matched_fixed.py` adding the baseline model alongside the
   Final full model and FT model. The script auto-detects x/y coordinate convention
   per model — the baseline (old-lineage model) will likely be (y,x), same as the
   other old-lineage models.

2. Regenerate `report/figures/reproduction_bias.pdf` (R2) from the updated
   `ft_matched_*` CSVs, replacing Original ti2 with the baseline.

3. Regenerate `report/figures/transfer_cost.pdf` (R5) using `forget_matched_*`
   updated to include the baseline.

4. Plot val loss vs wall-clock time for both models using the loss CSVs +
   `experiments/benchmarks/new_fresh_good_runs/` epoch_timing.csv.
