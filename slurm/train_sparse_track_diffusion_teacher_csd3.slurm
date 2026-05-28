#!/bin/bash
#SBATCH -J SPT_D_teacher_sparse
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=sparse_track_diffusion_teacher_%j.out
#SBATCH --error=sparse_track_diffusion_teacher_%j.err
#SBATCH --export=ALL

# ---------------------------------------------------------------------
# Module Loading & Environment Setup
# ---------------------------------------------------------------------

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load python/3.11.0-icl

export TEACHER_ENV="${TEACHER_ENV:-$HOME/envs/SPTnet/bin/activate}"
if [[ ! -f "$TEACHER_ENV" ]]; then
    echo "ERROR: Python environment activate script was not found: $TEACHER_ENV"
    echo "Set TEACHER_ENV=/path/to/activate when submitting, or create ~/envs/SPTnet."
    exit 1
fi
source "$TEACHER_ENV"

# ---------------------------------------------------------------------
# Run from this repo where sbatch was submitted.
# Expected layout:
#   ./experiments/
#   ./TrainData/*.h5
# ---------------------------------------------------------------------

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
ulimit -n 65536

export TEACHER_SPARSE_DATA="${TEACHER_SPARSE_DATA:-./TrainData/*.h5}"
export TEACHER_OUT_DIR="${TEACHER_OUT_DIR:-./experiments/csd3_sparse_teacher}"
export TEACHER_EPOCHS="${TEACHER_EPOCHS:-80}"
export TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-512}"
export TEACHER_LR="${TEACHER_LR:-0.001}"
export TEACHER_MAX_DIFF="${TEACHER_MAX_DIFF:-0.5}"
export TEACHER_NOISE_PX="${TEACHER_NOISE_PX:-0.25}"
export TEACHER_FRAME_DROP_PROB="${TEACHER_FRAME_DROP_PROB:-0.05}"
export TEACHER_TRUNCATE_MIN_FRAMES="${TEACHER_TRUNCATE_MIN_FRAMES:-10}"
export TEACHER_TRUNCATE_MAX_FRAMES="${TEACHER_TRUNCATE_MAX_FRAMES:-30}"
export TEACHER_FEATURE_SET="${TEACHER_FEATURE_SET:-multilag_msd}"

mkdir -p "$TEACHER_OUT_DIR"

TEACHER_CHECKPOINT="$TEACHER_OUT_DIR/track_diffusion_teacher_sparse.pt"
TEACHER_METRICS="$TEACHER_OUT_DIR/track_diffusion_teacher_sparse_metrics.csv"
TEACHER_EVAL="$TEACHER_OUT_DIR/track_diffusion_teacher_eval_sparse.csv"

echo "Starting sparse-only track-diffusion teacher training on CSD3..."
echo "Submit dir:          $SLURM_SUBMIT_DIR"
echo "Sparse data:         $TEACHER_SPARSE_DATA"
echo "Output dir:          $TEACHER_OUT_DIR"
echo "Checkpoint:          $TEACHER_CHECKPOINT"
echo "Epochs:              $TEACHER_EPOCHS"
echo "Batch size:          $TEACHER_BATCH_SIZE"
echo "Learning rate:       $TEACHER_LR"
echo "Max diffusion:       $TEACHER_MAX_DIFF"
echo "Feature set:         $TEACHER_FEATURE_SET"
echo "Noise px:            $TEACHER_NOISE_PX"
echo "Frame drop prob:     $TEACHER_FRAME_DROP_PROB"
echo "Truncate frames:     $TEACHER_TRUNCATE_MIN_FRAMES-$TEACHER_TRUNCATE_MAX_FRAMES"
echo "Python:              $(which python)"

python - <<'PY'
import h5py
import numpy
import torch
import tqdm

print(f"torch={torch.__version__}; cuda_available={torch.cuda.is_available()}")
print(f"h5py={h5py.__version__}; numpy={numpy.__version__}; tqdm={tqdm.__version__}")
PY

if [[ ! -d experiments ]]; then
    echo "ERROR: ./experiments was not found. Submit from the refactored repo root."
    exit 1
fi

if [[ ! -f pyproject.toml || ! -d src/sptnet ]]; then
    echo "ERROR: pyproject.toml and src/sptnet were not found in $(pwd)."
    echo "Submit the job from this repo root."
    exit 1
fi

if ! compgen -G "$TEACHER_SPARSE_DATA" > /dev/null; then
    echo "ERROR: no sparse files matched: $TEACHER_SPARSE_DATA"
    exit 1
fi

SPARSE_FILE_COUNT=$(python - "$TEACHER_SPARSE_DATA" <<'PY'
import glob
import sys

print(len(glob.glob(sys.argv[1])))
PY
)
echo "Sparse files matched: $SPARSE_FILE_COUNT"

python experiments/train_track_diffusion_teacher.py \
    --data "$TEACHER_SPARSE_DATA" \
    --output "$TEACHER_CHECKPOINT" \
    --metrics-csv "$TEACHER_METRICS" \
    --epochs "$TEACHER_EPOCHS" \
    --batch-size "$TEACHER_BATCH_SIZE" \
    --learning-rate "$TEACHER_LR" \
    --max-diff "$TEACHER_MAX_DIFF" \
    --feature-set "$TEACHER_FEATURE_SET" \
    --noise-px "$TEACHER_NOISE_PX" \
    --frame-drop-prob "$TEACHER_FRAME_DROP_PROB" \
    --truncate-min-frames "$TEACHER_TRUNCATE_MIN_FRAMES" \
    --truncate-max-frames "$TEACHER_TRUNCATE_MAX_FRAMES" \
    --device cuda

echo "Training complete. Evaluating best checkpoint on sparse files..."

python experiments/eval_track_diffusion_teacher.py \
    --checkpoint "$TEACHER_CHECKPOINT" \
    --data "$TEACHER_SPARSE_DATA" \
    --output-csv "$TEACHER_EVAL" \
    --batch-size "$TEACHER_BATCH_SIZE" \
    --device cuda

echo "Sparse-only track-diffusion teacher job completed."
echo "Sparse files:  $SPARSE_FILE_COUNT"
echo "Checkpoint:    $TEACHER_CHECKPOINT"
echo "Train metrics: $TEACHER_METRICS"
echo "Sparse eval:   $TEACHER_EVAL"
