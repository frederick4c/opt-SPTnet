#!/bin/bash
#SBATCH -J SPTnet_bench
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=sptnet_bench_%A_%a.out
#SBATCH --error=sptnet_bench_%A_%a.err
#SBATCH --export=ALL

# ---------------------------------------------------------------------
# Statistically-backed training-speed benchmark harness.
#
# Submit as a job ARRAY for independent repeats, e.g.:
#   sbatch --array=1-10 --export=ALL,SPT_SYSTEM=new,SPT_RUN_NAME=new_full,... \
#       slurm/train_sptnet_benchmark_csd3.slurm
#
# Each array task writes to <SPT_MODEL_BASE_DIR>/<SPT_RUN_NAME>/run_<NN>/ so its
# epoch_timing.csv and *_metrics.txt are isolated. Run without an array for a
# single run (treated as run_01).
#
# SPT_SYSTEM selects the code under test:
#   new -> sptnet-train       (this repo; optimized)
#   old -> $SPT_OLD_SCRIPT     (original ../SPTnet/SPTnet_training_old_cli.py)
#
# Decomposition toggles (read by the training scripts, exported below so they
# are inherited by the Python process):
#   SPT_DISABLE_AMP=1        full-precision (new only)
#   SPT_DISABLE_TF32=1       no TF32        (new only)
#   SPT_CUDNN_BENCHMARK=0    no cudnn autotune (new only)
#   SPT_DISABLE_PLOT=1       skip per-epoch matplotlib (old only)
# ---------------------------------------------------------------------

set -euo pipefail

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load python/3.11.0-icl

cd "$SLURM_SUBMIT_DIR"
source "${SPT_VENV:-$HOME/envs/SPTnet}/bin/activate"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export SPT_NUM_WORKERS="${SPT_NUM_WORKERS:-2}"

# With many HDF5 files, h5py/PyTorch can keep many file handles open.
ulimit -n 65536

# ---------------------------------------------------------------------
# Benchmark settings (override with sbatch --export=ALL,NAME=value,...)
# ---------------------------------------------------------------------

export SPT_SYSTEM="${SPT_SYSTEM:-new}"
export SPT_OLD_SCRIPT="${SPT_OLD_SCRIPT:-../SPTnet/SPTnet_training_old_cli.py}"

export SPT_TRAIN_DATA="${SPT_TRAIN_DATA:-./TrainData/*.mat}"
export SPT_MAX_FILES="${SPT_MAX_FILES:-100}"
export SPT_RUN_NAME="${SPT_RUN_NAME:-benchmark}"
export SPT_MODEL_BASE_DIR="${SPT_MODEL_BASE_DIR:-./Trained_models/benchmarks}"

export SPT_BATCH_SIZE="${SPT_BATCH_SIZE:-16}"
export SPT_QUERY="${SPT_QUERY:-20}"
export SPT_LEARNING_RATE="${SPT_LEARNING_RATE:-0.0001}"
export SPT_MAX_DC="${SPT_MAX_DC:-0.5}"
export SPT_MAX_EPOCHS="${SPT_MAX_EPOCHS:-4}"
export SPT_GPUS="${SPT_GPUS:-1}"
export SPT_CRLB_PATH="${SPT_CRLB_PATH:-}"

# new-only optional args (ignored for SPT_SYSTEM=old)
export SPT_VAL_EVERY="${SPT_VAL_EVERY:-1}"
export SPT_PATIENCE="${SPT_PATIENCE:-99}"
export SPT_GRAD_CLIP="${SPT_GRAD_CLIP:-1.0}"
export SPT_MAX_TRAIN_BATCHES="${SPT_MAX_TRAIN_BATCHES:-}"
export SPT_MAX_VAL_BATCHES="${SPT_MAX_VAL_BATCHES:-}"

# Decomposition toggles (export so the Python process inherits them, with
# defaults that preserve normal behaviour).
export SPT_DISABLE_AMP="${SPT_DISABLE_AMP:-0}"
export SPT_DISABLE_TF32="${SPT_DISABLE_TF32:-0}"
export SPT_CUDNN_BENCHMARK="${SPT_CUDNN_BENCHMARK:-1}"
export SPT_DISABLE_PLOT="${SPT_DISABLE_PLOT:-0}"

# AMP precision / gradient-scaler toggles (new only). Default fp16 keeps the
# GradScaler; SPT_AMP_DTYPE=bfloat16 selects the stable bfloat16 path (scaler off
# by default for bfloat16). Leave SPT_DISABLE_GRAD_SCALER unset to use that default,
# or set it to 1/0 to force the scaler off/on explicitly.
export SPT_AMP_DTYPE="${SPT_AMP_DTYPE:-float16}"
if [[ -n "${SPT_DISABLE_GRAD_SCALER:-}" ]]; then
    export SPT_DISABLE_GRAD_SCALER
fi

# Per-run directory so repeats do not overwrite each other.
run_index="${SLURM_ARRAY_TASK_ID:-1}"
run_tag=$(printf 'run_%02d' "$run_index")
export SPT_MODEL_DIR="${SPT_MODEL_DIR:-$SPT_MODEL_BASE_DIR/$SPT_RUN_NAME/$run_tag}"

if [[ ! -f pyproject.toml || ! -d src/sptnet ]]; then
    echo "ERROR: pyproject.toml and src/sptnet were not found in $(pwd)."
    echo "Submit the job from this repo root."
    exit 1
fi

if [[ "$SPT_SYSTEM" == "new" ]]; then
    if ! command -v sptnet-train >/dev/null 2>&1; then
        echo "ERROR: sptnet-train was not found in PATH."
        echo "Install this repo in the active environment with: python -m pip install -e ."
        exit 1
    fi
elif [[ "$SPT_SYSTEM" == "old" ]]; then
    if [[ ! -f "$SPT_OLD_SCRIPT" ]]; then
        echo "ERROR: old training script not found at: $SPT_OLD_SCRIPT"
        echo "Set SPT_OLD_SCRIPT to the path of SPTnet_training_old_cli.py."
        exit 1
    fi
else
    echo "ERROR: SPT_SYSTEM must be 'new' or 'old' (got '$SPT_SYSTEM')."
    exit 1
fi

# ---------------------------------------------------------------------
# Resolve the training-data file list (shared by both systems).
# ---------------------------------------------------------------------

train_data_list=$(mktemp)
trap 'rm -f "$train_data_list"' EXIT

python - "$SPT_TRAIN_DATA" "$SPT_MAX_FILES" > "$train_data_list" <<'PY'
import glob
import sys

matches = sorted(glob.glob(sys.argv[1]))
max_files = int(sys.argv[2])
if max_files > 0:
    matches = matches[:max_files]
for path in matches:
    print(path)
PY

train_data_files=()
while IFS= read -r path; do
    train_data_files+=("$path")
done < "$train_data_list"

match_count=${#train_data_files[@]}
if [[ "$match_count" -eq 0 ]]; then
    echo "ERROR: training data pattern matched no files: $SPT_TRAIN_DATA"
    exit 1
fi

mkdir -p "$SPT_MODEL_DIR"
benchmark_log="$SPT_MODEL_DIR/slurm_benchmark_${SLURM_JOB_ID:-manual}.log"
benchmark_metrics="$SPT_MODEL_DIR/slurm_benchmark_${SLURM_JOB_ID:-manual}_metrics.txt"

echo "Starting SPTnet benchmark on CSD3 Ampere node..."
echo "System under test: $SPT_SYSTEM"
echo "Job ID: ${SLURM_JOB_ID:-unknown}  Array task: ${SLURM_ARRAY_TASK_ID:-none}  Run tag: $run_tag"
echo "Host: $(hostname)"
echo "Submit directory: $SLURM_SUBMIT_DIR"
echo "Training data pattern: $SPT_TRAIN_DATA"
echo "Max files: $SPT_MAX_FILES  Matched files: $match_count"
echo "First training file: ${train_data_files[0]}"
echo "Last training file: ${train_data_files[$((match_count - 1))]}"
echo "Run name: $SPT_RUN_NAME"
echo "Model dir: $SPT_MODEL_DIR"
echo "Batch size: $SPT_BATCH_SIZE  Query: $SPT_QUERY  LR: $SPT_LEARNING_RATE  Max dc: $SPT_MAX_DC"
echo "Max epochs: $SPT_MAX_EPOCHS  GPUs: $SPT_GPUS  DataLoader workers: $SPT_NUM_WORKERS"
echo "Toggles: DISABLE_AMP=$SPT_DISABLE_AMP DISABLE_TF32=$SPT_DISABLE_TF32 CUDNN_BENCHMARK=$SPT_CUDNN_BENCHMARK DISABLE_PLOT=$SPT_DISABLE_PLOT AMP_DTYPE=$SPT_AMP_DTYPE DISABLE_GRAD_SCALER=${SPT_DISABLE_GRAD_SCALER:-auto}"
echo "Python: $(which python)"
echo "Benchmark log: $benchmark_log"

wall_start_epoch=$(date +%s)
wall_start_human=$(date '+%Y-%m-%d %H:%M:%S %Z')
echo "Benchmark wall-clock start: $wall_start_human"

# ---------------------------------------------------------------------
# Build the system-specific command.
# ---------------------------------------------------------------------

train_cmd=()
if [[ "$SPT_SYSTEM" == "new" ]]; then
    optional_args=()
    if [[ -n "$SPT_VAL_EVERY" ]]; then optional_args+=(--val-every "$SPT_VAL_EVERY"); fi
    if [[ -n "$SPT_PATIENCE" ]]; then optional_args+=(--patience "$SPT_PATIENCE"); fi
    if [[ -n "$SPT_GRAD_CLIP" ]]; then optional_args+=(--grad-clip "$SPT_GRAD_CLIP"); fi
    if [[ -n "$SPT_MAX_TRAIN_BATCHES" ]]; then optional_args+=(--max-train-batches "$SPT_MAX_TRAIN_BATCHES"); fi
    if [[ -n "$SPT_MAX_VAL_BATCHES" ]]; then optional_args+=(--max-val-batches "$SPT_MAX_VAL_BATCHES"); fi
    crlb_args=()
    if [[ -n "$SPT_CRLB_PATH" ]]; then crlb_args=(--crlb-path "$SPT_CRLB_PATH"); fi
    train_cmd=(sptnet-train
        --model-dir "$SPT_MODEL_DIR"
        --data "${train_data_files[@]}"
        --gpus "$SPT_GPUS"
        --batch-size "$SPT_BATCH_SIZE"
        --learning-rate "$SPT_LEARNING_RATE"
        --query "$SPT_QUERY"
        --max_dc "$SPT_MAX_DC"
        --max-epochs "$SPT_MAX_EPOCHS"
        "${optional_args[@]}"
        "${crlb_args[@]}")
else
    # Old script supports only: -b -g -lr -m -q -dc --max-epochs --data
    train_cmd=(python "$SPT_OLD_SCRIPT"
        --model-dir "$SPT_MODEL_DIR"
        --data "${train_data_files[@]}"
        --gpus "$SPT_GPUS"
        --batch-size "$SPT_BATCH_SIZE"
        --learning-rate "$SPT_LEARNING_RATE"
        --query "$SPT_QUERY"
        --max_dc "$SPT_MAX_DC"
        --max-epochs "$SPT_MAX_EPOCHS")
fi

set +e
/usr/bin/time -v "${train_cmd[@]}" 2>&1 | tee "$benchmark_log"
train_status=${PIPESTATUS[0]}
set -e

wall_end_epoch=$(date +%s)
wall_end_human=$(date '+%Y-%m-%d %H:%M:%S %Z')
total_wall_seconds=$((wall_end_epoch - wall_start_epoch))

training_seconds=$(
    sed -n 's/.*Training takes \([0-9][0-9]*\) s.*/\1/p' "$benchmark_log" | tail -1
)
if [[ -n "$training_seconds" ]]; then
    startup_and_data_seconds=$((total_wall_seconds - training_seconds))
else
    training_seconds="unknown"
    startup_and_data_seconds="unknown"
fi

{
    echo "system=$SPT_SYSTEM"
    echo "run_name=$SPT_RUN_NAME"
    echo "run_tag=$run_tag"
    echo "job_id=${SLURM_JOB_ID:-unknown}"
    echo "array_task_id=${SLURM_ARRAY_TASK_ID:-none}"
    echo "train_data=$SPT_TRAIN_DATA"
    echo "max_files=$SPT_MAX_FILES"
    echo "matched_training_files=$match_count"
    echo "first_training_file=${train_data_files[0]}"
    echo "last_training_file=${train_data_files[$((match_count - 1))]}"
    echo "model_dir=$SPT_MODEL_DIR"
    echo "batch_size=$SPT_BATCH_SIZE"
    echo "max_epochs=$SPT_MAX_EPOCHS"
    echo "num_workers=$SPT_NUM_WORKERS"
    echo "disable_amp=$SPT_DISABLE_AMP"
    echo "disable_tf32=$SPT_DISABLE_TF32"
    echo "cudnn_benchmark=$SPT_CUDNN_BENCHMARK"
    echo "disable_plot=$SPT_DISABLE_PLOT"
    echo "amp_dtype=$SPT_AMP_DTYPE"
    echo "disable_grad_scaler=${SPT_DISABLE_GRAD_SCALER:-auto}"
    echo "wall_start=$wall_start_human"
    echo "wall_end=$wall_end_human"
    echo "total_wall_seconds=$total_wall_seconds"
    echo "training_seconds=$training_seconds"
    echo "startup_plus_data_loading_seconds=$startup_and_data_seconds"
    echo "exit_status=$train_status"
} | tee "$benchmark_metrics"

echo "Benchmark wall-clock end: $wall_end_human"
echo "Total script wall time: ${total_wall_seconds}s"
echo "Training-loop time reported by Python: ${training_seconds}s"
echo "Per-epoch timing written to: $SPT_MODEL_DIR/epoch_timing.csv"

if [[ "$train_status" -ne 0 ]]; then
    echo "ERROR: training command failed with exit status $train_status"
    exit "$train_status"
fi

echo "Benchmark job completed."
