#!/bin/bash
#SBATCH -J SPTnet_final              # Job name
#SBATCH -A MPHIL-DIS-SL2-GPU         # Project / Account
#SBATCH -p ampere                    # Partition (queue) name for Ampere A100 nodes
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of scripts to run
#SBATCH --cpus-per-task=8            # Number of CPU cores per task
#SBATCH --gres=gpu:1                 # Number of GPUs per node (A100)
#SBATCH --time=20:00:00              # Time limit hrs:min:sec
#SBATCH --output=sptnet_final_%j.out # Standard output log (%j = job ID)
#SBATCH --error=sptnet_final_%j.err  # Standard error log
#SBATCH --export=ALL                 # Export all environment variables to the job

set -euo pipefail

# ---------------------------------------------------------------------
# Module Loading & Environment Setup 
# ---------------------------------------------------------------------

# Purge existing modules and load standard Ampere environment for Rocky/RHEL 8
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

# Load the standard Python 3 module (adjust if CSD3 uses a specific python/3.x module)
module load python/3.11.0-icl

# ---------------------------------------------------------------------
# Run the application
# ---------------------------------------------------------------------

# Go to the directory where the job was submitted from
cd "$SLURM_SUBMIT_DIR"
source "${SPT_VENV:-$HOME/envs/SPTnet}/bin/activate"

# Prevent standard python multiprocessing from taking all CPU cores
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export SPT_NUM_WORKERS="${SPT_NUM_WORKERS:-2}"
ulimit -n 65536

echo "Diagnostic info for virtual environment:"
which python
python -c "import sys; print('python=', sys.executable); print('sys.path[0:5]=', sys.path[:5])"
python -c "import torch; print('torch.cuda.is_available=', torch.cuda.is_available())"

if [[ ! -f pyproject.toml || ! -d src/sptnet ]]; then
    echo "ERROR: pyproject.toml and src/sptnet were not found in $(pwd)."
    echo "Submit the job from this repo root."
    exit 1
fi

if ! command -v sptnet-train >/dev/null 2>&1; then
    echo "ERROR: sptnet-train was not found in PATH."
    echo "Install this repo in the active environment with: python -m pip install -e ."
    exit 1
fi

export SPT_MODEL_DIR="${SPT_MODEL_DIR:-./perf_exp/final_full_model}"
export SPT_TRAIN_DATA="${SPT_TRAIN_DATA:-../SPTnet/TrainData/trainingvideos_*.mat}"
export SPT_GPUS="${SPT_GPUS:-1}"
export SPT_BATCH_SIZE="${SPT_BATCH_SIZE:-16}"
export SPT_QUERY="${SPT_QUERY:-20}"
export SPT_LEARNING_RATE="${SPT_LEARNING_RATE:-0.0001}"
export SPT_MAX_DC="${SPT_MAX_DC:-0.5}"
export SPT_MAX_EPOCHS="${SPT_MAX_EPOCHS:-0}"
export SPT_CRLB_PATH="${SPT_CRLB_PATH:-}"
export SPT_VAL_EVERY="${SPT_VAL_EVERY:-1}"
export SPT_PATIENCE="${SPT_PATIENCE:-5}"
export SPT_GRAD_CLIP="${SPT_GRAD_CLIP:-1.0}"
export SPT_MAX_TRAIN_BATCHES="${SPT_MAX_TRAIN_BATCHES:-}"
export SPT_MAX_VAL_BATCHES="${SPT_MAX_VAL_BATCHES:-}"
export SPT_RESUME="${SPT_RESUME:-}"
export SPT_RESUME_OPTIMIZER="${SPT_RESUME_OPTIMIZER:-}"
export SPT_NO_RESUME_OPTIMIZER="${SPT_NO_RESUME_OPTIMIZER:-0}"
export SPT_RESUME_HISTORY="${SPT_RESUME_HISTORY:-}"
export SPT_MATCH_H_WEIGHT="${SPT_MATCH_H_WEIGHT:-0.0}"
export SPT_MATCH_D_WEIGHT="${SPT_MATCH_D_WEIGHT:-0.0}"
export SPT_LOSS_H_WEIGHT="${SPT_LOSS_H_WEIGHT:-0.5}"
export SPT_LOSS_D_WEIGHT="${SPT_LOSS_D_WEIGHT:-0.5}"
export SPT_HURST_LOSS="${SPT_HURST_LOSS:-log}"
export SPT_DIFFUSION_LOSS="${SPT_DIFFUSION_LOSS:-log}"
export SPT_LOG_H_EPS="${SPT_LOG_H_EPS:-0.0001}"
export SPT_LOG_D_EPS="${SPT_LOG_D_EPS:-0.0001}"
export SPT_OBJECTNESS_LOSS="${SPT_OBJECTNESS_LOSS:-bce_logits}"
export SPT_ALLOW_EXISTING_OUTPUT="${SPT_ALLOW_EXISTING_OUTPUT:-0}"
export SPT_EXPECTED_FILES="${SPT_EXPECTED_FILES:-100000}"

matched_files=$(python -c 'import glob, sys; print(len(glob.glob(sys.argv[1])))' "$SPT_TRAIN_DATA")

if [[ "$matched_files" -eq 0 ]]; then
    echo "ERROR: training data pattern matched no files: $SPT_TRAIN_DATA"
    exit 1
fi
if [[ "$SPT_EXPECTED_FILES" -gt 0 && "$matched_files" -ne "$SPT_EXPECTED_FILES" ]]; then
    echo "WARNING: expected $SPT_EXPECTED_FILES training files but matched $matched_files."
    echo "Set SPT_EXPECTED_FILES=0 to disable this warning for dry runs or subsets."
fi

existing_history="$SPT_MODEL_DIR/trained_modelloss_history.csv"
if [[ -z "$SPT_RESUME" && -f "$existing_history" && "$SPT_ALLOW_EXISTING_OUTPUT" != "1" ]]; then
    echo "ERROR: existing training history found: $existing_history"
    echo "Use a clean SPT_MODEL_DIR, set SPT_RESUME, or submit with SPT_ALLOW_EXISTING_OUTPUT=1."
    exit 1
fi

mkdir -p "$SPT_MODEL_DIR"

run_log="$SPT_MODEL_DIR/slurm_final_${SLURM_JOB_ID:-manual}.log"
run_metrics="$SPT_MODEL_DIR/slurm_final_${SLURM_JOB_ID:-manual}_metrics.txt"

echo "Starting SPTnet training on CSD3 Ampere node..."
echo "Model dir: $SPT_MODEL_DIR"
echo "Training data pattern: $SPT_TRAIN_DATA"
echo "Matched training files: $matched_files"
echo "GPUs: $SPT_GPUS"
echo "Batch size: $SPT_BATCH_SIZE"
echo "Query count: $SPT_QUERY"
echo "Learning rate: $SPT_LEARNING_RATE"
echo "Max diffusion coefficient: $SPT_MAX_DC"
echo "Max epochs: $SPT_MAX_EPOCHS"
echo "Patience: $SPT_PATIENCE"
echo "Val every: $SPT_VAL_EVERY"
echo "Loss settings: objectness=$SPT_OBJECTNESS_LOSS, H=$SPT_HURST_LOSS, D=$SPT_DIFFUSION_LOSS"
echo "Matching weights: H=$SPT_MATCH_H_WEIGHT, D=$SPT_MATCH_D_WEIGHT"
echo "Final H/D loss weights: H=$SPT_LOSS_H_WEIGHT, D=$SPT_LOSS_D_WEIGHT"

crlb_args=()
if [[ -n "$SPT_CRLB_PATH" ]]; then
    crlb_args=(--crlb-path "$SPT_CRLB_PATH")
fi

optional_args=()
if [[ -n "$SPT_MAX_TRAIN_BATCHES" ]]; then optional_args+=(--max-train-batches "$SPT_MAX_TRAIN_BATCHES"); fi
if [[ -n "$SPT_MAX_VAL_BATCHES" ]]; then optional_args+=(--max-val-batches "$SPT_MAX_VAL_BATCHES"); fi
if [[ -n "$SPT_RESUME" ]]; then optional_args+=(--resume "$SPT_RESUME"); fi
if [[ -n "$SPT_RESUME_OPTIMIZER" ]]; then optional_args+=(--resume-optimizer "$SPT_RESUME_OPTIMIZER"); fi
if [[ "$SPT_NO_RESUME_OPTIMIZER" == "1" ]]; then optional_args+=(--no-resume-optimizer); fi
if [[ -n "$SPT_RESUME_HISTORY" ]]; then optional_args+=(--resume-history "$SPT_RESUME_HISTORY"); fi

cmd=(
    sptnet-train
    --model-dir "$SPT_MODEL_DIR"
    --data "$SPT_TRAIN_DATA"
    --gpus "$SPT_GPUS"
    --batch-size "$SPT_BATCH_SIZE"
    --learning-rate "$SPT_LEARNING_RATE"
    --query "$SPT_QUERY"
    --max_dc "$SPT_MAX_DC"
    --max-epochs "$SPT_MAX_EPOCHS"
    --patience "$SPT_PATIENCE"
    --val-every "$SPT_VAL_EVERY"
    --grad-clip "$SPT_GRAD_CLIP"
    --match-h-weight "$SPT_MATCH_H_WEIGHT"
    --match-d-weight "$SPT_MATCH_D_WEIGHT"
    --loss-h-weight "$SPT_LOSS_H_WEIGHT"
    --loss-d-weight "$SPT_LOSS_D_WEIGHT"
    --hurst-loss "$SPT_HURST_LOSS"
    --diffusion-loss "$SPT_DIFFUSION_LOSS"
    --log-h-eps "$SPT_LOG_H_EPS"
    --log-d-eps "$SPT_LOG_D_EPS"
    --objectness-loss "$SPT_OBJECTNESS_LOSS"
    "${optional_args[@]}"
    "${crlb_args[@]}"
)

printf -v exact_command '%q ' "${cmd[@]}"

echo "Output checkpoint: $SPT_MODEL_DIR/trained_model"
echo "Exact command: $exact_command"
echo "Run log: $run_log"

wall_start_epoch=$(date +%s)
wall_start_human=$(date '+%Y-%m-%d %H:%M:%S %Z')

set +e
/usr/bin/time -v "${cmd[@]}" 2>&1 | tee "$run_log"
train_status=${PIPESTATUS[0]}
set -e

wall_end_epoch=$(date +%s)
wall_end_human=$(date '+%Y-%m-%d %H:%M:%S %Z')
total_wall_seconds=$((wall_end_epoch - wall_start_epoch))

{
    echo "job_id=${SLURM_JOB_ID:-unknown}"
    echo "train_data=$SPT_TRAIN_DATA"
    echo "matched_training_files=$matched_files"
    echo "model_dir=$SPT_MODEL_DIR"
    echo "checkpoint=$SPT_MODEL_DIR/trained_model"
    echo "wall_start=$wall_start_human"
    echo "wall_end=$wall_end_human"
    echo "total_wall_seconds=$total_wall_seconds"
    echo "exit_status=$train_status"
    echo "objectness_loss=$SPT_OBJECTNESS_LOSS"
    echo "hurst_loss=$SPT_HURST_LOSS"
    echo "diffusion_loss=$SPT_DIFFUSION_LOSS"
    echo "match_h_weight=$SPT_MATCH_H_WEIGHT"
    echo "match_d_weight=$SPT_MATCH_D_WEIGHT"
    echo "loss_h_weight=$SPT_LOSS_H_WEIGHT"
    echo "loss_d_weight=$SPT_LOSS_D_WEIGHT"
    echo "exact_command=$exact_command"
} | tee "$run_metrics"

if [[ "$train_status" -ne 0 ]]; then
    echo "ERROR: final training failed with exit status $train_status"
    exit "$train_status"
fi

echo "Training job completed."
echo "Checkpoint: $SPT_MODEL_DIR/trained_model"
echo "Metrics: $run_metrics"

# Dry-run example:
#   sbatch --export=ALL,SPT_MODEL_DIR=./perf_exp/final_full_dry_run,SPT_EXPECTED_FILES=0,SPT_MAX_EPOCHS=1,SPT_MAX_TRAIN_BATCHES=2,SPT_MAX_VAL_BATCHES=1 slurm/train_sptnet_csd3.slurm
#
# Full final run:
#   sbatch slurm/train_sptnet_csd3.slurm
