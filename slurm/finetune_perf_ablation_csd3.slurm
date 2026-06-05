#!/bin/bash
#SBATCH -J SPT_ft_ablate
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --array=0-4
#SBATCH --output=finetune_perf_ablation_%A_%a.out
#SBATCH --error=finetune_perf_ablation_%A_%a.err
#SBATCH --export=ALL

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
ulimit -n 65536

export SPT_TRAIN_DATA="${SPT_TRAIN_DATA:-./diff_bins/*/*.h5}"
export SPT_PRETRAINED="${SPT_PRETRAINED:-./perf_exp/trained_model}"
export SPT_OUTPUT_BASE="${SPT_OUTPUT_BASE:-./perf_exp}"
export SPT_MAX_EPOCHS="${SPT_MAX_EPOCHS:-20}"
export SPT_LEARNING_RATE="${SPT_LEARNING_RATE:-1e-5}"
export SPT_BATCH_SIZE="${SPT_BATCH_SIZE:-16}"
export SPT_QUERY="${SPT_QUERY:-20}"
export SPT_MAX_DC="${SPT_MAX_DC:-0.5}"
export SPT_PATIENCE="${SPT_PATIENCE:-20}"
export SPT_VAL_EVERY="${SPT_VAL_EVERY:-1}"
export SPT_GPUS="${SPT_GPUS:-1}"
export SPT_GRAD_CLIP="${SPT_GRAD_CLIP:-1.0}"
export SPT_CRLB_PATH="${SPT_CRLB_PATH:-}"
export SPT_MAX_TRAIN_BATCHES="${SPT_MAX_TRAIN_BATCHES:-}"
export SPT_MAX_VAL_BATCHES="${SPT_MAX_VAL_BATCHES:-}"
export SPT_RELATIVE_D_EPS="${SPT_RELATIVE_D_EPS:-0.01}"
export SPT_HD_UP_WEIGHT="${SPT_HD_UP_WEIGHT:-1.0}"
export SPT_ALLOW_EXISTING_OUTPUT="${SPT_ALLOW_EXISTING_OUTPUT:-0}"

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

if [[ ! -f "$SPT_PRETRAINED" ]]; then
    echo "ERROR: pretrained checkpoint not found: $SPT_PRETRAINED"
    exit 1
fi

matched_files=$(python -c 'import glob, sys; print(len(glob.glob(sys.argv[1])))' "$SPT_TRAIN_DATA")

if [[ "$matched_files" -eq 0 ]]; then
    echo "ERROR: training data pattern matched no files: $SPT_TRAIN_DATA"
    exit 1
fi

task_id="${SLURM_ARRAY_TASK_ID:-0}"
variant_args=()

case "$task_id" in
    0)
        variant="baseline"
        ;;
    1)
        variant="hd_up_everywhere"
        variant_args+=(
            --match-h-weight "$SPT_HD_UP_WEIGHT"
            --match-d-weight "$SPT_HD_UP_WEIGHT"
            --loss-h-weight "$SPT_HD_UP_WEIGHT"
            --loss-d-weight "$SPT_HD_UP_WEIGHT"
        )
        ;;
    2)
        variant="hd_down_match"
        variant_args+=(
            --match-h-weight 0.0
            --match-d-weight 0.0
        )
        ;;
    3)
        variant="relative_d"
        variant_args+=(
            --diffusion-loss relative
            --relative-d-eps "$SPT_RELATIVE_D_EPS"
        )
        ;;
    4)
        variant="bce_logits"
        variant_args+=(
            --objectness-loss bce_logits
        )
        ;;
    *)
        echo "ERROR: unknown array task id: $task_id"
        exit 1
        ;;
esac

SPT_MODEL_DIR="$SPT_OUTPUT_BASE/finetune_ablation_$variant"

if [[ "$SPT_MAX_EPOCHS" -le 1 ]]; then
    echo "WARNING: SPT_MAX_EPOCHS=$SPT_MAX_EPOCHS; this is suitable only for a dry run."
fi

existing_history="$SPT_MODEL_DIR/trained_modelloss_history.csv"
if [[ -f "$existing_history" && "$SPT_ALLOW_EXISTING_OUTPUT" != "1" ]]; then
    echo "ERROR: existing ablation history found: $existing_history"
    echo "Use a clean SPT_OUTPUT_BASE, remove the previous dry-run directory, or submit with SPT_ALLOW_EXISTING_OUTPUT=1 to continue it."
    exit 1
fi

mkdir -p "$SPT_MODEL_DIR"

run_log="$SPT_MODEL_DIR/slurm_finetune_${SLURM_JOB_ID:-manual}_${task_id}.log"
run_metrics="$SPT_MODEL_DIR/slurm_finetune_${SLURM_JOB_ID:-manual}_${task_id}_metrics.txt"

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
    --resume "$SPT_PRETRAINED"
    --no-resume-optimizer
)

if [[ -n "$SPT_CRLB_PATH" ]]; then
    cmd+=(--crlb-path "$SPT_CRLB_PATH")
fi
if [[ -n "$SPT_MAX_TRAIN_BATCHES" ]]; then
    cmd+=(--max-train-batches "$SPT_MAX_TRAIN_BATCHES")
fi
if [[ -n "$SPT_MAX_VAL_BATCHES" ]]; then
    cmd+=(--max-val-batches "$SPT_MAX_VAL_BATCHES")
fi
cmd+=("${variant_args[@]}")

printf -v exact_command '%q ' "${cmd[@]}"

echo "Starting SPTnet fine-tuning ablation on CSD3..."
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Array task: $task_id"
echo "Variant: $variant"
echo "Host: $(hostname)"
echo "Submit directory: $SLURM_SUBMIT_DIR"
echo "Training data pattern: $SPT_TRAIN_DATA"
echo "Matched training files: $matched_files"
echo "Pretrained checkpoint: $SPT_PRETRAINED"
echo "Output directory: $SPT_MODEL_DIR"
echo "Output checkpoint: $SPT_MODEL_DIR/trained_model"
echo "Batch size: $SPT_BATCH_SIZE"
echo "Learning rate: $SPT_LEARNING_RATE"
echo "Max epochs: $SPT_MAX_EPOCHS"
echo "Patience: $SPT_PATIENCE"
echo "Val every: $SPT_VAL_EVERY"
echo "Python: $(which python)"
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
    echo "variant=$variant"
    echo "array_task=$task_id"
    echo "job_id=${SLURM_JOB_ID:-unknown}"
    echo "train_data=$SPT_TRAIN_DATA"
    echo "matched_training_files=$matched_files"
    echo "pretrained=$SPT_PRETRAINED"
    echo "model_dir=$SPT_MODEL_DIR"
    echo "checkpoint=$SPT_MODEL_DIR/trained_model"
    echo "wall_start=$wall_start_human"
    echo "wall_end=$wall_end_human"
    echo "total_wall_seconds=$total_wall_seconds"
    echo "exit_status=$train_status"
    echo "exact_command=$exact_command"
} | tee "$run_metrics"

if [[ "$train_status" -ne 0 ]]; then
    echo "ERROR: fine-tuning failed with exit status $train_status"
    exit "$train_status"
fi

echo "Fine-tuning ablation completed."
echo "Variant: $variant"
echo "Checkpoint: $SPT_MODEL_DIR/trained_model"
echo "Metrics: $run_metrics"

# Dry-run examples:
#   sbatch --array=0 --export=ALL,SPT_OUTPUT_BASE=./perf_exp/dry_run,SPT_MAX_EPOCHS=1,SPT_MAX_TRAIN_BATCHES=2,SPT_MAX_VAL_BATCHES=1 slurm/finetune_perf_ablation_csd3.slurm
#   sbatch --array=0-4 slurm/finetune_perf_ablation_csd3.slurm
