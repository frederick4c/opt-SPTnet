#!/bin/bash
#SBATCH -J sptnet_diff_infer
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=sptnet_diff_infer_%j.out
#SBATCH --error=sptnet_diff_infer_%j.err
#SBATCH --export=ALL

set -euo pipefail

ts() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*"; }

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load python/3.11.0-icl

cd "$SLURM_SUBMIT_DIR"
source "${SPT_VENV:-$HOME/envs/SPTnet}/bin/activate"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if [[ ! -f pyproject.toml || ! -d src/sptnet ]]; then
    echo "ERROR: pyproject.toml and src/sptnet were not found in $(pwd)."
    echo "Submit the job from this repo root."
    exit 1
fi

if ! command -v sptnet-inference >/dev/null 2>&1; then
    echo "ERROR: sptnet-inference was not found in PATH."
    echo "Install this repo in the active environment with: python -m pip install -e ."
    exit 1
fi

export SPT_MODEL_PATH="${SPT_MODEL_PATH:-../SPTnet/Trained_models/dense_sparse/trained_model}"
export SPT_DIFF_EVAL_ROOT="${SPT_DIFF_EVAL_ROOT:-$SLURM_SUBMIT_DIR/diff_evals}"
export SPT_DIFF_EVAL_INPUT_ROOT="${SPT_DIFF_EVAL_INPUT_ROOT:-$SPT_DIFF_EVAL_ROOT/generated}"
export SPT_INFER_BATCH_SIZE="${SPT_INFER_BATCH_SIZE:-8}"
export SPT_HDF5_CLIP_INDEX="${SPT_HDF5_CLIP_INDEX:-${SPT_MAT_CLIP_INDEX:-0}}"
export SPT_CONDITIONS="${SPT_CONDITIONS:-all}"

if [[ ! -f "$SPT_MODEL_PATH" ]]; then
    echo "ERROR: model file not found: $SPT_MODEL_PATH"
    exit 1
fi

if [[ ! -d "$SPT_DIFF_EVAL_INPUT_ROOT" ]]; then
    echo "ERROR: diffusion evaluation input directory not found: $SPT_DIFF_EVAL_INPUT_ROOT"
    exit 1
fi

condition_selected() {
    local condition="$1"

    if [[ "$SPT_CONDITIONS" == "all" ]]; then
        return 0
    fi

    case ":$SPT_CONDITIONS:" in
        *:"$condition":*) return 0 ;;
        *) return 1 ;;
    esac
}

mapfile -t condition_dirs < <(find "$SPT_DIFF_EVAL_INPUT_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)

if [[ "${#condition_dirs[@]}" -eq 0 ]]; then
    echo "ERROR: no condition directories found under $SPT_DIFF_EVAL_INPUT_ROOT"
    exit 1
fi

log "Starting SPTnet diffusion evaluation inference"
log "Model path: $SPT_MODEL_PATH"
log "Input root: $SPT_DIFF_EVAL_INPUT_ROOT"
log "Result layout: $SPT_DIFF_EVAL_INPUT_ROOT/<condition>/inference/inference_results"
log "Batch size: $SPT_INFER_BATCH_SIZE"
log "HDF5 clip index: $SPT_HDF5_CLIP_INDEX"
log "Conditions: $SPT_CONDITIONS"
log "Python: $(which python)"
python -c "import torch; print('torch.cuda.is_available=', torch.cuda.is_available())"

for condition_dir in "${condition_dirs[@]}"; do
    condition="$(basename "$condition_dir")"

    if ! condition_selected "$condition"; then
        log "Skipping $condition; not listed in SPT_CONDITIONS."
        continue
    fi

    if ! compgen -G "$condition_dir/trainingvideos_*.h5" >/dev/null; then
        log "Skipping $condition; no trainingvideos_*.h5 files found."
        continue
    fi

    result_dir="$condition_dir/inference"
    mkdir -p "$result_dir"
    ln -sfn "$(realpath "$SPT_MODEL_PATH")" "$result_dir/trained_model"

    log "Running inference for $condition"
    log "Input pattern: $condition_dir/trainingvideos_*.h5"
    log "Condition result dir: $result_dir/inference_results"

    sptnet-inference \
        --model-path "$result_dir/trained_model" \
        --data "$condition_dir/trainingvideos_*.h5" \
        --batch-size "$SPT_INFER_BATCH_SIZE" \
        --hdf5-clip-index "$SPT_HDF5_CLIP_INDEX"
done

log "Diffusion evaluation inference completed."
log "Results saved under: $SPT_DIFF_EVAL_INPUT_ROOT/<condition>/inference/inference_results"
