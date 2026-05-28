#!/bin/bash
#SBATCH -J SPTnet_infer_par           # Job name
#SBATCH -A MPHIL-DIS-SL2-GPU         # Project / Account
#SBATCH -p ampere                    # Partition (queue) name for Ampere A100 nodes
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of scripts to run
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --gres=gpu:1                 # Number of GPUs per node (A100)
#SBATCH --time=00:30:00              # Time limit hrs:min:sec
#SBATCH --output=sptnet_infer_par_%j.out # Standard output log (%j = job ID)
#SBATCH --error=sptnet_infer_par_%j.err  # Standard error log
#SBATCH --export=ALL                 # Export all environment variables to the job

set -euo pipefail

ts() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*"; }

log "Job starting: id=${SLURM_JOB_ID:-unknown} host=$(hostname) pwd=$(pwd)"
log "Submit dir: ${SLURM_SUBMIT_DIR:-unset}"
touch "sptnet_infer_par_${SLURM_JOB_ID:-manual}.start"

# ---------------------------------------------------------------------
# Module Loading & Environment Setup 
# ---------------------------------------------------------------------

# Purge existing modules and load standard Ampere environment for Rocky/RHEL 8
log "Loading modules..."
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
log "Changed to submit dir: $(pwd)"
log "Activating virtual environment..."
source "${SPT_VENV:-$HOME/envs/SPTnet}/bin/activate"

# Prevent standard python multiprocessing from taking all CPU cores
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

log "Diagnostic info for virtual environment:"
which python
python -c "import sys; print('python=', sys.executable); print('sys.path[0:5]=', sys.path[:5])"
python -c "import torch; print('torch.cuda.is_available=', torch.cuda.is_available())"

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

export SPT_MODEL_PATH="${SPT_MODEL_PATH:-./Trained_models/full_run/trained_model}"
export SPT_INFER_DATA="${SPT_INFER_DATA:-./TestData/testdata_*.h5}"
export SPT_INFER_BATCH_SIZE="${SPT_INFER_BATCH_SIZE:-8}"
export SPT_HDF5_CLIP_INDEX="${SPT_HDF5_CLIP_INDEX:-${SPT_MAT_CLIP_INDEX:-0}}"

log "Starting Parallel SPTnet inference on CSD3 Ampere node..."
log "Model path: $SPT_MODEL_PATH"
log "Inference data pattern: $SPT_INFER_DATA"
log "Batch size: $SPT_INFER_BATCH_SIZE"
log "HDF5 clip index: $SPT_HDF5_CLIP_INDEX"

sptnet-inference \
    --model-path "$SPT_MODEL_PATH" \
    --data "$SPT_INFER_DATA" \
    --batch-size "$SPT_INFER_BATCH_SIZE" \
    --hdf5-clip-index "$SPT_HDF5_CLIP_INDEX"

log "Inference job completed."
