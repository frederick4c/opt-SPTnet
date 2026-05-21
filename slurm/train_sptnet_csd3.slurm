#!/bin/bash
#SBATCH -J SPTnet_train              # Job name
#SBATCH -A MPHIL-DIS-SL2-GPU         # Project / Account
#SBATCH -p ampere                    # Partition (queue) name for Ampere A100 nodes
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of scripts to run
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --gres=gpu:1                 # Number of GPUs per node (A100)
#SBATCH --time=00:30:00              # Time limit hrs:min:sec
#SBATCH --output=sptnet_train_%j.out # Standard output log (%j = job ID)
#SBATCH --error=sptnet_train_%j.err  # Standard error log
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
source "${SPT_VENV:-$HOME/envs/SPTnet}/bin/activate"

# ---------------------------------------------------------------------
# Run the application
# ---------------------------------------------------------------------

# Go to the directory where the job was submitted from
cd "$SLURM_SUBMIT_DIR"

# Prevent standard python multiprocessing from taking all CPU cores
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
export SPT_NUM_WORKERS="${SPT_NUM_WORKERS:-2}"

echo "Diagnostic info for virtual environment:"
which python
python -c "import sys; print('python=', sys.executable); print('sys.path[0:5]=', sys.path[:5])"
python -c "import torch; print('torch.cuda.is_available=', torch.cuda.is_available())"

if [[ ! -f "src/sptnet/training/cli.py" ]]; then
    echo "ERROR: src/sptnet/training/cli.py was not found in $(pwd)"
    echo "Submit the job from the refactored SPTnet repo root."
    exit 1
fi

export SPT_MODEL_DIR="${SPT_MODEL_DIR:-./Trained_models}"
export SPT_TRAIN_DATA="${SPT_TRAIN_DATA:-./TestData/eg_trainingvideos_*.h5}"
export SPT_GPUS="${SPT_GPUS:-1}"
export SPT_BATCH_SIZE="${SPT_BATCH_SIZE:-16}"
export SPT_QUERY="${SPT_QUERY:-20}"
export SPT_LEARNING_RATE="${SPT_LEARNING_RATE:-0.0001}"
export SPT_MAX_DC="${SPT_MAX_DC:-0.5}"
export SPT_MAX_EPOCHS="${SPT_MAX_EPOCHS:-0}"
export SPT_CRLB_PATH="${SPT_CRLB_PATH:-}"

echo "Starting SPTnet training on CSD3 Ampere node..."
echo "Model dir: $SPT_MODEL_DIR"
echo "Training data pattern: $SPT_TRAIN_DATA"
echo "GPUs: $SPT_GPUS"
echo "Batch size: $SPT_BATCH_SIZE"
echo "Query count: $SPT_QUERY"
echo "Learning rate: $SPT_LEARNING_RATE"
echo "Max diffusion coefficient: $SPT_MAX_DC"
echo "Max epochs: $SPT_MAX_EPOCHS"

crlb_args=()
if [[ -n "$SPT_CRLB_PATH" ]]; then
    crlb_args=(--crlb-path "$SPT_CRLB_PATH")
fi

python -u -m sptnet.training.cli \
    --model-dir "$SPT_MODEL_DIR" \
    --data "$SPT_TRAIN_DATA" \
    --gpus "$SPT_GPUS" \
    --batch-size "$SPT_BATCH_SIZE" \
    --learning-rate "$SPT_LEARNING_RATE" \
    --query "$SPT_QUERY" \
    --max_dc "$SPT_MAX_DC" \
    --max-epochs "$SPT_MAX_EPOCHS" \
    "${crlb_args[@]}"

echo "Training job completed."
