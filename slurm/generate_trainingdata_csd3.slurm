#!/bin/bash
#SBATCH -J sptnet_gen
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=sptnet_gen_%j.out
#SBATCH --error=sptnet_gen_%j.err
#SBATCH --export=ALL

set -euo pipefail

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl
module load python/3.11.0-icl

cd "$SLURM_SUBMIT_DIR"
source "${SPT_VENV:-$HOME/envs/SPTnet}/bin/activate"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"

echo "Submit directory: $SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"

if [[ ! -f "src/sptnet/training/data_generator.py" ]]; then
    echo "ERROR: src/sptnet/training/data_generator.py was not found in $(pwd)"
    echo "Submit the job from the refactored SPTnet repo root."
    exit 1
fi

# Override these at submit time if needed, e.g.:
#   sbatch --export=ALL,SPT_NUM_FILES=50,SPT_VIDEOS_PER_FILE=100,SPT_SEED=123 generate_trainingdata_csd3.slurm
export SPT_NUM_FILES="${SPT_NUM_FILES:-10}"
export SPT_VIDEOS_PER_FILE="${SPT_VIDEOS_PER_FILE:-100}"
export SPT_OUTPUT_DIR="${SPT_OUTPUT_DIR:-$SLURM_SUBMIT_DIR/DenseTrain}"

# If submitted as an array, each task writes a separate contiguous block.
# Example: --array=1-100 with SPT_NUM_FILES=10 writes files 1-1000.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -z "${SPT_FILE_START:-}" ]]; then
    export SPT_FILE_START=$(( (SLURM_ARRAY_TASK_ID - 1) * SPT_NUM_FILES + 1 ))
fi

mkdir -p "$SPT_OUTPUT_DIR"

echo "Starting SPTnet training-data generation on $(hostname)"
echo "Output dir: $SPT_OUTPUT_DIR"
echo "Files: $SPT_NUM_FILES, videos/file: $SPT_VIDEOS_PER_FILE"
echo "File start: ${SPT_FILE_START:-1}"
echo "Seed: ${SPT_SEED:-shuffle}"

python -u -m sptnet.training.data_generator

echo "Generation job completed."
