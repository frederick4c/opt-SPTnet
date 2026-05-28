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
export PYTHONUNBUFFERED=1

echo "Submit directory: $SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"

if [[ ! -f pyproject.toml || ! -d src/sptnet ]]; then
    echo "ERROR: pyproject.toml and src/sptnet were not found in $(pwd)."
    echo "Submit the job from this repo root."
    exit 1
fi

if ! command -v sptnet-generate-training-data >/dev/null 2>&1; then
    echo "ERROR: sptnet-generate-training-data was not found in PATH."
    echo "Install this repo in the active environment with: python -m pip install -e ."
    exit 1
fi

# Override these at submit time if needed, e.g.:
#   sbatch --export=ALL,SPT_NUM_FILES=50,SPT_VIDEOS_PER_FILE=100,SPT_SEED=123 generate_trainingdata_csd3.slurm
export SPT_NUM_FILES="${SPT_NUM_FILES:-10}"
export SPT_VIDEOS_PER_FILE="${SPT_VIDEOS_PER_FILE:-100}"
export SPT_OUTPUT_DIR="${SPT_OUTPUT_DIR:-$SLURM_SUBMIT_DIR/DenseTrain}"
export SPT_OUTPUT_EXTENSION="${SPT_OUTPUT_EXTENSION:-.h5}"

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
echo "Output extension: $SPT_OUTPUT_EXTENSION"
echo "Particles/video: ${SPT_P_NUM_MIN:-1}-${SPT_P_NUM_MAX:-10}"
echo "Diffusion range: ${SPT_D_MIN:-0.001}-${SPT_D_MAX:-0.5}"

generator_args=(
    --output-dir "$SPT_OUTPUT_DIR"
    --output-extension "$SPT_OUTPUT_EXTENSION"
    --num-files "$SPT_NUM_FILES"
    --videos-per-file "$SPT_VIDEOS_PER_FILE"
    --file-start "${SPT_FILE_START:-1}"
)

if [[ -n "${SPT_SEED:-}" ]]; then generator_args+=(--seed "$SPT_SEED"); fi
if [[ -n "${SPT_FRAMES:-}" ]]; then generator_args+=(--frames "$SPT_FRAMES"); fi
if [[ -n "${SPT_IMAGE_DIMS:-}" ]]; then generator_args+=(--image-dims "$SPT_IMAGE_DIMS"); fi
if [[ -n "${SPT_P_NUM_MIN:-}" ]]; then generator_args+=(--p-num-min "$SPT_P_NUM_MIN"); fi
if [[ -n "${SPT_P_NUM_MAX:-}" ]]; then generator_args+=(--p-num-max "$SPT_P_NUM_MAX"); fi
if [[ -n "${SPT_PHOTON_MIN:-}" ]]; then generator_args+=(--photon-min "$SPT_PHOTON_MIN"); fi
if [[ -n "${SPT_PHOTON_MAX:-}" ]]; then generator_args+=(--photon-max "$SPT_PHOTON_MAX"); fi
if [[ -n "${SPT_BG_MIN:-}" ]]; then generator_args+=(--bg-min "$SPT_BG_MIN"); fi
if [[ -n "${SPT_BG_MAX:-}" ]]; then generator_args+=(--bg-max "$SPT_BG_MAX"); fi
if [[ -n "${SPT_PERLIN_BG_MIN:-}" ]]; then generator_args+=(--perlin-bg-min "$SPT_PERLIN_BG_MIN"); fi
if [[ -n "${SPT_PERLIN_BG_MAX:-}" ]]; then generator_args+=(--perlin-bg-max "$SPT_PERLIN_BG_MAX"); fi
if [[ "${SPT_MOTION_BLUR:-0}" == "1" || "${SPT_MOTION_BLUR:-}" == "true" ]]; then generator_args+=(--motion-blur); fi
if [[ "${SPT_NO_PROGRESS:-0}" == "1" || "${SPT_NO_PROGRESS:-}" == "true" ]]; then generator_args+=(--no-progress); fi

sptnet-generate-training-data "${generator_args[@]}"

echo "Generation job completed."
