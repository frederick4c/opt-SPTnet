#!/bin/bash
#SBATCH -J sptnet_diff_eval
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=sptnet_diff_eval_%j.out
#SBATCH --error=sptnet_diff_eval_%j.err
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
#   sbatch --export=ALL,SPT_BASE_SEED=68 slurm/generate_diff_eval_csd3.slurm
export SPT_NUM_FILES="${SPT_NUM_FILES:-100}"
export SPT_VIDEOS_PER_FILE="${SPT_VIDEOS_PER_FILE:-100}"
export SPT_OUTPUT_EXTENSION="${SPT_OUTPUT_EXTENSION:-.h5}"
export SPT_OUTPUT_ROOT="${SPT_OUTPUT_ROOT:-$SLURM_SUBMIT_DIR/diff_evals}"
export SPT_BASE_SEED="${SPT_BASE_SEED:-1889}"
export SPT_CONDITIONS="${SPT_CONDITIONS:-all}"
export SPT_LINK_CONDITIONS="${SPT_LINK_CONDITIONS:-$SPT_CONDITIONS}"

generated_dir="$SPT_OUTPUT_ROOT/generated"
mean_sweep_dir="$SPT_OUTPUT_ROOT/mean_sweep"
range_sweep_dir="$SPT_OUTPUT_ROOT/range_sweep"
manifest="${SPT_MANIFEST:-$SPT_OUTPUT_ROOT/manifest.csv}"

mkdir -p "$generated_dir" "$mean_sweep_dir" "$range_sweep_dir"

echo "Starting SPTnet diffusion evaluation data generation on $(hostname)"
echo "Output root: $SPT_OUTPUT_ROOT"
echo "Files/condition: $SPT_NUM_FILES"
echo "Videos/file: $SPT_VIDEOS_PER_FILE"
echo "Particles/video: 10"
echo "Output extension: $SPT_OUTPUT_EXTENSION"
echo "Base seed: $SPT_BASE_SEED"
echo "Generate conditions: $SPT_CONDITIONS"
echo "Link/manifest conditions: $SPT_LINK_CONDITIONS"
echo "Python: $(which python)"

printf "experiment,condition,path,d_min,d_max,target_mean,half_width,seed,num_files,videos_per_file,particles_per_video\n" > "$manifest"

condition_in_list() {
    local condition_list="$1"
    local condition="$2"

    if [[ "$condition_list" == "all" ]]; then
        return 0
    fi

    case ":$condition_list:" in
        *:"$condition":*) return 0 ;;
        *) return 1 ;;
    esac
}

generation_condition_selected() {
    local condition="$1"

    condition_in_list "$SPT_CONDITIONS" "$condition"
}

link_condition_selected() {
    local condition="$1"

    condition_in_list "$SPT_LINK_CONDITIONS" "$condition"
}

generate_condition() {
    local experiment="$1"
    local condition="$2"
    local d_min="$3"
    local d_max="$4"
    local target_mean="$5"
    local half_width="$6"
    local seed="$7"
    local output_dir="$generated_dir/$condition"

    if ! generation_condition_selected "$condition"; then
        echo "Skipping $condition; not listed in SPT_CONDITIONS."
        return 0
    fi

    mkdir -p "$output_dir"

    echo
    echo "Generating $condition"
    echo "  experiment: $experiment"
    echo "  D range: $d_min-$d_max"
    echo "  output: $output_dir"
    echo "  seed: $seed"

    SPT_D_MIN="$d_min" SPT_D_MAX="$d_max" sptnet-generate-training-data \
        --output-dir "$output_dir" \
        --output-extension "$SPT_OUTPUT_EXTENSION" \
        --num-files "$SPT_NUM_FILES" \
        --videos-per-file "$SPT_VIDEOS_PER_FILE" \
        --p-num-min 10 \
        --p-num-max 10 \
        --file-start 1 \
        --seed "$seed"

    # The manifest records experiment views after all symlinks are created.
}

link_condition() {
    local experiment_dir="$1"
    local condition="$2"
    local link_path="$experiment_dir/$condition"

    if ! link_condition_selected "$condition"; then
        return 0
    fi

    mkdir -p "$experiment_dir"
    ln -sfn "../generated/$condition" "$link_path"
}

record_condition_use() {
    local experiment="$1"
    local condition="$2"
    local d_min="$3"
    local d_max="$4"
    local target_mean="$5"
    local half_width="$6"
    local seed="$7"
    local path="$8"

    if ! link_condition_selected "$condition"; then
        return 0
    fi

    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,10\n" \
        "$experiment" "$condition" "$path" "$d_min" "$d_max" \
        "$target_mean" "$half_width" "$seed" "$SPT_NUM_FILES" "$SPT_VIDEOS_PER_FILE" \
        >> "$manifest"
}

# Unique datasets. The D=0.25 +/- 0.05 condition is generated once, then linked
# into both the mean-sweep and range-sweep views.
generate_condition "mean_sweep" "D_mean_0p05_pm_0p05" "0.00" "0.10" "0.05" "0.05" "$((SPT_BASE_SEED + 1000))"
generate_condition "mean_sweep" "D_mean_0p15_pm_0p05" "0.10" "0.20" "0.15" "0.05" "$((SPT_BASE_SEED + 2000))"
generate_condition "shared" "D_mean_0p25_pm_0p05" "0.20" "0.30" "0.25" "0.05" "$((SPT_BASE_SEED + 3000))"
generate_condition "mean_sweep" "D_mean_0p35_pm_0p05" "0.30" "0.40" "0.35" "0.05" "$((SPT_BASE_SEED + 4000))"
generate_condition "mean_sweep" "D_mean_0p45_pm_0p05" "0.40" "0.50" "0.45" "0.05" "$((SPT_BASE_SEED + 5000))"
generate_condition "range_sweep" "D_mean_0p25_pm_0p01" "0.24" "0.26" "0.25" "0.01" "$((SPT_BASE_SEED + 6000))"
generate_condition "range_sweep" "D_mean_0p25_pm_0p15" "0.10" "0.40" "0.25" "0.15" "$((SPT_BASE_SEED + 7000))"

link_condition "$mean_sweep_dir" "D_mean_0p05_pm_0p05"
link_condition "$mean_sweep_dir" "D_mean_0p15_pm_0p05"
link_condition "$mean_sweep_dir" "D_mean_0p25_pm_0p05"
link_condition "$mean_sweep_dir" "D_mean_0p35_pm_0p05"
link_condition "$mean_sweep_dir" "D_mean_0p45_pm_0p05"

link_condition "$range_sweep_dir" "D_mean_0p25_pm_0p01"
link_condition "$range_sweep_dir" "D_mean_0p25_pm_0p05"
link_condition "$range_sweep_dir" "D_mean_0p25_pm_0p15"

record_condition_use "mean_sweep" "D_mean_0p05_pm_0p05" "0.00" "0.10" "0.05" "0.05" "$((SPT_BASE_SEED + 1000))" "$mean_sweep_dir/D_mean_0p05_pm_0p05"
record_condition_use "mean_sweep" "D_mean_0p15_pm_0p05" "0.10" "0.20" "0.15" "0.05" "$((SPT_BASE_SEED + 2000))" "$mean_sweep_dir/D_mean_0p15_pm_0p05"
record_condition_use "mean_sweep" "D_mean_0p25_pm_0p05" "0.20" "0.30" "0.25" "0.05" "$((SPT_BASE_SEED + 3000))" "$mean_sweep_dir/D_mean_0p25_pm_0p05"
record_condition_use "mean_sweep" "D_mean_0p35_pm_0p05" "0.30" "0.40" "0.35" "0.05" "$((SPT_BASE_SEED + 4000))" "$mean_sweep_dir/D_mean_0p35_pm_0p05"
record_condition_use "mean_sweep" "D_mean_0p45_pm_0p05" "0.40" "0.50" "0.45" "0.05" "$((SPT_BASE_SEED + 5000))" "$mean_sweep_dir/D_mean_0p45_pm_0p05"
record_condition_use "range_sweep" "D_mean_0p25_pm_0p01" "0.24" "0.26" "0.25" "0.01" "$((SPT_BASE_SEED + 6000))" "$range_sweep_dir/D_mean_0p25_pm_0p01"
record_condition_use "range_sweep" "D_mean_0p25_pm_0p05" "0.20" "0.30" "0.25" "0.05" "$((SPT_BASE_SEED + 3000))" "$range_sweep_dir/D_mean_0p25_pm_0p05"
record_condition_use "range_sweep" "D_mean_0p25_pm_0p15" "0.10" "0.40" "0.25" "0.15" "$((SPT_BASE_SEED + 7000))" "$range_sweep_dir/D_mean_0p25_pm_0p15"

echo
echo "Diffusion evaluation data generation completed."
echo "Generated datasets: $generated_dir"
echo "Mean sweep view: $mean_sweep_dir"
echo "Range sweep view: $range_sweep_dir"
echo "Manifest: $manifest"
