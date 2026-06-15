# SPTnet training-speed benchmark

Controlled, repeated, bootstrapped comparison of `opt-SPTnet` (this repo) against
the original `../SPTnet`, used for the report's runtime-optimisation results.

## Why the earlier numbers were wrong

The first committed runs (`standard_old/`, `standard_new/`) do **not** support a
12x claim:

- `standard_new` **diverged to NaN** (no gradient clipping in that run; 3,297
  skipped batches; validation frozen from epoch 1) — not a valid trained model.
- The "12x" was `training_seconds` 10585/852, but the runs ran a **different
  number of epochs** (22 vs 8, the broken run early-stopped). Not like-for-like.
- The only honest signal is **per-iteration throughput ~4.9 vs ~1.15 it/s ≈
  4.2x**, consistent across all healthy optimized runs.

This harness replaces that with a per-epoch, equal-work, repeated measurement.

## Design

- **Metric:** steady-state **training** seconds per epoch. Epoch 1 of every run
  is dropped as warmup (DataLoader spin-up, cudnn autotuning, CUDA allocator).
  Validation time is logged separately and excluded from the headline.
- **Fixed identical workload** for old and new: same data subset, batch size 16,
  query 20, lr 1e-4, max_dc 0.5, `--max-epochs 4`. The old script hardcodes
  early-stop patience 6, so 4 epochs always run fully with no early-stop
  confound; the new run sets `--patience 99` and `--grad-clip 1.0` (stays
  healthy).
- **Repeats:** independent SLURM array tasks per config capture run-to-run
  variance (node, thermal, scheduling). Per-epoch rows within a run feed a
  hierarchical bootstrap.
- **Decomposition:** the new system is re-run with one optimization disabled at a
  time (AMP, TF32, cudnn.benchmark) and the old system with plotting removed, to
  attribute the speedup. Effects are marginal/non-additive.

> Note: DataLoader workers/pin_memory/persistent_workers are already present in
> **both** codebases, so this benchmark isolates compute (AMP/TF32/cudnn) and
> removed per-epoch plotting — not data loading. Data-loading/startup speedups
> show up separately in `startup_plus_data_loading_seconds` in each
> `*_metrics.txt`.

## Instrumentation

- New (`src/sptnet/training/cli.py`) writes `epoch_timing.csv` per run with
  `epoch,train_seconds,val_seconds,n_train_batches,n_val_batches,amp,tf32,cudnn_benchmark`
  (timed with `torch.cuda.synchronize()` around each pass). Env toggles:
  `SPT_DISABLE_AMP=1`, `SPT_DISABLE_TF32=1`, `SPT_CUDNN_BENCHMARK=0`.
- Old (`../SPTnet/SPTnet_training_old_cli.py`) needs the **same** `epoch_timing.csv`
  emitter plus an `SPT_DISABLE_PLOT=1` toggle. This must be added on CSD3 (the
  old repo lives there alongside its data); see the patch instructions at the
  bottom of this file.

## Run matrix

Headline (K=10 each):

| Config | System | Toggles |
|---|---|---|
| `new_full` | new | none |
| `old` | old | none (plotting on) |

Decomposition (K=3 each):

| Config | System | Toggle |
|---|---|---|
| `new_no_amp` | new | `SPT_DISABLE_AMP=1` |
| `new_no_tf32` | new | `SPT_DISABLE_TF32=1` |
| `new_no_cudnn_bench` | new | `SPT_CUDNN_BENCHMARK=0` |
| `old_no_plot` | old | `SPT_DISABLE_PLOT=1` |

## Submitting (CSD3)

From the `opt-SPTnet` repo root, with the venv installed (`pip install -e .`).
Set `SPT_TRAIN_DATA`/`SPT_MAX_FILES` to the exact same subset for every config
(the data is split across the two repos on CSD3 — point both systems at the same
files; absolute paths are safest).

```bash
# Headline — new (10 repeats)
sbatch --array=1-10 \
  --export=ALL,SPT_SYSTEM=new,SPT_RUN_NAME=new_full,\
SPT_TRAIN_DATA="/path/to/TrainData/*.mat",SPT_MAX_FILES=100 \
  slurm/train_sptnet_benchmark_csd3.slurm

# Headline — old (10 repeats)
sbatch --array=1-10 \
  --export=ALL,SPT_SYSTEM=old,SPT_RUN_NAME=old,\
SPT_OLD_SCRIPT=/path/to/SPTnet/SPTnet_training_old_cli.py,\
SPT_TRAIN_DATA="/path/to/TrainData/*.mat",SPT_MAX_FILES=100 \
  slurm/train_sptnet_benchmark_csd3.slurm

# Decomposition (3 repeats each)
sbatch --array=1-3 --export=ALL,SPT_SYSTEM=new,SPT_RUN_NAME=new_no_amp,SPT_DISABLE_AMP=1,SPT_TRAIN_DATA="/path/to/TrainData/*.mat",SPT_MAX_FILES=100 slurm/train_sptnet_benchmark_csd3.slurm
sbatch --array=1-3 --export=ALL,SPT_SYSTEM=new,SPT_RUN_NAME=new_no_tf32,SPT_DISABLE_TF32=1,SPT_TRAIN_DATA="/path/to/TrainData/*.mat",SPT_MAX_FILES=100 slurm/train_sptnet_benchmark_csd3.slurm
sbatch --array=1-3 --export=ALL,SPT_SYSTEM=new,SPT_RUN_NAME=new_no_cudnn_bench,SPT_CUDNN_BENCHMARK=0,SPT_TRAIN_DATA="/path/to/TrainData/*.mat",SPT_MAX_FILES=100 slurm/train_sptnet_benchmark_csd3.slurm
sbatch --array=1-3 --export=ALL,SPT_SYSTEM=old,SPT_RUN_NAME=old_no_plot,SPT_DISABLE_PLOT=1,SPT_OLD_SCRIPT=/path/to/SPTnet/SPTnet_training_old_cli.py,SPT_TRAIN_DATA="/path/to/TrainData/*.mat",SPT_MAX_FILES=100 slurm/train_sptnet_benchmark_csd3.slurm
```

Each task writes to `Trained_models/benchmarks/<RUN_NAME>/run_NN/`
(`epoch_timing.csv`, `slurm_benchmark_*_metrics.txt`, log).

## Analysis

Collect the per-run `epoch_timing.csv` files under one tree
(`experiments/benchmarks/<config>/run_NN/epoch_timing.csv`), then:

```bash
python experiments/benchmarks/analyze_benchmarks.py \
  --results-root experiments/benchmarks \
  --new-config new_full --old-config old \
  --warmup-epochs 1 --bootstrap 10000
```

Outputs to `experiments/benchmarks/results/`:
- `summary.csv` — per-config mean/median per-epoch time + 95% CI,
- `summary.md` — headline speedup (old/new) with CI + decomposition table,
- `per_epoch_times.png` — per-config distribution plot.

Commit `results/` plus every per-run `epoch_timing.csv` and `*_metrics.txt` so
each report figure maps to a committed artifact.

## Old-script patch (apply on CSD3)

Mirror the new instrumentation in `../SPTnet/SPTnet_training_old_cli.py`:

1. Before the epoch `while` loop, create `epoch_timing.csv` in `args.model_dir`
   with the same 8-column header, and define
   `def _sync(): torch.cuda.synchronize() if torch.cuda.is_available()`.
2. Read `_DISABLE_PLOT = os.environ.get("SPT_DISABLE_PLOT","0") == "1"`. When set,
   replace `fig, ax = plt.subplots(...)` with a no-op axis shim (an object whose
   `__getitem__` returns self and whose `plot`/`set_title` are no-ops) and skip
   the per-epoch `plt.tight_layout()/plt.pause()/plt.savefig()`.
3. Time the train loop and the validation loop with `_sync()` + `time.time()`
   and append one row per epoch: `amp=0, tf32=0, cudnn_benchmark=0` (the old
   script's actual settings).

See the "Codex prompt" handed off with this change for the exact edits.
