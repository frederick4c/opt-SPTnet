# SLURM job scripts (cluster-specific, not part of the package)

These are the batch scripts used to run training, inference, data generation, and
the runtime benchmark on the Cambridge CSD3 cluster. They are **one-off
operational scripts**, not part of the installable `sptnet` package: they hard
code cluster paths, partitions, module loads, and account names, and they invoke
the package's command-line entry points (`sptnet-train`, `sptnet-inference`,
`sptnet-generate-training-data`, ...).

Treat them as reference/provenance for how the reported runs were produced. To
reuse them elsewhere, adjust the `#SBATCH` directives, module loads, and absolute
data paths for your environment.

| Script | Purpose |
| --- | --- |
| `train_sptnet_csd3.slurm` | Standard model training. |
| `train_sptnet_benchmark_csd3.slurm` | Runtime benchmark harness (old/new system switch, per-run dirs, `SPT_*` toggles). |
| `inference_sptnet_csd3.slurm` | Model inference. |
| `inference_diff_eval_csd3.slurm` / `generate_diff_eval_csd3.slurm` | Diffusion-condition evaluation data + inference. |
| `generate_trainingdata_csd3.slurm` | Synthetic training-data generation. |
| `finetune_perf_ablation_csd3.slurm` / `train_scratch_ablation_csd3.slurm` | Diffusion loss-objective ablations (secondary chapter). |
| `train_sparse_track_diffusion_teacher_csd3.slurm` | Side experiment (diffusion teacher); not part of the main narrative. |

The benchmark protocol and analysis live in `experiments/benchmarks/`.
