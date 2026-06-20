# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026

First packaged release. opt-SPTnet refactors the original MATLAB-dependent
[Huang Lab SPTnet](https://github.com/HuanglabPurdue/SPTnet) script collection
into an installable, tested, documented, MATLAB-free Python package.

### Added
- Single `src/sptnet` package with command-line entry points for every workflow
  stage: data conversion, TrackMate combination, training-data generation, CRLB
  computation, training, inference, segmentation, stitching, and preprocessing.
- Deterministic, MATLAB-free Python replacements for the training-data generator
  and CRLB matrix computation.
- Per-epoch wall-time instrumentation (`epoch_timing.csv`) and `SPT_*`
  optimisation toggles (AMP, TF32, cudnn autotuning, autocast dtype, gradient
  scaler) used by the runtime benchmark.
- pytest suite covering datasets, losses, models, segmentation, stitching,
  conversion, CRLB, the CLIs, the benchmark instrumentation, and visualization,
  plus a meta-test enforcing API-documentation coverage.
- Sphinx documentation (published on Read the Docs).
- Packaging and reproducibility surface: `LICENSE`, `Dockerfile`,
  `constraints.txt`, GitHub Actions CI, and project-health files.

### Changed
- Removed the MATLAB / DIPimage dependency throughout, making the workflows
  license-free and runnable by independent researchers.
- Vectorised per-sample input normalisation and removed per-epoch plotting
  overhead, contributing to a substantial measured per-epoch training speedup
  over the original implementation.
