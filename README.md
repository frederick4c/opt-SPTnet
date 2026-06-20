# opt-SPTnet

<!-- Replace OWNER with your GitHub user/org once the public repo is created. -->
[![CI](https://github.com/OWNER/opt-SPTnet/actions/workflows/ci.yml/badge.svg)](https://github.com/frederick4c/opt-SPTnet/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Python package for SPTnet model components, datasets, training, inference,
segmentation, stitching, data conversion, and notebook visualization utilities
used in single-particle tracking workflows.

This repository refactors the original
[Huang Lab SPTnet](https://github.com/HuanglabPurdue/SPTnet) codebase into an
installable package under `src/sptnet/`, with command-line entry points for the
main workflows.

Documentation can be found at
https://app.readthedocs.org/projects/opt-sptnet/.

## Installation

Install the package in editable mode from the repository root:

```bash
python -m pip install -e .
```

For a deterministic install pinned to the tested dependency versions, add the
committed lock file:

```bash
python -m pip install -e . -c constraints.txt
```

For documentation builds:

```bash
python -m pip install -e ".[docs]"
```

For tests:

```bash
python -m pip install -e ".[test]"
```

## Run with Docker

A CPU-only image builds the package and its dependencies into a reproducible
environment with no local Python, GPU, or MATLAB setup required:

```bash
docker build -t opt-sptnet .
```

Run any of the command-line tools in the container, mounting a host directory
for data and outputs:

```bash
docker run --rm opt-sptnet sptnet-train --help

docker run --rm -v "$PWD/data:/data" opt-sptnet \
  sptnet-inference --model-path /data/trained_model --data "/data/test/*.h5"
```

The image installs the CPU PyTorch build. For GPU training, build from an NVIDIA
CUDA base image and install the matching CUDA PyTorch wheel; see the comments in
the [`Dockerfile`](Dockerfile).

## Command-Line Tools

The package installs these entry points:

| Command | Purpose |
| --- | --- |
| `sptnet-hdf5-to-tiff` | Convert HDF5-backed movies to ImageJ-compatible TIFF stacks |
| `sptnet-mat-to-tiff` | Compatibility alias for MATLAB v7.3 `.mat` movie conversion |
| `sptnet-combine-trackmate` | Combine a TIFF movie and TrackMate XML export into HDF5 |
| `sptnet-generate-training-data` | Generate synthetic SPTnet training data in Python |
| `sptnet-compute-crlb` | Generate or validate the CRLB matrix used during training |
| `sptnet-train` | Train an SPTnet model |
| `sptnet-inference` | Run model inference on HDF5, MAT, or tiled data |
| `sptnet-segment` | Split large movies into SPTnet-sized HDF5 tiles |
| `sptnet-stitch` | Stitch per-tile inference results back into global tracks |

## Convert Movies to TIFF

HDF5 files containing `ims` or `timelapsedata` can be converted to
ImageJ-compatible TIFF stacks. Native `.h5`/`.hdf5` files and MATLAB v7.3
`.mat` files are both supported because MATLAB v7.3 files are HDF5-backed:

```bash
sptnet-hdf5-to-tiff "data/*.h5" --output-dir data/tiff_output
```

The older `sptnet-mat-to-tiff` command name is kept as a compatibility alias.

## Combine TIFF and TrackMate XML

TrackMate XML exports can be bundled with the corresponding TIFF movie into one
HDF5 file:

```bash
sptnet-combine-trackmate \
  RealData/full_realdata.tif \
  RealData/realdata_tracks.xml \
  -o RealData/full_realdata_trackmate.h5
```

The output stores the movie as `timelapsedata` in `T,Y,X` order, plus
`trackmate_tracks`, `trackmate_positions`, and `trackmate_lengths`.

## Train and Run Inference

Train a model from HDF5 training files:

```bash
sptnet-train --data "training/*.h5" --model-dir runs/example
```

Run inference with the trained model:

```bash
sptnet-inference --model-path runs/example/trained_model --data "test/*.h5"
```

Existing MATLAB v7.3 `.mat` training and test files remain supported by the same
commands. Pass a `*.mat` glob when using older data.

## Segment Large Movies and Stitch Results

Large `.h5`, `.hdf5`, `.mat`, `.tif`, and `.tiff` movies can be split into
SPTnet-sized HDF5 tiles without MATLAB:

```bash
sptnet-segment \
  "raw/*.h5" \
  --output-dir raw/tiles \
  --block-shape 30 64 64 \
  --overlap 0 0 0 \
  --dtype none
```

Tile files are named by 1-based spatial tile order, for example
`movie_x001_y001.h5` or `movie_n001_x001_y001.h5` for batched inputs. Each
spatial tile stores all temporal clips for that position as `timelapsedata` in
`N,T,Y,X` order, where `N` is the number of temporal blocks.

Run inference on the resulting tile files:

```bash
sptnet-inference \
  --model-path Trained_models/full_run/trained_model \
  --data "RealData/realdata_tiles/full_realdata_x*.h5" \
  --batch-size 8
```

After inference, stitch per-tile predictions back into global tracks and remove
duplicates from overlapping tiles:

```bash
sptnet-stitch \
  "RealData/realdata_tiles/inference_results/result_full_realdata_*.h5" \
  --output RealData/stitched_tracks.csv \
  --score-threshold 0.90 \
  --min-track-len 5 \
  --dedup-distance 3.0
```

The stitcher discards predictions in padded tile regions and merges duplicate
tracks from overlapping tiles. `--dedup-distance 3.0` is a useful default for
edge-aligned overlapping tiles.

## Generate Training Data

The MATLAB training-data generator has a deterministic Python replacement:

```bash
sptnet-generate-training-data --seed 123 --num-files 1 --videos-per-file 100
```

It writes native h5py-readable `.h5` HDF5 files by default and can also write
HDF5-backed `.mat` files for compatibility:

```bash
sptnet-generate-training-data --output-extension .mat
```

## CRLB Matrix

The CRLB matrix used by training can be regenerated without MATLAB:

```bash
sptnet-compute-crlb --output CRLB_H_D_frame.h5 --progress
```

If the matrix is missing when training starts, `sptnet-train` computes it once,
saves it to the configured CRLB path, and reuses it on later runs.

## Notebook Visualization

In a notebook, load the packaged visualization helper:

```python
from IPython.display import HTML
from sptnet.visualization import show_video

ani = show_video(
    test_data_path="test/example.h5",
    results_path="runs/example/inference_results/result_example.h5",
    threshold=0.5,
)
HTML(ani.to_jshtml())
```

## Documentation

Documentation can be found at
https://app.readthedocs.org/projects/opt-sptnet/.

The documentation is built with Sphinx from the pages in `docs/` and API
docstrings in `src/sptnet`. Read the Docs is configured via `.readthedocs.yaml`
and installs documentation requirements from `docs/requirements.txt`.

Build the documentation locally with:

```bash
python -m pip install -e ".[docs]"
sphinx-build -W -b html docs docs/_build/html
```

The generated site will be written to `docs/_build/html`.

## Tests

Run the lightweight pytest suite from the repository root:

```bash
python -m pip install -e ".[test]"
pytest
```

The tests use small synthetic HDF5/TIFF files and tensors, so they should run
comfortably on a laptop. Torch-backed tests skip automatically in minimal
environments where PyTorch is not installed.

## License and Attribution

`opt-SPTnet` is released under the MIT License (see [`LICENSE`](LICENSE)).

This project is a derivative of the original
[Huang Lab SPTnet](https://github.com/HuanglabPurdue/SPTnet) (MIT License,
© 2025 BoilermakerSR). The upstream copyright notice is retained in `LICENSE` as
required. The refactor replaces the original MATLAB-dependent script collection
with an installable, tested, documented, MATLAB-free Python package: a single
`src/sptnet` package with command-line entry points for every workflow stage,
a pytest suite, Sphinx documentation, and runtime optimisations (automatic mixed
precision, TF32, vectorised preprocessing, and removed per-epoch plotting
overhead) that give a measured steady-state training speedup over the original.
