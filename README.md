# SPTnet

**WIP**
Refactor of SPTnet (https://github.com/HuanglabPurdue/SPTnet) to optimise the code and make the package accessible 


The refactored package lives under `src/sptnet/`. 

## Convert HDF5 Movies to TIFF

Install the package in editable mode first:

```bash
python -m pip install -e .
```

HDF5 files containing `ims` or `timelapsedata` can be converted to
ImageJ-compatible TIFF stacks. This includes both native h5py-readable
`.h5`/`.hdf5` files and MATLAB v7.3 `.mat` files because they are HDF5-backed:

```bash
sptnet-hdf5-to-tiff "data/*.h5" --output-dir data/tiff_output
```

The older `sptnet-mat-to-tiff` command name is kept as a compatibility alias.

## Combine TIFF and TrackMate XML

TrackMate XML exports can be bundled with the corresponding TIFF movie into one
HDF5 file:

```bash
sptnet-combine-trackmate RealData/full_realdata.tif RealData/realdata_tracks.xml -o RealData/full_realdata_trackmate.h5
```

The output stores the movie as `timelapsedata` in `T,Y,X` order, plus
`trackmate_tracks`, `trackmate_positions`, and `trackmate_lengths`.

## Training and Inference

The package entry points replace the old root-level scripts:

```bash
sptnet-train --data "training/*.h5" --model-dir runs/example
sptnet-inference --model-path runs/example/trained_model --data "test/*.h5"
```

Existing MATLAB v7.3 `.mat` training/test files remain supported by the same
commands; pass a `*.mat` glob instead when using older data.

## Segment Large Movies and Stitch Results

Large `.h5`, `.hdf5`, `.mat`, and `.tif`/`.tiff` movies can be split into
SPTnet-sized HDF5 tiles without MATLAB:

```bash
sptnet-segment "raw/*.h5" --output-dir raw/tiles --block-shape 30 64 64 --overlap 0 0 0 --dtype none
```

Tile files are named by 1-based spatial tile order, for example
`movie_x001_y001.h5` or `movie_n001_x001_y001.h5` for batched inputs. Each
spatial tile stores all temporal clips for that position as `timelapsedata` in
`N,T,Y,X` order, where `N` is the number of temporal blocks. The temporal starts
are saved in file metadata for stitching. Tile starts are edge-aligned by
default, so the last spatial and temporal tiles snap back onto the real movie
instead of becoming mostly padding. Use `--no-align-edges` only when you need
old stride-only tiling. Python-native inputs use `TYX` or `NTYX` axes by
default; legacy MATLAB arrays saved as `H,W,T,N` can be split with
`--input-axes YXTN`. TIFF inputs are written as unlabeled HDF5 tile files
containing `timelapsedata` only.

For a real movie that is not a clean multiple of 64 pixels, edge alignment is
usually preferable to padding. For example, an `89x80` movie split into `64x64`
tiles uses `x` starts `0,16` and `y` starts `0,25`, producing four overlapping
tiles with full image context instead of thin padded edge strips. Disable this
only for reproducing old stride-only splits:

```bash
sptnet-segment RealData/full_realdata.h5 \
  --output-dir RealData/realdata_tiles \
  --block-shape 30 64 64 \
  --overlap 0 0 0 \
  --dtype none
```

Run inference on the resulting tile files:

```bash
sptnet-inference \
  --model-path Trained_models/full_run/trained_model \
  --data "RealData/realdata_tiles/full_realdata_x*.h5" \
  --batch-size 8
```

On CSD3, the provided SLURM script can be pointed at the tiles:

```bash
SPT_INFER_DATA="./RealData/realdata_tiles/full_realdata_x*.h5" \
sbatch slurm/inference_sptnet_csd3.slurm
```

After running inference on the tiles, stitch per-tile predictions back into
global tracks and remove repeated tracks from overlapping tiles:

```bash
sptnet-stitch \
  "RealData/realdata_tiles/inference_results/result_full_realdata_*.h5" \
  --output RealData/stitched_tracks.csv \
  --score-threshold 0.90 \
  --min-track-len 5 \
  --dedup-distance 3.0
```

The stitcher discards predictions in padded tile regions and merges duplicate
tracks from overlapping tiles. `--dedup-distance 3.0` is a good default for
edge-aligned overlapping tiles; lower values are stricter and higher values are
more aggressive. The stitcher also accepts legacy names like
`resultblock001_x2_y3_t4.mat` when you provide `--stride T Y X`. Current SPTnet
inference outputs store coordinates as `Y,X`; use `--xy-order xy` only for older
files that used `X,Y`.

The generated Sphinx docs include a longer walkthrough in
`docs/segmentation.rst`.

The CRLB matrix used by training can be regenerated without MATLAB:

```bash
sptnet-compute-crlb --output CRLB_H_D_frame.h5 --progress
```

If the matrix is missing when training starts, `sptnet-train` will compute it once,
save it to the configured CRLB path, and reuse it on later runs. HDF5 training
data defaults to `CRLB_H_D_frame.h5`; MATLAB `.mat` training data keeps the
legacy `CRLB_H_D_frame.mat` default. Existing matrices are checked against the
current training frame count, diffusion range, and sampled reference values
before reuse.

## Generate Training Data

The MATLAB training-data generator has a deterministic Python replacement:

```bash
sptnet-generate-training-data --seed 123 --num-files 1 --videos-per-file 100
```

It writes native h5py-readable `.h5` HDF5 files by default and can also write
HDF5-backed `.mat` files for compatibility via `--output-extension .mat`. See
the Sphinx page `docs/training_data_generation.rst` for supported environment
variables, reproducibility notes, and references for the PSF/Zernike/Perlin
model components ported from the original MATLAB code.

## Visualize Inference Results

In a notebook, load the packaged visualization helper:

```python
from sptnet.visualization import show_video
from IPython.display import HTML

ani = show_video(
    test_data_path="test/example.h5",
    results_path="runs/example/inference_results/result_example.h5",
    threshold=0.5,
)
HTML(ani.to_jshtml())
```

## Documentation

API documentation is built with Sphinx from the docstrings in `src/sptnet`.
Read the Docs is configured via `.readthedocs.yaml` and installs the lightweight
documentation requirements from `docs/requirements.txt`.

```bash
python -m pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
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
