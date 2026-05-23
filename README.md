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

## Training and Inference

The package entry points replace the old root-level scripts:

```bash
sptnet-train --data "training/*.h5" --model-dir runs/example
sptnet-inference --model-path runs/example/trained_model --data "test/*.h5"
```

Existing MATLAB v7.3 `.mat` training/test files remain supported by the same
commands; pass a `*.mat` glob instead when using older data.

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
