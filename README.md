# SPTnet

**WIP**
Refactor of SPTnet (https://github.com/HuanglabPurdue/SPTnet) to optimise the code and make the package accessible 


The refactored package lives under `src/sptnet/`. 

## Convert MAT Movies to TIFF

Install the package in editable mode first:

```bash
python -m pip install -e .
```

MAT/HDF5 files containing `ims` or `timelapsedata` can be converted to
ImageJ-compatible TIFF stacks:

```bash
sptnet-mat-to-tiff "data/*.mat" --output-dir data/tiff_output
```

## Training and Inference

The package entry points replace the old root-level scripts:

```bash
sptnet-train --data "training/*.mat" --model-dir runs/example
sptnet-inference --model-path runs/example/trained_model --data "test/*.mat"
```

The CRLB matrix used by training can be regenerated without MATLAB:

```bash
sptnet-compute-crlb --output CRLB_H_D_frame.mat --progress
```

If the matrix is missing when training starts, `sptnet-train` will compute it once,
save it to the configured CRLB path, and reuse it on later runs. Existing matrices
are checked against the current training frame count, diffusion range, and sampled
reference values before reuse.

## Visualize Inference Results

In a notebook, load the packaged visualization helper:

```python
from sptnet.visualization import show_video
from IPython.display import HTML

ani = show_video(
    test_data_path="test/example.mat",
    results_path="runs/example/inference_results/result_example.mat",
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
