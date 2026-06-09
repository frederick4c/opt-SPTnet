# Core SPTnet Package Changes vs `../SPTnet`

This optimized repository refactors the original script-based SPTnet code in
`../SPTnet` into an installable Python package under `src/sptnet/`. This list
only covers changes to reusable package functionality and optimization paths;
experiment scripts, one-off SLURM variants, and small experiment-enabling
tweaks are intentionally omitted.

## Package Structure and Public API

- Converted the original root-level scripts and `SPTnet_toolbox.py` monolith
  into a package with focused modules:
  - `sptnet.models`: backbone, SPTnet model, and transformer blocks.
  - `sptnet.data`: HDF5/MATLAB datasets, inference datasets, TIFF conversion,
    TrackMate conversion, and DataLoader helpers.
  - `sptnet.training`: CLI training loop, reusable loss, CRLB generation, input
    normalization, and Python training-data generation.
  - `sptnet.inference`: checkpoint loading, metadata inference, batched
    prediction, CLI, and result serialization.
  - `sptnet.segmentation`: Python movie tiling and stitching.
  - `sptnet.visualization`: notebook-oriented result visualization.
- Added `pyproject.toml` so the project can be installed with `pip install -e .`.
- Added command-line entry points for core workflows:
  `sptnet-train`, `sptnet-inference`, `sptnet-generate-training-data`,
  `sptnet-compute-crlb`, `sptnet-hdf5-to-tiff`, `sptnet-mat-to-tiff`,
  `sptnet-combine-trackmate`, `sptnet-segment`, and `sptnet-stitch`.
- Preserved important legacy public names where practical:
  `SPTnet`, `BackBone`, `ResidualBlock`, `Transformer`, `Transformer3d`, and
  `TransformerMatDataset` are exported from `sptnet`.

## Model and Transformer Code

- Moved the original nested model classes out of `SPTnet_toolbox.py` into
  `src/sptnet/models/backbone.py` and `src/sptnet/models/sptnet.py`.
- The SPTnet architecture is intended to stay compatible with the original
  checkpoint layout: same backbone block structure, same query embedding, same
  spatial/temporal decoder combination, and same prediction heads.
- Device handling in model forward pass no longer hard-codes CUDA. Masks and
  positional encodings are allocated on the input feature device, so CPU and
  non-default CUDA devices work.
- Added `return_objectness_logits` to `SPTnet`. Default behavior still returns
  sigmoid objectness probabilities; the training CLI can request raw logits for
  `BCEWithLogitsLoss` style objectness training.
- Replaced the duplicated original `transformer.py` and `transformer3d.py`
  implementations with one dimension-generic `FlattenedTransformer` in
  `src/sptnet/models/transformers.py`.
- Kept compatibility classes `Transformer` and `Transformer3d`, but both now
  share the same flatten/encode/decode/reshape implementation and validate
  input, mask, position, and query shapes.

## Training Data Loading

- Replaced `SPTnet_toolbox.Transformer_mat2python` with
  `sptnet.data.TransformerMatDataset`.
- The dataset supports both labeled training files and unlabeled/movie-only
  files containing `timelapsedata`.
- HDF5-backed `.h5`, `.hdf5`, and MATLAB v7.3 `.mat` files are handled through
  the same dataset path.
- Added explicit HDF5 file lifecycle methods (`close`, context manager support,
  destructor cleanup) to reduce leaked file handles.
- Training datasets now read file metadata at construction and open HDF5/MAT
  handles lazily while fetching samples, so large runs with many individual
  files do not keep every file open at once.
- Added validation that `num_queries` is large enough for the label slots before
  padding labels to query count.
- Preserved the original class/position label semantics, including:
  - `traceposition` labels are padded to `num_queries`.
  - inactive positions remain `NaN`.
  - out-of-field-of-view positions have their class labels set to background.
- Added reusable DataLoader helpers with conservative worker defaults:
  `SPT_NUM_WORKERS` can override worker count, otherwise workers are capped to
  avoid HDF5 handle pressure on dense training jobs.
- Tiny datasets are handled more safely: split lengths are validated, empty
  train splits are adjusted when possible, and `drop_last` is disabled when a
  split is smaller than the batch size.

## Training Loop and Loss

- Moved the inline Hungarian-matching loss from the old training script into
  `sptnet.training.hungarian_matched_loss`.
- The packaged loss preserves the original matching structure:
  objectness BCE, coordinate distance, CRLB-weighted Hurst error, CRLB-weighted
  diffusion error, and background penalty for unmatched queries.
- Added numerical guards to the loss:
  - clamps/normalizes NaN and infinite predictions before BCE and matching;
  - validates `num_queries > number_of_particles`;
  - clamps CRLB weights away from zero;
  - replaces non-finite matching/final cost matrix entries before Hungarian
    assignment.
- Added independent matching and final-loss weights for Hurst and diffusion.
  This lets matching behavior differ from the scalar loss contribution.
- Added scalar-loss modes:
  - Hurst: normalized absolute error or log-space SmoothL1.
  - Diffusion: normalized absolute error, relative error, or log-space SmoothL1.
- Added objectness loss mode selection:
  - `bce` keeps legacy sigmoid probability outputs.
  - `bce_logits` trains raw objectness logits.
- Vectorized per-sample training input normalization in
  `normalize_training_inputs`, replacing the old per-sample Python loop.
- Training CLI no longer uses Tk file dialogs. Training data and output paths
  are supplied by command-line arguments, which makes headless cluster runs
  reproducible.
- Training only reads the first training file to infer `(T, H, W)` instead of
  opening every file just to discover dimensions.
- Training concatenates datasets in one flat `ConcatDataset`, avoiding the
  nested `ConcatDataset` recursion problem from iterative concatenation.
- Enabled GPU performance features in the packaged training path:
  - TF32 matmul/cuDNN where CUDA supports it.
  - `torch.amp.autocast` for forward passes.
  - `GradScaler` for mixed-precision training.
  - `optimizer.zero_grad(set_to_none=True)`.
  - cuDNN benchmark mode for fixed-size inputs.
- Loss computation is cast back to float32 after autocast because BCE is unsafe
  under autocast.
- Added gradient clipping and non-finite gradient handling.
- Added explicit finite-loss checks that raise early when training loss becomes
  invalid.
- Added training controls useful for long runs and debugging:
  `--val-every`, `--patience`, `--max-epochs`, `--max-train-batches`,
  `--max-val-batches`, and `--grad-clip`.
- Added checkpoint resume support for model weights, optimizer state, and CSV
  loss history. Optimizer resume can be disabled with `--no-resume-optimizer`.
- Added CSV loss logging in addition to the legacy text log. Learning-curve PNGs
  are written periodically rather than on every epoch to reduce headless-node
  overhead.

## CRLB Matrix Handling

- Ported CRLB matrix generation from MATLAB into
  `src/sptnet/training/crlb.py`.
- Training can now load, validate, or generate the CRLB matrix automatically.
  Missing or invalid CRLB files are regenerated once and saved.
- Native HDF5 training defaults to `CRLB_H_D_frame.h5`; legacy all-MATLAB
  training data keeps the `.mat` default.
- CRLB validation checks shape, finite values, sentinel frame-zero behavior, and
  selected recomputed grid values so incompatible matrices fail early.

## Python Training-Data Generation

- Added `sptnet.training.data_generator`, a Python replacement for the MATLAB
  training-data generator.
- The generator preserves the core simulation model:
  fractional Brownian motion trajectories, Wyant Zernike pupil PSF rendering,
  OTF rescaling, Perlin background, Poisson noise, optional motion blur, and
  HDF5/MATLAB-style labels.
- Added dataclass parameter objects (`SimulationParams`, `PSFParams`) plus
  environment-variable and CLI configuration.
- Generation can be deterministic via `--seed`/`SPT_SEED`; multiple files use
  offset seeds so repeated array jobs are reproducible but distinct.
- Output is native h5py-readable `.h5` by default, with `.mat` naming available
  for compatibility. Label cell arrays are written through HDF5 references so
  the same dataset loader can consume generated files.

## Inference

- Moved checkpoint and prediction helpers into `sptnet.inference.predict`.
- Added checkpoint metadata inference:
  - `get_num_queries` reads query count from `query_embed.weight`.
  - `get_num_frames` reads expected frame count from `conv_temp.weight`.
- Added state-dict normalization for `module.` prefix mismatches between
  DataParallel and non-DataParallel checkpoints.
- Added `load_checkpoint_strict_enough`, which still tolerates wrapper-prefix
  differences but fails if no tensors, or too few tensors, load successfully.
- Inference normalization is now per sample in a batch, instead of one global
  min/max for the whole input.
- Added `FileSampleDataset` for inference over `.h5`, `.hdf5`, MATLAB v7.3
  `.mat`, and `.tif/.tiff` movies.
- Inference records are grouped by `(T, H, W)` so batching only combines videos
  with compatible shapes.
- The inference CLI validates that input frame count matches the checkpoint
  frame count before running the model.
- For 4D HDF5/MAT files, inference can select one clip by index; segmentation
  tile files are detected by metadata and all tile clips are included.
- Result writing moved to `sptnet.inference.results_io`:
  - `.mat` inputs produce `.mat` results.
  - native HDF5/TIFF inputs produce `.h5` results.
  - writes are atomic via a temporary file followed by `os.replace`.
  - HDF5 results include format/source metadata.

## Data Conversion and TrackMate Import

- Replaced one-off MAT-to-TIFF conversion with package utilities that support
  native HDF5 and MATLAB v7.3 files.
- Movie dataset discovery tries `ims` and `timelapsedata`; callers can override
  dataset names and input axes.
- TIFF conversion writes ImageJ-compatible `T,Y,X` stacks and supports 3D
  single movies or 4D batches.
- Added TrackMate XML + TIFF import. `sptnet-combine-trackmate` stores:
  `timelapsedata`, a flat `trackmate_tracks` table, dense
  `trackmate_positions`, track lengths, and source/unit metadata.

## Segmentation and Stitching

- Added Python segmentation tiling as a package feature, replacing MATLAB
  preprocessing scripts for large movies.
- The splitter reads `.h5`, `.hdf5`, `.mat`, `.tif`, and `.tiff`, normalizes
  axes to `N,T,Y,X`, and writes fixed-size HDF5 tiles with metadata.
- Edge handling can align final tiles back onto the source movie, reducing
  mostly padded edge tiles.
- Each tile records source path, original/source shapes, tile starts, block
  shape, stride, axis metadata, and temporal starts for later stitching.
- Added stitching utilities that read per-tile inference results, recover global
  coordinates from tile metadata or manifests, discard predictions in padded
  regions, and merge duplicate tracks from overlapping tiles.

## Visualization

- Notebook visualization was moved into `sptnet.visualization`.
- Helpers can load HDF5, MATLAB, TIFF, and packaged inference result formats.
- Visualization includes alignment helpers for MATLAB/HDF5 axis differences and
  optional prediction coordinate calibration for result inspection.

## Tests and Documentation

- Added pytest coverage for model shapes, data loading, training loss, CRLB,
  inference/result IO, conversion, segmentation, TrackMate import, and
  visualization helpers.
- Added Sphinx documentation under `docs/` with API coverage checks.
