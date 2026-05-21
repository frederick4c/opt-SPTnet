"""Data conversion utilities for SPTnet."""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import tifffile


DEFAULT_DATASET_NAMES = ("ims", "timelapsedata")
DEFAULT_OUTPUT_DIR_NAME = "tiff_output"
TARGET_TIFF_AXES = "TYX"


@dataclass(frozen=True)
class TiffConversionResult:
    """Summary for one TIFF written or skipped during HDF5 conversion."""

    input_path: Path
    output_path: Path
    dataset_name: str
    source_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    sample_index: Optional[int]
    skipped: bool = False


def expand_file_patterns(paths_or_patterns: Sequence[str]) -> List[Path]:
    """Expand file paths and glob patterns into a sorted, de-duplicated list."""
    paths = []
    for item in paths_or_patterns:
        matches = glob.glob(item) if glob.has_magic(item) else [item]
        paths.extend(Path(match) for match in matches)
    return sorted(set(paths))


def find_movie_dataset(h5_file: h5py.File, dataset_names: Sequence[str] = DEFAULT_DATASET_NAMES):
    """Return the first available movie dataset from an open HDF5/MAT file."""
    for name in dataset_names:
        if name in h5_file:
            return name, h5_file[name]
    names = ", ".join(dataset_names)
    raise KeyError(f"None of the expected datasets were found: {names}")


def _default_input_axes(ndim: int) -> str:
    if ndim == 3:
        return "TYX"
    if ndim == 4:
        return "NTYX"
    raise ValueError(f"Only 3D or 4D movie datasets are supported, got ndim={ndim}.")


def _normalize_axes(axes: Optional[str], ndim: int) -> str:
    axes = _default_input_axes(ndim) if axes is None else axes.upper()
    if len(axes) != ndim:
        raise ValueError(f"input_axes={axes!r} has length {len(axes)}, but data has {ndim} dimensions.")
    if len(set(axes)) != len(axes):
        raise ValueError(f"input_axes={axes!r} contains duplicate axis labels.")
    valid = set("NTYX")
    invalid = set(axes) - valid
    if invalid:
        raise ValueError(f"input_axes={axes!r} contains unsupported labels: {sorted(invalid)}")
    return axes


def _slice_sample(dataset, sample_axis: int, sample_index: int):
    indexer = [slice(None)] * dataset.ndim
    indexer[sample_axis] = sample_index
    return dataset[tuple(indexer)]


def _to_tyx(video: np.ndarray, source_axes: str) -> np.ndarray:
    if set(source_axes) != set(TARGET_TIFF_AXES):
        raise ValueError(
            f"Each output movie must have exactly axes {TARGET_TIFF_AXES!r}; got {source_axes!r}."
        )
    order = [source_axes.index(axis) for axis in TARGET_TIFF_AXES]
    return np.transpose(video, order)


def _coerce_video(video, source_axes: str, dtype: np.dtype) -> np.ndarray:
    arr = np.asarray(video)
    arr = _to_tyx(arr, source_axes)
    return arr.astype(dtype, copy=False)


def _write_tiff(video: np.ndarray, output_path: Path, overwrite: bool) -> bool:
    if output_path.exists() and not overwrite:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        output_path,
        video,
        imagej=True,
        metadata={"axes": TARGET_TIFF_AXES},
    )
    return True


def convert_mat_file_to_tiff(
    mat_path: os.PathLike,
    output_dir: Optional[os.PathLike] = None,
    *,
    dataset_names: Sequence[str] = DEFAULT_DATASET_NAMES,
    input_axes: Optional[str] = None,
    dtype: str = "float32",
    overwrite: bool = True,
) -> List[TiffConversionResult]:
    """Convert one HDF5 movie file into ImageJ-compatible TIFF stacks.

    By default, 3D arrays are interpreted as ``T,Y,X`` and 4D arrays as
    ``N,T,Y,X``. This supports native ``.h5``/``.hdf5`` files and MATLAB v7.3
    ``.mat`` files. For MATLAB-origin files whose axes appear in a different
    order through ``h5py``, pass ``input_axes`` explicitly, for example
    ``"NTXY"`` or ``"YXTN"``.
    """
    mat_path = Path(mat_path)
    output_dir = Path(output_dir) if output_dir is not None else mat_path.parent / DEFAULT_OUTPUT_DIR_NAME
    dtype_np = np.dtype(dtype)

    results: List[TiffConversionResult] = []
    with h5py.File(mat_path, "r") as h5_file:
        dataset_name, dataset = find_movie_dataset(h5_file, dataset_names)
        axes = _normalize_axes(input_axes, dataset.ndim)
        source_shape = tuple(int(dim) for dim in dataset.shape)

        if dataset.ndim == 3:
            video = _coerce_video(dataset[()], axes, dtype_np)
            output_path = output_dir / f"{mat_path.stem}.tif"
            wrote = _write_tiff(video, output_path, overwrite=overwrite)
            results.append(
                TiffConversionResult(
                    input_path=mat_path,
                    output_path=output_path,
                    dataset_name=dataset_name,
                    source_shape=source_shape,
                    output_shape=tuple(int(dim) for dim in video.shape),
                    sample_index=None,
                    skipped=not wrote,
                )
            )
            return results

        sample_axis = axes.index("N")
        per_sample_axes = axes.replace("N", "")
        num_samples = int(dataset.shape[sample_axis])
        for sample_index in range(num_samples):
            video = _coerce_video(
                _slice_sample(dataset, sample_axis, sample_index),
                per_sample_axes,
                dtype_np,
            )
            output_path = output_dir / f"{mat_path.stem}_{sample_index:03d}.tif"
            wrote = _write_tiff(video, output_path, overwrite=overwrite)
            results.append(
                TiffConversionResult(
                    input_path=mat_path,
                    output_path=output_path,
                    dataset_name=dataset_name,
                    source_shape=source_shape,
                    output_shape=tuple(int(dim) for dim in video.shape),
                    sample_index=sample_index,
                    skipped=not wrote,
                )
            )
    return results


def convert_hdf5_file_to_tiff(
    hdf5_path: os.PathLike,
    output_dir: Optional[os.PathLike] = None,
    *,
    dataset_names: Sequence[str] = DEFAULT_DATASET_NAMES,
    input_axes: Optional[str] = None,
    dtype: str = "float32",
    overwrite: bool = True,
) -> List[TiffConversionResult]:
    """Convert one native HDF5 or MATLAB v7.3 file into TIFF stacks.

    This is the preferred public name. ``convert_mat_file_to_tiff`` remains as
    a compatibility alias for older scripts and notebooks.
    """
    return convert_mat_file_to_tiff(
        hdf5_path,
        output_dir=output_dir,
        dataset_names=dataset_names,
        input_axes=input_axes,
        dtype=dtype,
        overwrite=overwrite,
    )


def convert_mat_files_to_tiff(
    mat_paths: Iterable[os.PathLike],
    output_dir: Optional[os.PathLike] = None,
    *,
    dataset_names: Sequence[str] = DEFAULT_DATASET_NAMES,
    input_axes: Optional[str] = None,
    dtype: str = "float32",
    overwrite: bool = True,
) -> List[TiffConversionResult]:
    """Convert multiple HDF5 movie files into TIFF stacks."""
    results: List[TiffConversionResult] = []
    for mat_path in mat_paths:
        results.extend(
            convert_mat_file_to_tiff(
                mat_path,
                output_dir=output_dir,
                dataset_names=dataset_names,
                input_axes=input_axes,
                dtype=dtype,
                overwrite=overwrite,
            )
        )
    return results


def convert_hdf5_files_to_tiff(
    hdf5_paths: Iterable[os.PathLike],
    output_dir: Optional[os.PathLike] = None,
    *,
    dataset_names: Sequence[str] = DEFAULT_DATASET_NAMES,
    input_axes: Optional[str] = None,
    dtype: str = "float32",
    overwrite: bool = True,
) -> List[TiffConversionResult]:
    """Convert multiple native HDF5 or MATLAB v7.3 files into TIFF stacks."""
    return convert_mat_files_to_tiff(
        hdf5_paths,
        output_dir=output_dir,
        dataset_names=dataset_names,
        input_axes=input_axes,
        dtype=dtype,
        overwrite=overwrite,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for HDF5-to-TIFF conversion."""
    parser = argparse.ArgumentParser(
        description="Convert SPTnet HDF5 movie datasets into ImageJ-compatible TIFF stacks."
    )
    parser.add_argument(
        "mat_files",
        nargs="+",
        help="HDF5 file paths or glob patterns (.h5/.hdf5 or MATLAB v7.3 .mat).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help=f"Directory for TIFF outputs. Defaults to ./{DEFAULT_OUTPUT_DIR_NAME} beside the first input file.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset name to try. Can be passed more than once. Defaults to ims then timelapsedata.",
    )
    parser.add_argument(
        "--input-axes",
        default=None,
        help="Axis order of the source dataset, e.g. TYX, NTYX, NTXY, or YXTN. Defaults to TYX/NTYX.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help="Output dtype passed to NumPy, default: float32.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip existing TIFF files instead of overwriting them.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only print errors.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for HDF5-to-TIFF conversion commands."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    mat_paths = expand_file_patterns(args.mat_files)
    if not mat_paths:
        parser.error("No HDF5 files matched the provided paths or patterns.")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = mat_paths[0].parent / DEFAULT_OUTPUT_DIR_NAME

    results = convert_mat_files_to_tiff(
        mat_paths,
        output_dir=output_dir,
        dataset_names=tuple(args.datasets) if args.datasets else DEFAULT_DATASET_NAMES,
        input_axes=args.input_axes,
        dtype=args.dtype,
        overwrite=not args.no_overwrite,
    )

    if not args.quiet:
        for result in results:
            action = "Skipped" if result.skipped else "Saved"
            sample = "" if result.sample_index is None else f" sample {result.sample_index}"
            print(
                f"{action}{sample}: {result.output_path} "
                f"(dataset={result.dataset_name}, shape={result.output_shape})"
            )
        print(f"Converted {sum(not result.skipped for result in results)} TIFF file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
