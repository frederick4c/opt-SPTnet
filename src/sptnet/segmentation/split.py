"""Split large SPTnet movies into inference-sized tiles."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tifffile

from sptnet.segmentation.io import DEFAULT_DATASET_NAMES, read_named_array, write_hdf5_array


DEFAULT_BLOCK_SHAPE = (30, 64, 64)  # T, Y, X
DEFAULT_OUTPUT_DIR_NAME = "segmentation_tiles"
TIFF_EXTENSIONS = {".tif", ".tiff"}


@dataclass(frozen=True)
class TileMetadata:
    """Metadata for a split movie tile."""

    source_path: Path
    output_path: Path
    dataset_name: str
    sample_index: int
    tile_index: int
    t_index: int
    y_index: int
    x_index: int
    t_start: int
    y_start: int
    x_start: int
    block_shape: Tuple[int, int, int]
    stride: Tuple[int, int, int]
    source_shape: Tuple[int, int, int]
    padded_shape: Tuple[int, int, int]
    skipped: bool = False


@dataclass(frozen=True)
class SegmentationResult:
    """Summary for one segmented input movie."""

    input_path: Path
    output_dir: Path
    dataset_name: str
    original_shape: Tuple[int, ...]
    input_axes: str
    tiles: Tuple[TileMetadata, ...]
    manifest_path: Path
    settings_path: Path


def expand_file_patterns(paths_or_patterns: Sequence[str]) -> List[Path]:
    """Expand file paths and glob patterns into a sorted, de-duplicated list."""
    paths = []
    for item in paths_or_patterns:
        matches = glob.glob(item) if glob.has_magic(item) else [item]
        paths.extend(Path(match) for match in matches)
    return sorted(set(paths))


def _default_axes(ndim: int) -> str:
    if ndim == 3:
        return "TYX"
    if ndim == 4:
        return "NTYX"
    raise ValueError(f"Only 3D or 4D movies are supported, got ndim={ndim}.")


def normalize_axes(axes: Optional[str], ndim: int) -> str:
    """Validate source axes.

    Supported labels are ``N`` sample, ``T`` time, ``Y`` rows/height, and
    ``X`` columns/width. MATLAB-origin files stored as ``H,W,T,N`` should use
    ``YXTN``.
    """
    axes = _default_axes(ndim) if axes is None else axes.upper()
    if len(axes) != ndim:
        raise ValueError(f"input_axes={axes!r} has length {len(axes)}, but data has {ndim} dimensions.")
    if len(set(axes)) != len(axes):
        raise ValueError(f"input_axes={axes!r} contains duplicate labels.")
    required = set("TYX") if ndim == 3 else set("NTYX")
    if set(axes) != required:
        raise ValueError(f"input_axes={axes!r} must contain exactly {sorted(required)}.")
    return axes


def to_ntyx(array: np.ndarray, axes: Optional[str] = None) -> tuple[np.ndarray, str]:
    """Return movie data as ``N,T,Y,X`` plus the validated source axes."""
    axes = normalize_axes(axes, array.ndim)
    target = "NTYX"
    if array.ndim == 3:
        order = [axes.index(axis) for axis in "TYX"]
        return np.transpose(array, order)[np.newaxis, ...], axes
    order = [axes.index(axis) for axis in target]
    return np.transpose(array, order), axes


def read_movie_array(
    input_path: os.PathLike,
    dataset_names: Sequence[str] = DEFAULT_DATASET_NAMES,
) -> tuple[np.ndarray, str, Tuple[int, ...]]:
    """Read TIFF/HDF5/MATLAB movie data plus dataset name and original shape."""
    input_path = Path(input_path)
    if input_path.suffix.lower() in TIFF_EXTENSIONS:
        raw = np.asarray(tifffile.imread(input_path))
        original_shape = tuple(int(dim) for dim in raw.shape)
        if raw.ndim == 2:
            raw = raw[np.newaxis, ...]
        if raw.ndim != 3:
            raise ValueError(f"{input_path} must be a 2D image or 3D T,Y,X TIFF stack, got shape {original_shape}.")
        return raw, "timelapsedata", original_shape

    raw, dataset_name = read_named_array(input_path, dataset_names)
    return raw, dataset_name, tuple(int(dim) for dim in raw.shape)


def _as_shape3(value: Sequence[int] | int, name: str, *, allow_zero: bool = False) -> Tuple[int, int, int]:
    if isinstance(value, int):
        value = (value, value, value)
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values in T,Y,X order.")
    out = tuple(int(v) for v in value)
    min_value = 0 if allow_zero else 1
    if any(v < min_value for v in out):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} values must be {qualifier}, got {out}.")
    return out


def _overlap_to_stride(block_shape: Tuple[int, int, int], overlap: Sequence[int] | int) -> Tuple[int, int, int]:
    overlap = _as_shape3(overlap, "overlap", allow_zero=True)
    stride = tuple(block - ov for block, ov in zip(block_shape, overlap))
    if any(value <= 0 for value in stride):
        raise ValueError(f"Overlap must be smaller than block shape; block={block_shape}, overlap={overlap}.")
    return stride


def _starts(length: int, block: int, stride: int) -> List[int]:
    count = max(1, math.ceil((max(length, block) - block) / stride) + 1)
    return [index * stride for index in range(count)]


def _pad_video(video: np.ndarray, padded_shape: Tuple[int, int, int], padding: str) -> np.ndarray:
    pad_width = tuple((0, padded - current) for current, padded in zip(video.shape, padded_shape))
    if padding == "zero":
        return np.pad(video, pad_width, mode="constant", constant_values=0)
    if padding == "edge":
        return np.pad(video, pad_width, mode="edge")
    raise ValueError("padding must be 'zero' or 'edge'.")


def tile_filename(
    source_stem: str,
    sample_index: int,
    x_index: int,
    y_index: int,
    ext: str,
    *,
    include_sample: bool = False,
) -> str:
    """Build a concise 1-based spatial tile filename."""
    ext = ext if ext.startswith(".") else f".{ext}"
    tile_part = f"x{x_index + 1:03d}_y{y_index + 1:03d}"
    if include_sample:
        tile_part = f"n{sample_index + 1:03d}_{tile_part}"
    return f"{source_stem}_{tile_part}{ext}"


def _attrs(
    meta: TileMetadata,
    input_axes: str,
    original_shape: Tuple[int, ...],
    *,
    t_indices: Sequence[int],
    t_starts: Sequence[int],
) -> dict[str, object]:
    attrs = {
        "format": "sptnet-segmentation-tile",
        "source_file": str(meta.source_path),
        "source_stem": meta.source_path.stem,
        "dataset_name": meta.dataset_name,
        "sample_index": meta.sample_index,
        "tile_index": meta.tile_index,
        "t_index": meta.t_index,
        "y_index": meta.y_index,
        "x_index": meta.x_index,
        "t_start": meta.t_start,
        "y_start": meta.y_start,
        "x_start": meta.x_start,
        "t_indices": tuple(int(value) for value in t_indices),
        "t_starts": tuple(int(value) for value in t_starts),
        "block_shape_tyx": meta.block_shape,
        "stride_tyx": meta.stride,
        "source_shape_tyx": meta.source_shape,
        "padded_shape_tyx": meta.padded_shape,
        "input_axes": input_axes,
        "original_shape": original_shape,
    }
    return attrs


def split_movie_file(
    input_path: os.PathLike,
    output_dir: Optional[os.PathLike] = None,
    *,
    dataset_names: Sequence[str] = DEFAULT_DATASET_NAMES,
    input_axes: Optional[str] = None,
    block_shape: Sequence[int] = DEFAULT_BLOCK_SHAPE,
    overlap: Sequence[int] = (0, 0, 0),
    padding: str = "zero",
    output_ext: str = ".h5",
    output_dataset: str = "timelapsedata",
    overwrite: bool = True,
    dtype: str | None = "float32",
) -> SegmentationResult:
    """Split one movie into ``T,Y,X`` HDF5 tiles for SPTnet inference."""
    input_path = Path(input_path)
    output_dir = Path(output_dir) if output_dir is not None else input_path.parent / DEFAULT_OUTPUT_DIR_NAME
    block_shape = _as_shape3(block_shape, "block_shape")
    overlap = _as_shape3(overlap, "overlap", allow_zero=True)
    stride = _overlap_to_stride(block_shape, overlap)

    raw, dataset_name, original_shape = read_movie_array(input_path, dataset_names)
    movies, axes = to_ntyx(raw, input_axes)
    if isinstance(dtype, str) and dtype.lower() == "none":
        dtype = None
    if dtype is not None:
        movies = movies.astype(np.dtype(dtype), copy=False)

    sample_count, source_t, source_y, source_x = movies.shape
    t_starts = _starts(source_t, block_shape[0], stride[0])
    y_starts = _starts(source_y, block_shape[1], stride[1])
    x_starts = _starts(source_x, block_shape[2], stride[2])
    padded_shape = (
        t_starts[-1] + block_shape[0],
        y_starts[-1] + block_shape[1],
        x_starts[-1] + block_shape[2],
    )
    source_shape = (source_t, source_y, source_x)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiles: list[TileMetadata] = []
    tile_index = 0
    for sample_index in range(sample_count):
        padded = _pad_video(movies[sample_index], padded_shape, padding)
        for y_index, y_start in enumerate(y_starts):
            for x_index, x_start in enumerate(x_starts):
                output_path = output_dir / tile_filename(
                    input_path.stem,
                    sample_index,
                    x_index,
                    y_index,
                    output_ext,
                    include_sample=sample_count > 1,
                )
                patches = []
                file_metas = []
                for t_index, t_start in enumerate(t_starts):
                    patch = padded[
                        t_start : t_start + block_shape[0],
                        y_start : y_start + block_shape[1],
                        x_start : x_start + block_shape[2],
                    ]
                    patches.append(patch)
                    file_metas.append(
                        TileMetadata(
                            source_path=input_path,
                            output_path=output_path,
                            dataset_name=dataset_name,
                            sample_index=sample_index,
                            tile_index=tile_index,
                            t_index=t_index,
                            y_index=y_index,
                            x_index=x_index,
                            t_start=t_start,
                            y_start=y_start,
                            x_start=x_start,
                            block_shape=block_shape,
                            stride=stride,
                            source_shape=source_shape,
                            padded_shape=padded_shape,
                        )
                    )
                    tile_index += 1
                tile_stack = np.stack(patches, axis=0)
                wrote = write_hdf5_array(
                    output_path,
                    output_dataset,
                    tile_stack,
                    _attrs(
                        file_metas[0],
                        axes,
                        original_shape,
                        t_indices=[meta.t_index for meta in file_metas],
                        t_starts=[meta.t_start for meta in file_metas],
                    ),
                    overwrite=overwrite,
                )
                if not wrote:
                    file_metas = [TileMetadata(**{**asdict(meta), "skipped": True}) for meta in file_metas]
                tiles.extend(file_metas)

    manifest_path = output_dir / f"{input_path.stem}__segmentation_manifest.csv"
    settings_path = output_dir / f"{input_path.stem}__segmentation_settings.json"
    result = SegmentationResult(
        input_path=input_path,
        output_dir=output_dir,
        dataset_name=dataset_name,
        original_shape=original_shape,
        input_axes=axes,
        tiles=tuple(tiles),
        manifest_path=manifest_path,
        settings_path=settings_path,
    )
    _write_manifest(result)
    _write_settings(result, block_shape, overlap, padding, output_dataset)
    return result


def split_movie_files(
    input_paths: Iterable[os.PathLike],
    output_dir: Optional[os.PathLike] = None,
    **kwargs,
) -> List[SegmentationResult]:
    """Split multiple movie files."""
    return [split_movie_file(path, output_dir=output_dir, **kwargs) for path in input_paths]


def _write_manifest(result: SegmentationResult) -> None:
    fields = [
        "output_path",
        "source_path",
        "dataset_name",
        "sample_index",
        "tile_index",
        "t_index",
        "y_index",
        "x_index",
        "t_start",
        "y_start",
        "x_start",
        "block_t",
        "block_y",
        "block_x",
        "stride_t",
        "stride_y",
        "stride_x",
        "source_t",
        "source_y",
        "source_x",
        "padded_t",
        "padded_y",
        "padded_x",
        "skipped",
    ]
    with result.manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for tile in result.tiles:
            writer.writerow(
                {
                    "output_path": tile.output_path,
                    "source_path": tile.source_path,
                    "dataset_name": tile.dataset_name,
                    "sample_index": tile.sample_index,
                    "tile_index": tile.tile_index,
                    "t_index": tile.t_index,
                    "y_index": tile.y_index,
                    "x_index": tile.x_index,
                    "t_start": tile.t_start,
                    "y_start": tile.y_start,
                    "x_start": tile.x_start,
                    "block_t": tile.block_shape[0],
                    "block_y": tile.block_shape[1],
                    "block_x": tile.block_shape[2],
                    "stride_t": tile.stride[0],
                    "stride_y": tile.stride[1],
                    "stride_x": tile.stride[2],
                    "source_t": tile.source_shape[0],
                    "source_y": tile.source_shape[1],
                    "source_x": tile.source_shape[2],
                    "padded_t": tile.padded_shape[0],
                    "padded_y": tile.padded_shape[1],
                    "padded_x": tile.padded_shape[2],
                    "skipped": tile.skipped,
                }
            )


def _write_settings(
    result: SegmentationResult,
    block_shape: Tuple[int, int, int],
    overlap: Sequence[int],
    padding: str,
    output_dataset: str,
) -> None:
    settings = {
        "format": "sptnet-segmentation-settings",
        "input_path": str(result.input_path),
        "dataset_name": result.dataset_name,
        "input_axes": result.input_axes,
        "original_shape": result.original_shape,
        "block_shape_tyx": block_shape,
        "overlap_tyx": tuple(int(v) for v in overlap),
        "stride_tyx": result.tiles[0].stride if result.tiles else None,
        "padding": padding,
        "output_dataset": output_dataset,
        "tile_count": len({str(tile.output_path) for tile in result.tiles}),
        "clip_count": len(result.tiles),
        "manifest_path": str(result.manifest_path),
    }
    with result.settings_path.open("w") as handle:
        json.dump(settings, handle, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split SPTnet movies into named HDF5 tiles.")
    parser.add_argument("inputs", nargs="+", help="Input .h5/.hdf5/.mat/.tif/.tiff files or glob patterns.")
    parser.add_argument("-o", "--output-dir", default=None, help=f"Output directory. Defaults to {DEFAULT_OUTPUT_DIR_NAME}.")
    parser.add_argument("--dataset", action="append", dest="datasets", help="Dataset name to try, repeatable.")
    parser.add_argument("--input-axes", default=None, help="Source axes, e.g. TYX, NTYX, YXT, or YXTN.")
    parser.add_argument("--block-shape", nargs=3, type=int, default=DEFAULT_BLOCK_SHAPE, metavar=("T", "Y", "X"))
    parser.add_argument("--overlap", nargs=3, type=int, default=(0, 0, 0), metavar=("T", "Y", "X"))
    parser.add_argument("--padding", choices=("zero", "edge"), default="zero")
    parser.add_argument("--output-ext", default=".h5", help="Tile extension, default .h5. Use .mat for HDF5-backed .mat files.")
    parser.add_argument("--dtype", default="float32", help="Tile dtype, or 'none' to preserve input dtype.")
    parser.add_argument("--no-overwrite", action="store_true", help="Skip existing tiles.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    paths = expand_file_patterns(args.inputs)
    if not paths:
        parser.error("No input files matched the provided paths or patterns.")

    dtype = None if args.dtype.lower() == "none" else args.dtype
    results = split_movie_files(
        paths,
        output_dir=args.output_dir,
        dataset_names=tuple(args.datasets) if args.datasets else DEFAULT_DATASET_NAMES,
        input_axes=args.input_axes,
        block_shape=tuple(args.block_shape),
        overlap=tuple(args.overlap),
        padding=args.padding,
        output_ext=args.output_ext,
        overwrite=not args.no_overwrite,
        dtype=dtype,
    )

    for result in results:
        output_paths = {tile.output_path for tile in result.tiles}
        skipped_paths = {tile.output_path for tile in result.tiles if tile.skipped}
        wrote = len(output_paths - skipped_paths)
        print(f"{result.input_path}: wrote {wrote}/{len(output_paths)} spatial tile file(s) to {result.output_dir}")
        print(f"  manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
