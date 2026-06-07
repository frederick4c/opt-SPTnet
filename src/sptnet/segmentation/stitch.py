"""Stitch segmented SPTnet inputs and inference outputs.

The splitter writes one HDF5 file per spatial tile, with all temporal clips for
that tile stacked in ``timelapsedata``. After model inference, this module maps
per-tile ``result_*.h5``/``.mat`` files back into full-movie coordinates,
filters predictions that landed in padded tile regions, and merges duplicate
tracks from overlapping tiles.

Current SPTnet inference outputs store normalized coordinates in ``Y,X`` order,
so the public stitching helpers default to ``xy_order="yx"`` and convert the
arrays to plotting/global ``x,y`` internally.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import scipy.io as sio

from sptnet.segmentation.io import read_hdf5_attrs, read_named_array, write_hdf5_array


SPATIAL_TILE_WITH_SAMPLE_RE = re.compile(
    r"^(?P<source>.+)_n(?P<sample>\d+)_x(?P<xidx>\d+)_y(?P<yidx>\d+)$"
)
SPATIAL_TILE_RE = re.compile(
    r"^(?P<source>.+)_x(?P<xidx>\d+)_y(?P<yidx>\d+)$"
)
TEMPORAL_TILE_WITH_SAMPLE_RE = re.compile(
    r"^(?P<source>.+)_n(?P<sample>\d+)_t(?P<tidx>\d+)_x(?P<xidx>\d+)_y(?P<yidx>\d+)$"
)
TEMPORAL_TILE_RE = re.compile(
    r"^(?P<source>.+)_t(?P<tidx>\d+)_x(?P<xidx>\d+)_y(?P<yidx>\d+)$"
)
START_TILE_RE = re.compile(
    r"^(?P<source>.+)__n(?P<sample>\d+)__t(?P<t>\d+)_y(?P<y>\d+)_x(?P<x>\d+)$"
)
LEGACY_TILE_RE = re.compile(r"^(?:result)?block(?P<sample>\d+)_x(?P<yidx>\d+)_y(?P<xidx>\d+)_t(?P<tidx>\d+)$")


@dataclass(frozen=True)
class TileLocation:
    """Zero-based location of one tile clip in the source movie."""

    sample_index: int
    t_start: int
    y_start: int
    x_start: int
    source_stem: str = ""


@dataclass(frozen=True)
class Track:
    """One stitched query track in global movie coordinates.

    ``points`` is a float array with columns ``frame,y,x,score``. ``h`` and
    ``diffusion`` are the model's per-query constants for the track, and
    ``track_id`` is assigned after deduplication/stitching.
    """

    points: np.ndarray  # columns: frame, y, x, score
    h: float
    diffusion: float
    query_index: int
    sample_index: int
    tile_path: Path
    track_id: int = -1

    @property
    def length(self) -> int:
        return int(self.points.shape[0])

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.points[:, 3])) if self.length else 0.0


def expand_file_patterns(paths_or_patterns: Sequence[str]) -> List[Path]:
    paths = []
    for item in paths_or_patterns:
        matches = glob.glob(item) if glob.has_magic(item) else [item]
        paths.extend(Path(match) for match in matches)
    return sorted(set(paths))


def _attr_ints(attrs: dict[str, object], name: str) -> list[int]:
    value = attrs[name]
    arr = np.asarray(value)
    if arr.ndim == 0:
        return [int(arr.item())]
    return [int(item) for item in arr.tolist()]


def _candidate_source_file_for_result(path: os.PathLike) -> Path | None:
    """Find the sibling tile file for a result when ``source_file`` is stale."""
    path = Path(path)
    stem = path.stem
    if not stem.startswith("result_"):
        return None
    source_name = f"{stem[len('result_') :]}{path.suffix}"
    candidates = [
        path.parent / source_name,
        path.parent.parent / source_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_related_path(anchor: os.PathLike, target: os.PathLike) -> Path:
    """Resolve a relative stored path against likely project/result roots."""
    target_path = Path(str(target))
    if target_path.is_absolute():
        return target_path

    anchor_path = Path(anchor)
    roots = [Path.cwd(), anchor_path.parent]
    try:
        roots.extend(anchor_path.resolve().parents)
    except OSError:
        roots.extend(anchor_path.absolute().parents)

    seen: set[Path] = set()
    unique_roots: list[Path] = []
    for root in roots:
        resolved_root = root.resolve(strict=False)
        if resolved_root not in seen:
            seen.add(resolved_root)
            unique_roots.append(resolved_root)

    for root in unique_roots:
        candidate = (root / target_path).resolve(strict=False)
        if candidate.exists():
            return candidate

    first_part = target_path.parts[0] if target_path.parts else ""
    for root in unique_roots:
        candidate = (root / target_path).resolve(strict=False)
        if first_part and (root / first_part).exists():
            return candidate

    return target_path


def _candidate_manifest_paths(path: os.PathLike, source_file: os.PathLike | None = None) -> list[Path]:
    """Return likely segmentation manifests for a result or source tile path."""
    path = Path(path)
    roots = [path.parent, path.parent.parent]
    if source_file is not None:
        source_path = Path(source_file)
        roots.extend([source_path.parent, source_path.parent.parent])

    manifests: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        manifests.extend(sorted(root.glob("*__segmentation_manifest.csv")))
    return sorted(set(manifests))


def _manifest_rows_for_tile(path: os.PathLike, source_file: os.PathLike | None = None) -> list[dict[str, str]]:
    """Find manifest rows for the tile corresponding to ``path``."""
    path = Path(path)
    stem = path.stem
    if stem.startswith("result_"):
        stem = stem[len("result_") :]
    tile_name = f"{stem}{path.suffix}"

    for manifest in _candidate_manifest_paths(path, source_file=source_file):
        with manifest.open(newline="") as handle:
            rows = [
                row
                for row in csv.DictReader(handle)
                if Path(row["output_path"]).name == tile_name
            ]
        if rows:
            rows.sort(key=lambda row: int(row["t_index"]))
            return rows
    return []


def _locations_from_manifest_rows(rows: list[dict[str, str]], count: int | None = None) -> list[TileLocation]:
    if count is not None and len(rows) != count:
        raise ValueError(f"Manifest has {len(rows)} rows for tile but result contains {count} clip(s).")
    return [
        TileLocation(
            sample_index=int(row["sample_index"]) + index,
            t_start=int(row["t_start"]),
            y_start=int(row["y_start"]),
            x_start=int(row["x_start"]),
            source_stem=Path(row["source_path"]).stem,
        )
        for index, row in enumerate(rows)
    ]


def parse_tile_locations(
    path: os.PathLike,
    count: int | None = None,
    stride: Sequence[int] | None = None,
) -> List[TileLocation]:
    """Parse one global location per temporal clip in a tile or result file.

    Metadata attrs written by :func:`sptnet.segmentation.split.split_movie_file`
    are preferred. If the path is an inference result, the result's
    ``source_file`` attr is followed to the source tile. If that attr is stale,
    the function also checks for a matching tile next to the ``inference_results``
    directory. Filename parsing is used only as a fallback and requires
    ``stride`` for order-based names such as ``movie_x002_y001.h5``.
    """
    path = Path(path)
    attrs = read_hdf5_attrs(path)
    if {"sample_index", "y_start", "x_start"} <= set(attrs) and (
        "t_starts" in attrs or "t_start" in attrs
    ):
        t_starts = _attr_ints(attrs, "t_starts") if "t_starts" in attrs else _attr_ints(attrs, "t_start")
        if count is not None and len(t_starts) != count:
            if len(t_starts) == 1:
                t_starts = t_starts * count
            else:
                raise ValueError(f"{path} has {len(t_starts)} t_starts but result contains {count} clip(s).")
        return [
            TileLocation(
                sample_index=int(attrs["sample_index"]) + index,
                t_start=t_start,
                y_start=int(attrs["y_start"]),
                x_start=int(attrs["x_start"]),
                source_stem=str(attrs.get("source_stem", "")),
            )
            for index, t_start in enumerate(t_starts)
        ]

    if "source_file" in attrs:
        source_file = _resolve_related_path(path, attrs["source_file"])
        if not source_file.exists():
            source_file = _candidate_source_file_for_result(path) or source_file
        if source_file.exists():
            try:
                return parse_tile_locations(source_file, count=count, stride=stride)
            except ValueError:
                pass
        manifest_rows = _manifest_rows_for_tile(path, source_file=source_file)
        if manifest_rows:
            return _locations_from_manifest_rows(manifest_rows, count=count)

    sibling_source = _candidate_source_file_for_result(path)
    if sibling_source is not None:
        try:
            return parse_tile_locations(sibling_source, count=count, stride=stride)
        except ValueError:
            pass

    manifest_rows = _manifest_rows_for_tile(path)
    if manifest_rows:
        return _locations_from_manifest_rows(manifest_rows, count=count)

    return [parse_tile_location(path, stride=stride)]


def parse_tile_location(path: os.PathLike, stride: Sequence[int] | None = None) -> TileLocation:
    """Parse one tile location from attrs or a supported filename.

    This is the scalar version of :func:`parse_tile_locations`. Prefer
    ``parse_tile_locations`` for modern spatial tile files because one spatial
    tile can contain many temporal clips.
    """
    path = Path(path)
    attrs = read_hdf5_attrs(path)
    if {"sample_index", "t_start", "y_start", "x_start"} <= set(attrs):
        return TileLocation(
            sample_index=int(attrs["sample_index"]),
            t_start=int(attrs["t_start"]),
            y_start=int(attrs["y_start"]),
            x_start=int(attrs["x_start"]),
            source_stem=str(attrs.get("source_stem", "")),
        )
    if "source_file" in attrs:
        source_file = _resolve_related_path(path, attrs["source_file"])
        if not source_file.exists():
            source_file = _candidate_source_file_for_result(path) or source_file
        if source_file.exists():
            try:
                return parse_tile_location(source_file, stride=stride)
            except ValueError:
                pass
        manifest_rows = _manifest_rows_for_tile(path, source_file=source_file)
        if manifest_rows:
            return _locations_from_manifest_rows(manifest_rows)[0]

    sibling_source = _candidate_source_file_for_result(path)
    if sibling_source is not None:
        try:
            return parse_tile_location(sibling_source, stride=stride)
        except ValueError:
            pass

    manifest_rows = _manifest_rows_for_tile(path)
    if manifest_rows:
        return _locations_from_manifest_rows(manifest_rows)[0]

    stem = path.stem
    if stem.startswith("result_"):
        stem = stem[len("result_") :]
    match = TEMPORAL_TILE_WITH_SAMPLE_RE.match(stem) or TEMPORAL_TILE_RE.match(stem)
    if match:
        if stride is None:
            raise ValueError(f"Order-based tile name {path.name!r} requires --stride T Y X unless attrs are available.")
        stride_t, stride_y, stride_x = (int(v) for v in stride)
        sample = match.groupdict().get("sample")
        return TileLocation(
            sample_index=(int(sample) - 1) if sample is not None else 0,
            t_start=(int(match.group("tidx")) - 1) * stride_t,
            y_start=(int(match.group("yidx")) - 1) * stride_y,
            x_start=(int(match.group("xidx")) - 1) * stride_x,
            source_stem=match.group("source"),
        )

    match = SPATIAL_TILE_WITH_SAMPLE_RE.match(stem) or SPATIAL_TILE_RE.match(stem)
    if match:
        if stride is None:
            raise ValueError(f"Order-based tile name {path.name!r} requires --stride T Y X unless attrs are available.")
        _, stride_y, stride_x = (int(v) for v in stride)
        sample = match.groupdict().get("sample")
        return TileLocation(
            sample_index=(int(sample) - 1) if sample is not None else 0,
            t_start=0,
            y_start=(int(match.group("yidx")) - 1) * stride_y,
            x_start=(int(match.group("xidx")) - 1) * stride_x,
            source_stem=match.group("source"),
        )

    match = START_TILE_RE.match(stem)
    if match:
        return TileLocation(
            sample_index=int(match.group("sample")),
            t_start=int(match.group("t")),
            y_start=int(match.group("y")),
            x_start=int(match.group("x")),
            source_stem=match.group("source"),
        )

    legacy = LEGACY_TILE_RE.match(stem)
    if legacy:
        if stride is None:
            raise ValueError(f"Legacy tile name {path.name!r} requires --stride T Y X.")
        stride_t, stride_y, stride_x = (int(v) for v in stride)
        return TileLocation(
            sample_index=int(legacy.group("sample")) - 1,
            t_start=(int(legacy.group("tidx")) - 1) * stride_t,
            y_start=(int(legacy.group("yidx")) - 1) * stride_y,
            x_start=(int(legacy.group("xidx")) - 1) * stride_x,
            source_stem="block",
        )
    raise ValueError(f"Could not parse tile coordinates from {path}.")


def stitch_movie_tiles(
    tile_paths: Iterable[os.PathLike],
    output_path: os.PathLike,
    *,
    dataset_names: Sequence[str] = ("timelapsedata", "ims"),
    source_shape: Sequence[int] | None = None,
    stride: Sequence[int] | None = None,
    overwrite: bool = True,
) -> np.ndarray:
    """Average overlapping raw movie tiles back into one ``T,Y,X`` movie.

    Parameters
    ----------
    tile_paths:
        Tile files produced by the segmentation splitter.
    output_path:
        HDF5 file written with a ``timelapsedata`` dataset containing the
        stitched movie.
    dataset_names:
        Dataset names to try in each tile file.
    source_shape:
        Optional crop shape in ``T,Y,X`` order. If omitted, the output covers
        the full extent implied by the tiles.
    stride:
        Required only for legacy/order-based filenames that lack metadata attrs.
    overwrite:
        Whether to replace an existing output file.
    """
    tile_paths = [Path(path) for path in tile_paths]
    if not tile_paths:
        raise ValueError("No tile paths were provided.")

    records = []
    padded_shape = np.zeros(3, dtype=int)
    for path in tile_paths:
        tile_data, _ = read_named_array(path, dataset_names)
        if tile_data.ndim == 3:
            tile_stack = tile_data[np.newaxis, ...]
        elif tile_data.ndim == 4:
            tile_stack = tile_data
        else:
            raise ValueError(f"{path} must contain T,Y,X or N,T,Y,X tiles, got {tile_data.shape}.")
        locations = parse_tile_locations(path, count=tile_stack.shape[0], stride=stride)
        for tile, loc in zip(tile_stack, locations):
            start = np.array([loc.t_start, loc.y_start, loc.x_start], dtype=int)
            padded_shape = np.maximum(padded_shape, start + np.array(tile.shape, dtype=int))
            records.append((path, tile.astype(np.float32, copy=False), loc))

    if source_shape is None:
        source_shape = tuple(int(v) for v in padded_shape)
    else:
        source_shape = tuple(int(v) for v in source_shape)
    summed = np.zeros(tuple(int(v) for v in padded_shape), dtype=np.float32)
    counts = np.zeros_like(summed)
    for _, tile, loc in records:
        ts = slice(loc.t_start, loc.t_start + tile.shape[0])
        ys = slice(loc.y_start, loc.y_start + tile.shape[1])
        xs = slice(loc.x_start, loc.x_start + tile.shape[2])
        summed[ts, ys, xs] += tile
        counts[ts, ys, xs] += 1

    mask = counts > 0
    stitched = np.zeros_like(summed)
    stitched[mask] = summed[mask] / counts[mask]
    cropped = stitched[: source_shape[0], : source_shape[1], : source_shape[2]]
    write_hdf5_array(
        output_path,
        "timelapsedata",
        cropped,
        {"format": "sptnet-stitched-movie", "tile_count": len(records)},
        overwrite=overwrite,
    )
    return cropped


def load_inference_arrays(
    path: os.PathLike,
    *,
    tile_shape_yx: Sequence[int] = (64, 64),
    xy_order: str = "yx",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one inference result as normalized arrays in tile-pixel coordinates.

    Returns
    -------
    obj:
        Detection probabilities with shape ``N,T,Q``.
    xy:
        Pixel coordinates with shape ``N,T,Q,2`` in ``x,y`` order after applying
        ``xy_order`` and scaling normalized ``[-1,1]`` outputs into tile pixels.
    h, diffusion:
        Per-query constants with shape ``N,Q``. Diffusion is scaled by ``0.5``
        to match the legacy visualization/stitching convention.
    """
    data = _read_inference_data(path)
    return _format_inference_arrays(data, tile_shape_yx=tile_shape_yx, xy_order=xy_order)


def _read_inference_data(path: os.PathLike) -> dict[str, np.ndarray]:
    path = Path(path)
    try:
        with h5py.File(path, "r") as handle:
            return {name: np.asarray(handle[name]) for name in handle.keys()}
    except OSError:
        return {key: value for key, value in sio.loadmat(path).items() if not key.startswith("__")}


def _format_inference_arrays(
    data: dict[str, np.ndarray],
    *,
    tile_shape_yx: Sequence[int] = (64, 64),
    xy_order: str = "xy",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize inference arrays to obj ``N,T,Q`` and xy ``N,T,Q,2``."""
    if xy_order not in {"xy", "yx"}:
        raise ValueError(f"xy_order must be 'xy' or 'yx', got {xy_order!r}.")

    obj = np.asarray(data["obj_estimation"])
    xy = np.asarray(data["estimation_xy"])
    h = np.squeeze(np.asarray(data["estimation_H"]))
    diffusion = np.squeeze(np.asarray(data["estimation_C"])) * 0.5

    if obj.ndim == 4:
        obj = np.squeeze(np.transpose(obj, (0, 3, 2, 1)), axis=-1)
    elif obj.ndim != 3:
        raise ValueError(f"Unexpected obj_estimation shape {obj.shape}.")

    if xy.ndim != 4:
        raise ValueError(f"Unexpected estimation_xy shape {xy.shape}.")
    if xy_order == "yx":
        xy = xy[..., [1, 0]]
    tile_y, tile_x = (int(v) for v in tile_shape_yx)
    xy_scale = np.asarray([tile_x / 2.0, tile_y / 2.0], dtype=np.float32)
    xy = np.transpose(xy * xy_scale + xy_scale, (0, 2, 1, 3))

    if h.ndim == 0:
        h = h.reshape(1, 1)
        diffusion = diffusion.reshape(1, 1)
    elif h.ndim == 1:
        if obj.shape[0] > 1 and obj.shape[2] == 1 and h.shape[0] == obj.shape[0]:
            h = h[:, np.newaxis]
            diffusion = diffusion[:, np.newaxis]
        else:
            h = h[np.newaxis, :]
            diffusion = diffusion[np.newaxis, :]
    return obj, xy, h, diffusion


def _result_source_shape_tyx(path: os.PathLike) -> tuple[int, int, int] | None:
    """Return original source movie shape for a result/tile file when available."""
    attrs = read_hdf5_attrs(path)
    if "source_shape_tyx" in attrs:
        return tuple(int(value) for value in np.asarray(attrs["source_shape_tyx"]).tolist())

    source_file = attrs.get("source_file")
    if source_file is None:
        source_path = _candidate_source_file_for_result(path)
        if source_path is None:
            return None
    else:
        source_path = _resolve_related_path(path, source_file)
        if not source_path.exists():
            source_path = _candidate_source_file_for_result(path) or source_path
    if not source_path.exists():
        manifest_rows = _manifest_rows_for_tile(path, source_file=source_path)
        if not manifest_rows:
            return None
        row = manifest_rows[0]
        return (int(row["source_t"]), int(row["source_y"]), int(row["source_x"]))
    source_attrs = read_hdf5_attrs(source_path)
    if "source_shape_tyx" not in source_attrs:
        manifest_rows = _manifest_rows_for_tile(path, source_file=source_path)
        if not manifest_rows:
            return None
        row = manifest_rows[0]
        return (int(row["source_t"]), int(row["source_y"]), int(row["source_x"]))
    return tuple(int(value) for value in np.asarray(source_attrs["source_shape_tyx"]).tolist())


def _valid_tile_shape_for_location(
    loc: TileLocation,
    source_shape_tyx: tuple[int, int, int] | None,
    tile_shape_yx: Sequence[int],
) -> tuple[int, int]:
    """Real, unpadded ``Y,X`` extent for one tile location."""
    tile_y, tile_x = (int(value) for value in tile_shape_yx)
    if source_shape_tyx is None:
        return tile_y, tile_x

    _, source_y, source_x = source_shape_tyx
    valid_y = max(0, min(tile_y, int(source_y) - int(loc.y_start)))
    valid_x = max(0, min(tile_x, int(source_x) - int(loc.x_start)))
    return valid_y, valid_x


def tracks_from_result_file(
    result_path: os.PathLike,
    *,
    score_threshold: float = 0.9,
    min_track_len: int = 5,
    max_step: float | None = None,
    stride: Sequence[int] | None = None,
    tile_shape_yx: Sequence[int] = (64, 64),
    xy_order: str = "yx",
) -> List[Track]:
    """Convert one tile result file into global-coordinate tracks.

    The function follows tile metadata to add ``t_start``, ``y_start``, and
    ``x_start`` to per-tile predictions. Predictions falling in padded tile
    regions are discarded using ``source_shape_tyx`` metadata when available.

    Parameters
    ----------
    result_path:
        One ``result_*.h5`` or ``result_*.mat`` file.
    score_threshold:
        Minimum object probability for a frame to be included.
    min_track_len:
        Minimum number of kept frames for a query to become a track.
    max_step:
        Optional maximum allowed frame-to-frame displacement in pixels.
    stride:
        Required only for metadata-free legacy/order-based filenames.
    tile_shape_yx:
        Tile size used to scale normalized model coordinates.
    xy_order:
        Coordinate order in ``estimation_xy``. Current SPTnet outputs are
        ``"yx"``.
    """
    result_path = Path(result_path)
    obj, xy, h, diffusion = load_inference_arrays(result_path, tile_shape_yx=tile_shape_yx, xy_order=xy_order)
    locations = parse_tile_locations(result_path, count=obj.shape[0], stride=stride)
    source_shape_tyx = _result_source_shape_tyx(result_path)
    tracks: list[Track] = []
    for sample_index in range(obj.shape[0]):
        loc = locations[sample_index]
        valid_y, valid_x = _valid_tile_shape_for_location(loc, source_shape_tyx, tile_shape_yx)
        if valid_y <= 0 or valid_x <= 0:
            continue
        for query_index in range(obj.shape[2]):
            xy_query = xy[sample_index, :, query_index, :]
            in_real_tile = (
                (xy_query[:, 0] >= 0)
                & (xy_query[:, 0] < valid_x)
                & (xy_query[:, 1] >= 0)
                & (xy_query[:, 1] < valid_y)
            )
            keep = (obj[sample_index, :, query_index] >= score_threshold) & in_real_tile
            if int(np.sum(keep)) < min_track_len:
                continue
            frames = np.nonzero(keep)[0] + loc.t_start
            xy_local = xy_query[keep, :]
            points = np.column_stack(
                [
                    frames,
                    xy_local[:, 1] + loc.y_start,
                    xy_local[:, 0] + loc.x_start,
                    obj[sample_index, keep, query_index],
                ]
            ).astype(np.float32)
            if max_step is not None and points.shape[0] > 1:
                step = np.sqrt(np.sum(np.diff(points[:, 1:3], axis=0) ** 2, axis=1))
                if np.any(step > max_step):
                    continue
            tracks.append(
                Track(
                    points=points,
                    h=float(h[sample_index, query_index]),
                    diffusion=float(diffusion[sample_index, query_index]),
                    query_index=query_index,
                    sample_index=loc.sample_index,
                    tile_path=result_path,
                )
            )
    return tracks


def _is_duplicate(
    track: Track,
    kept: Track,
    min_overlap: int,
    distance_threshold: float,
    *,
    close_fraction: float = 0.5,
) -> bool:
    frames_a = track.points[:, 0].astype(np.int64)
    frames_b = kept.points[:, 0].astype(np.int64)
    common, idx_a, idx_b = np.intersect1d(frames_a, frames_b, return_indices=True)
    if common.size < min_overlap:
        return False
    dist = np.sqrt(np.sum((track.points[idx_a, 1:3] - kept.points[idx_b, 1:3]) ** 2, axis=1))
    return (
        float(np.median(dist)) <= distance_threshold
        and float(np.mean(dist <= distance_threshold)) >= close_fraction
    )


def _merge_duplicate_tracks(primary: Track, duplicate: Track) -> Track:
    """Merge duplicate tracks, keeping the highest-confidence point per frame."""
    by_frame: dict[int, np.ndarray] = {}
    for points in (primary.points, duplicate.points):
        for point in points:
            frame = int(point[0])
            existing = by_frame.get(frame)
            if existing is None or point[3] > existing[3]:
                by_frame[frame] = point.copy()

    merged_points = np.asarray([by_frame[frame] for frame in sorted(by_frame)], dtype=np.float32)
    primary_weight = max(primary.length, 1)
    duplicate_weight = max(duplicate.length, 1)
    total_weight = primary_weight + duplicate_weight

    return replace(
        primary,
        points=merged_points,
        h=(primary.h * primary_weight + duplicate.h * duplicate_weight) / total_weight,
        diffusion=(
            primary.diffusion * primary_weight + duplicate.diffusion * duplicate_weight
        ) / total_weight,
    )


def _track_index_keys(
    track: Track,
    *,
    frame_bin_size: int,
    spatial_bin_size: float,
) -> set[tuple[int, int, int]]:
    """Coarse index keys used to avoid all-pairs duplicate checks."""
    if track.length == 0:
        return set()
    frames = track.points[:, 0]
    y = track.points[:, 1]
    x = track.points[:, 2]
    frame_bins = range(
        int(np.floor(np.min(frames) / frame_bin_size)),
        int(np.floor(np.max(frames) / frame_bin_size)) + 1,
    )
    y_bin = int(np.floor(float(np.median(y)) / spatial_bin_size))
    x_bin = int(np.floor(float(np.median(x)) / spatial_bin_size))
    return {
        (frame_bin, y_bin + dy, x_bin + dx)
        for frame_bin in frame_bins
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    }


def deduplicate_tracks(
    tracks: Sequence[Track],
    *,
    min_overlap: int = 5,
    distance_threshold: float = 3.0,
) -> List[Track]:
    """Merge repeated tracks using overlapping frames and spatial distance.

    Higher mean confidence wins the representative identity, but duplicate
    tracks are merged frame-by-frame so useful points from a lower-ranked
    overlapping tile are not discarded.
    """
    ranked = sorted(tracks, key=lambda tr: (tr.mean_score, tr.length), reverse=True)
    kept: list[Track] = []
    index: dict[tuple[int, int, int], set[int]] = {}
    frame_bin_size = 30
    spatial_bin_size = max(float(distance_threshold) * 4.0, 8.0)

    def add_to_index(track_index: int) -> None:
        for key in _track_index_keys(
            kept[track_index],
            frame_bin_size=frame_bin_size,
            spatial_bin_size=spatial_bin_size,
        ):
            index.setdefault(key, set()).add(track_index)

    for track in ranked:
        candidate_indices: set[int] = set()
        for key in _track_index_keys(
            track,
            frame_bin_size=frame_bin_size,
            spatial_bin_size=spatial_bin_size,
        ):
            candidate_indices.update(index.get(key, set()))
        duplicate_index = next(
            (
                index
                for index in candidate_indices
                for existing in (kept[index],)
                if _is_duplicate(track, existing, min_overlap, distance_threshold)
            ),
            None,
        )
        if duplicate_index is not None:
            kept[duplicate_index] = _merge_duplicate_tracks(kept[duplicate_index], track)
            add_to_index(duplicate_index)
        else:
            kept.append(track)
            add_to_index(len(kept) - 1)
    return [replace(track, track_id=index) for index, track in enumerate(kept)]


def stitch_inference_results(
    result_paths: Iterable[os.PathLike],
    *,
    score_threshold: float = 0.9,
    min_track_len: int = 5,
    max_step: float | None = None,
    deduplicate: bool = True,
    dedup_overlap: int = 5,
    dedup_distance: float = 3.0,
    stride: Sequence[int] | None = None,
    tile_shape_yx: Sequence[int] = (64, 64),
    xy_order: str = "yx",
) -> List[Track]:
    """Stitch many per-tile inference result files into global tracks.

    Parameters
    ----------
    result_paths:
        Iterable of result files, or expanded paths from ``result_*.h5`` globs.
    score_threshold, min_track_len, max_step:
        Per-file filtering options forwarded to :func:`tracks_from_result_file`.
    deduplicate:
        If true, merge duplicate tracks created by overlapping spatial tiles.
    dedup_overlap:
        Minimum number of common frames before two tracks can be duplicates.
    dedup_distance:
        Median spatial distance threshold in pixels for duplicate merging.
        ``3.0`` is a practical default for heavily overlapping edge-aligned
        tiles; use smaller values for stricter behavior.
    stride:
        Required only for metadata-free legacy/order-based filenames.
    tile_shape_yx:
        Tile size used for coordinate scaling.
    xy_order:
        Coordinate order in ``estimation_xy``. Defaults to ``"yx"`` for current
        SPTnet inference outputs.

    Returns
    -------
    list[Track]
        Tracks with global ``frame,y,x,score`` points and assigned ``track_id``.
    """
    tracks: list[Track] = []
    for path in result_paths:
        tracks.extend(
            tracks_from_result_file(
                path,
                score_threshold=score_threshold,
                min_track_len=min_track_len,
                max_step=max_step,
                stride=stride,
                tile_shape_yx=tile_shape_yx,
                xy_order=xy_order,
            )
        )
    if deduplicate:
        return deduplicate_tracks(tracks, min_overlap=dedup_overlap, distance_threshold=dedup_distance)
    return [replace(track, track_id=index) for index, track in enumerate(tracks)]


def write_tracks_csv(tracks: Sequence[Track], output_path: os.PathLike) -> None:
    """Write stitched tracks as one CSV row per detected point.

    The CSV columns are ``track_id,frame,y,x,score,h,diffusion,query_index,
    sample_index,tile_path``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["track_id", "frame", "y", "x", "score", "h", "diffusion", "query_index", "sample_index", "tile_path"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for track in tracks:
            for frame, y, x, score in track.points:
                writer.writerow(
                    {
                        "track_id": track.track_id,
                        "frame": int(frame),
                        "y": float(y),
                        "x": float(x),
                        "score": float(score),
                        "h": track.h,
                        "diffusion": track.diffusion,
                        "query_index": track.query_index,
                        "sample_index": track.sample_index,
                        "tile_path": track.tile_path,
                    }
                )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stitch per-tile SPTnet inference results.")
    parser.add_argument("results", nargs="+", help="Result .h5/.mat files or glob patterns.")
    parser.add_argument("-o", "--output", default="stitched_tracks.csv", help="Output CSV path.")
    parser.add_argument("--score-threshold", type=float, default=0.9)
    parser.add_argument("--min-track-len", type=int, default=5)
    parser.add_argument("--max-step", type=float, default=None, help="Reject tracks with any step larger than this many pixels.")
    parser.add_argument("--no-deduplicate", action="store_true")
    parser.add_argument("--dedup-overlap", type=int, default=5)
    parser.add_argument("--dedup-distance", type=float, default=3.0)
    parser.add_argument("--stride", nargs=3, type=int, metavar=("T", "Y", "X"), help="Required for legacy block###_x#_y#_t# names.")
    parser.add_argument("--tile-shape", nargs=2, type=int, default=(64, 64), metavar=("Y", "X"), help="Tile size used to scale normalized xy predictions.")
    parser.add_argument("--xy-order", choices=("xy", "yx"), default="yx", help="Coordinate order in estimation_xy. Current SPTnet inference outputs are yx.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    paths = expand_file_patterns(args.results)
    if not paths:
        parser.error("No result files matched the provided paths or patterns.")
    tracks = stitch_inference_results(
        paths,
        score_threshold=args.score_threshold,
        min_track_len=args.min_track_len,
        max_step=args.max_step,
        deduplicate=not args.no_deduplicate,
        dedup_overlap=args.dedup_overlap,
        dedup_distance=args.dedup_distance,
        stride=args.stride,
        tile_shape_yx=args.tile_shape,
        xy_order=args.xy_order,
    )
    write_tracks_csv(tracks, args.output)
    print(f"Wrote {len(tracks)} stitched track(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
