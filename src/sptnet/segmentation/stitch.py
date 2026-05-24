"""Stitch segmented SPTnet inputs and inference outputs."""

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
    """Location of one tile in the source movie using zero-based coordinates."""

    sample_index: int
    t_start: int
    y_start: int
    x_start: int
    source_stem: str = ""


@dataclass(frozen=True)
class Track:
    """One stitched query track in global movie coordinates."""

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


def parse_tile_locations(
    path: os.PathLike,
    count: int | None = None,
    stride: Sequence[int] | None = None,
) -> List[TileLocation]:
    """Parse one location per clip stored in a tile/result file."""
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
        source_file = Path(str(attrs["source_file"]))
        if source_file.exists():
            try:
                return parse_tile_locations(source_file, count=count, stride=stride)
            except ValueError:
                pass

    return [parse_tile_location(path, stride=stride)]


def parse_tile_location(path: os.PathLike, stride: Sequence[int] | None = None) -> TileLocation:
    """Parse tile coordinates from attrs or the standard tile filename."""
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
        source_file = Path(str(attrs["source_file"]))
        if source_file.exists():
            try:
                return parse_tile_location(source_file, stride=stride)
            except ValueError:
                pass

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
    """Average overlapping raw movie tiles back into one ``T,Y,X`` movie."""
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load inference arrays as obj ``N,T,Q`` and xy ``N,T,Q,2`` in tile pixels."""
    path = Path(path)
    try:
        with h5py.File(path, "r") as handle:
            data = {name: np.asarray(handle[name]) for name in handle.keys()}
    except OSError:
        data = {key: value for key, value in sio.loadmat(path).items() if not key.startswith("__")}

    obj = np.asarray(data["obj_estimation"])
    xy = np.asarray(data["estimation_xy"])
    h = np.squeeze(np.asarray(data["estimation_H"]))
    diffusion = np.squeeze(np.asarray(data["estimation_C"])) * 0.5

    if obj.ndim == 4:
        obj = np.squeeze(np.transpose(obj, (0, 3, 2, 1)), axis=-1)
    elif obj.ndim != 3:
        raise ValueError(f"Unexpected obj_estimation shape {obj.shape} in {path}.")

    if xy.ndim != 4:
        raise ValueError(f"Unexpected estimation_xy shape {xy.shape} in {path}.")
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


def tracks_from_result_file(
    result_path: os.PathLike,
    *,
    score_threshold: float = 0.9,
    min_track_len: int = 5,
    max_step: float | None = None,
    stride: Sequence[int] | None = None,
    tile_shape_yx: Sequence[int] = (64, 64),
) -> List[Track]:
    """Convert one tile result file into global-coordinate tracks."""
    result_path = Path(result_path)
    obj, xy, h, diffusion = load_inference_arrays(result_path, tile_shape_yx=tile_shape_yx)
    locations = parse_tile_locations(result_path, count=obj.shape[0], stride=stride)
    tracks: list[Track] = []
    for sample_index in range(obj.shape[0]):
        loc = locations[sample_index]
        for query_index in range(obj.shape[2]):
            keep = obj[sample_index, :, query_index] >= score_threshold
            if int(np.sum(keep)) < min_track_len:
                continue
            frames = np.nonzero(keep)[0] + loc.t_start
            xy_local = xy[sample_index, keep, query_index, :]
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


def _is_duplicate(track: Track, kept: Track, min_overlap: int, distance_threshold: float) -> bool:
    frames_a = track.points[:, 0].astype(np.int64)
    frames_b = kept.points[:, 0].astype(np.int64)
    common, idx_a, idx_b = np.intersect1d(frames_a, frames_b, return_indices=True)
    if common.size < min_overlap:
        return False
    dist = np.sqrt(np.sum((track.points[idx_a, 1:3] - kept.points[idx_b, 1:3]) ** 2, axis=1))
    return int(np.sum(dist <= distance_threshold)) >= min_overlap


def deduplicate_tracks(
    tracks: Sequence[Track],
    *,
    min_overlap: int = 5,
    distance_threshold: float = 1.0,
) -> List[Track]:
    """Remove repeated tracks using overlapping frames and spatial distance.

    Higher mean confidence wins, with track length as the tie breaker. Unlike
    the MATLAB helper, this compares actual overlapping global frames rather
    than assuming aligned row positions in a cell array.
    """
    ranked = sorted(tracks, key=lambda tr: (tr.mean_score, tr.length), reverse=True)
    kept: list[Track] = []
    for track in ranked:
        if any(_is_duplicate(track, existing, min_overlap, distance_threshold) for existing in kept):
            continue
        kept.append(track)
    return [replace(track, track_id=index) for index, track in enumerate(kept)]


def stitch_inference_results(
    result_paths: Iterable[os.PathLike],
    *,
    score_threshold: float = 0.9,
    min_track_len: int = 5,
    max_step: float | None = None,
    deduplicate: bool = True,
    dedup_overlap: int = 5,
    dedup_distance: float = 1.0,
    stride: Sequence[int] | None = None,
    tile_shape_yx: Sequence[int] = (64, 64),
) -> List[Track]:
    """Load per-tile inference outputs, transform to global coordinates, and deduplicate."""
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
            )
        )
    if deduplicate:
        return deduplicate_tracks(tracks, min_overlap=dedup_overlap, distance_threshold=dedup_distance)
    return [replace(track, track_id=index) for index, track in enumerate(tracks)]


def write_tracks_csv(tracks: Sequence[Track], output_path: os.PathLike) -> None:
    """Write stitched tracks as one CSV row per detected point."""
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
    parser.add_argument("--dedup-distance", type=float, default=1.0)
    parser.add_argument("--stride", nargs=3, type=int, metavar=("T", "Y", "X"), help="Required for legacy block###_x#_y#_t# names.")
    parser.add_argument("--tile-shape", nargs=2, type=int, default=(64, 64), metavar=("Y", "X"), help="Tile size used to scale normalized xy predictions.")
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
    )
    write_tracks_csv(tracks, args.output)
    print(f"Wrote {len(tracks)} stitched track(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
