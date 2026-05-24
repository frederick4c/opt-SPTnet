"""Utilities for combining TIFF movies with TrackMate XML tracks."""

from __future__ import annotations

import argparse
import csv
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import h5py
import numpy as np
import tifffile


TRACK_TABLE_COLUMNS = ("track_id", "frame", "x", "y", "z")


@dataclass(frozen=True)
class TrackMateTrack:
    """One TrackMate particle/track."""

    track_id: int
    detections: np.ndarray  # columns: frame, x, y, z

    @property
    def length(self) -> int:
        return int(self.detections.shape[0])


@dataclass(frozen=True)
class CombinedTrackMateResult:
    """Summary of one TIFF/XML combination."""

    tiff_path: Path
    xml_path: Path
    output_path: Path
    movie_shape: tuple[int, int, int]
    num_tracks: int
    num_detections: int


def load_tiff_movie(tiff_path: os.PathLike, *, dtype: str | None = None) -> np.ndarray:
    """Load a TIFF movie as ``T,Y,X``.

    A 2D TIFF is treated as a single-frame movie. Higher-dimensional TIFFs are
    rejected so callers do not accidentally flatten channels or z-stacks.
    """
    movie = np.asarray(tifffile.imread(tiff_path))
    if movie.ndim == 2:
        movie = movie[np.newaxis, ...]
    if movie.ndim != 3:
        raise ValueError(f"{tiff_path} must be a 2D image or 3D T,Y,X movie, got shape {movie.shape}.")
    if isinstance(dtype, str) and dtype.lower() == "none":
        dtype = None
    if dtype is not None:
        movie = movie.astype(np.dtype(dtype), copy=False)
    return movie


def parse_trackmate_xml(xml_path: os.PathLike) -> List[TrackMateTrack]:
    """Parse TrackMate XML tracks from the simple ``Tracks/particle`` export."""
    root = ET.parse(xml_path).getroot()
    particles = list(root.findall("particle"))
    if not particles:
        particles = list(root.findall(".//particle"))
    if not particles:
        raise ValueError(f"{xml_path} does not contain TrackMate particle tracks.")

    tracks: list[TrackMateTrack] = []
    for track_id, particle in enumerate(particles):
        rows = []
        for detection in particle.findall("detection"):
            rows.append(
                [
                    int(float(detection.attrib["t"])),
                    float(detection.attrib["x"]),
                    float(detection.attrib["y"]),
                    float(detection.attrib.get("z", 0.0)),
                ]
            )
        if not rows:
            continue
        detections = np.asarray(rows, dtype=np.float32)
        order = np.argsort(detections[:, 0], kind="stable")
        tracks.append(TrackMateTrack(track_id=track_id, detections=detections[order]))
    return tracks


def tracks_to_table(tracks: Sequence[TrackMateTrack]) -> np.ndarray:
    """Flatten tracks to a table with columns ``track_id, frame, x, y, z``."""
    rows = []
    for track in tracks:
        if track.length == 0:
            continue
        track_ids = np.full((track.length, 1), track.track_id, dtype=np.float32)
        rows.append(np.concatenate([track_ids, track.detections.astype(np.float32, copy=False)], axis=1))
    if not rows:
        return np.empty((0, len(TRACK_TABLE_COLUMNS)), dtype=np.float32)
    return np.vstack(rows).astype(np.float32, copy=False)


def tracks_to_dense_positions(
    tracks: Sequence[TrackMateTrack],
    num_frames: int,
    *,
    coordinate_order: str = "xy",
) -> np.ndarray:
    """Return dense per-track positions with NaNs for missing frames.

    The default output shape is ``tracks, frames, 2`` with coordinates
    ``x,y``. Pass ``coordinate_order="yx"`` for row/column ordering.
    """
    coordinate_order = coordinate_order.lower()
    if coordinate_order not in {"xy", "yx"}:
        raise ValueError("coordinate_order must be 'xy' or 'yx'.")

    positions = np.full((len(tracks), num_frames, 2), np.nan, dtype=np.float32)
    for track_index, track in enumerate(tracks):
        frames = track.detections[:, 0].astype(np.int64)
        valid = (frames >= 0) & (frames < num_frames)
        frames = frames[valid]
        xy = track.detections[valid, 1:3]
        if coordinate_order == "yx":
            xy = xy[:, ::-1]
        positions[track_index, frames, :] = xy
    return positions


def combine_tiff_trackmate(
    tiff_path: os.PathLike,
    xml_path: os.PathLike,
    output_path: Optional[os.PathLike] = None,
    *,
    movie_dataset: str = "timelapsedata",
    dtype: str | None = "float32",
    overwrite: bool = True,
) -> CombinedTrackMateResult:
    """Combine one TIFF movie and matching TrackMate XML into one HDF5 file.

    The output contains:

    - ``timelapsedata``: movie stack in ``T,Y,X`` order.
    - ``trackmate_tracks``: flat table with columns stored in the
      ``columns`` attribute: ``track_id, frame, x, y, z``.
    - ``trackmate_positions``: dense ``tracks, frames, 2`` array in ``x,y``
      order with NaNs where a track is absent.
    - ``trackmate_lengths``: number of detections per track.
    """
    tiff_path = Path(tiff_path)
    xml_path = Path(xml_path)
    if output_path is None:
        output_path = tiff_path.with_name(f"{tiff_path.stem}_trackmate.h5")
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass overwrite=True to replace it.")

    movie = load_tiff_movie(tiff_path, dtype=dtype)
    tracks = parse_trackmate_xml(xml_path)
    table = tracks_to_table(tracks)
    dense_positions = tracks_to_dense_positions(tracks, movie.shape[0], coordinate_order="xy")
    lengths = np.asarray([track.length for track in tracks], dtype=np.int32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        movie_ds = handle.create_dataset(movie_dataset, data=movie)
        movie_ds.attrs["axes"] = "TYX"

        table_ds = handle.create_dataset("trackmate_tracks", data=table)
        table_ds.attrs["columns"] = np.asarray(TRACK_TABLE_COLUMNS, dtype="S")

        pos_ds = handle.create_dataset("trackmate_positions", data=dense_positions)
        pos_ds.attrs["axes"] = "track,frame,xy"

        handle.create_dataset("trackmate_lengths", data=lengths)
        handle.attrs["format"] = "sptnet-tiff-trackmate"
        handle.attrs["source_tiff"] = str(tiff_path)
        handle.attrs["source_trackmate_xml"] = str(xml_path)
        handle.attrs["num_tracks"] = len(tracks)
        handle.attrs["num_detections"] = int(table.shape[0])
        handle.attrs["space_units"] = root_attr(xml_path, "spaceUnits", default="")
        handle.attrs["time_units"] = root_attr(xml_path, "timeUnits", default="")
        handle.attrs["frame_interval"] = float(root_attr(xml_path, "frameInterval", default="1.0"))

    return CombinedTrackMateResult(
        tiff_path=tiff_path,
        xml_path=xml_path,
        output_path=output_path,
        movie_shape=tuple(int(dim) for dim in movie.shape),
        num_tracks=len(tracks),
        num_detections=int(table.shape[0]),
    )


def root_attr(xml_path: os.PathLike, name: str, *, default: str) -> str:
    """Read one root XML attribute without exposing XML details to callers."""
    return ET.parse(xml_path).getroot().attrib.get(name, default)


def write_trackmate_csv(tracks: Sequence[TrackMateTrack], output_path: os.PathLike) -> None:
    """Write TrackMate tracks to CSV for quick inspection."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(TRACK_TABLE_COLUMNS)
        for row in tracks_to_table(tracks):
            writer.writerow([int(row[0]), int(row[1]), float(row[2]), float(row[3]), float(row[4])])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combine a TIFF movie with matching TrackMate XML tracks.")
    parser.add_argument("tiff", help="Input TIFF movie.")
    parser.add_argument("xml", help="TrackMate XML file exported as Tracks/particle/detection.")
    parser.add_argument("-o", "--output", default=None, help="Output HDF5 path. Defaults beside TIFF.")
    parser.add_argument("--dtype", default="float32", help="Movie dtype, or 'none' to preserve TIFF dtype.")
    parser.add_argument("--no-overwrite", action="store_true", help="Fail if output already exists.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    dtype = None if args.dtype.lower() == "none" else args.dtype
    result = combine_tiff_trackmate(
        args.tiff,
        args.xml,
        output_path=args.output,
        dtype=dtype,
        overwrite=not args.no_overwrite,
    )
    print(
        f"Wrote {result.output_path} "
        f"(movie_shape={result.movie_shape}, tracks={result.num_tracks}, detections={result.num_detections})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
