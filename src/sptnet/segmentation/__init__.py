"""Python segmentation and stitching utilities for large SPTnet movies.

The public API covers three common steps:

``split_movie_file`` / ``split_movie_files``
    Convert large ``.h5``, ``.mat``, or TIFF movies into fixed-size HDF5 tiles
    suitable for SPTnet inference.
``stitch_inference_results``
    Convert per-tile inference outputs back into global tracks and merge
    duplicates from overlapping tiles.
``stitch_movie_tiles``
    Reconstruct raw movie tiles for validation or debugging.
"""

from sptnet.segmentation.split import (
    DEFAULT_BLOCK_SHAPE,
    DEFAULT_DATASET_NAMES,
    SegmentationResult,
    TileMetadata,
    split_movie_file,
    split_movie_files,
)
from sptnet.segmentation.stitch import (
    Track,
    deduplicate_tracks,
    stitch_inference_results,
    stitch_movie_tiles,
    write_tracks_csv,
)

__all__ = [
    "DEFAULT_BLOCK_SHAPE",
    "DEFAULT_DATASET_NAMES",
    "SegmentationResult",
    "TileMetadata",
    "Track",
    "deduplicate_tracks",
    "split_movie_file",
    "split_movie_files",
    "stitch_inference_results",
    "stitch_movie_tiles",
    "write_tracks_csv",
]
