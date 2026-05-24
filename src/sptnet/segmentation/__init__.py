"""Python segmentation and stitching utilities for SPTnet movies."""

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
