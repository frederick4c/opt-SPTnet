import csv

import h5py
import numpy as np
import pytest
import scipy.io as sio
import tifffile

from sptnet.inference.results_io import stack_result_arrays, write_inference_result_file
from sptnet.segmentation import (
    deduplicate_tracks,
    split_movie_file,
    stitch_inference_results,
    stitch_movie_tiles,
)
from sptnet.segmentation.stitch import Track, parse_tile_location


def _write_hdf5(path, name, data):
    with h5py.File(path, "w") as handle:
        handle.create_dataset(name, data=data)


def test_split_hdf5_movie_writes_named_tyx_tiles_and_manifest(tmp_path):
    movie = np.arange(5 * 4 * 4, dtype=np.float32).reshape(5, 4, 4)
    input_path = tmp_path / "movie.h5"
    _write_hdf5(input_path, "timelapsedata", movie)

    result = split_movie_file(
        input_path,
        output_dir=tmp_path / "tiles",
        block_shape=(3, 3, 3),
        overlap=(1, 1, 1),
        dtype="none",
    )

    assert len(result.tiles) == 8
    first = result.tiles[0]
    assert first.output_path.name == "movie_x001_y001.h5"
    assert result.manifest_path.exists()

    with h5py.File(first.output_path, "r") as handle:
        assert handle["timelapsedata"].shape == (2, 3, 3, 3)
        np.testing.assert_array_equal(handle["timelapsedata"][0], movie[:3, :3, :3])
        assert handle.attrs["format"] == "sptnet-segmentation-tile"
        assert handle.attrs["t_start"] == 0
        np.testing.assert_array_equal(handle.attrs["t_starts"], np.array([0, 2]))
        assert handle.attrs["y_start"] == 0
        assert handle.attrs["x_start"] == 0


def test_split_matlab_order_mat_file_to_python_tyx_tiles(tmp_path):
    # MATLAB legacy segmentation comments use H,W,T,N; the Python repo wants N,T,Y,X/T,Y,X.
    matlab_order = np.arange(2 * 3 * 4 * 1, dtype=np.float32).reshape(2, 3, 4, 1)
    input_path = tmp_path / "legacy.mat"
    sio.savemat(input_path, {"ims": matlab_order})

    result = split_movie_file(
        input_path,
        output_dir=tmp_path / "tiles",
        input_axes="YXTN",
        block_shape=(4, 2, 3),
        dtype="none",
    )

    assert len(result.tiles) == 1
    expected = np.transpose(matlab_order[..., 0], (2, 0, 1))
    with h5py.File(result.tiles[0].output_path, "r") as handle:
        assert handle["timelapsedata"].shape == (1, 4, 2, 3)
        np.testing.assert_array_equal(handle["timelapsedata"][0], expected)


def test_split_tiff_movie_writes_unlabeled_hdf5_tiles(tmp_path):
    movie = np.arange(5 * 4 * 4, dtype=np.uint16).reshape(5, 4, 4)
    input_path = tmp_path / "movie.tif"
    tifffile.imwrite(input_path, movie)

    result = split_movie_file(
        input_path,
        output_dir=tmp_path / "tiles",
        block_shape=(3, 4, 4),
        overlap=(1, 0, 0),
        dtype="none",
    )

    assert result.dataset_name == "timelapsedata"
    assert result.original_shape == movie.shape
    assert result.input_axes == "TYX"
    assert len(result.tiles) == 2
    with h5py.File(result.tiles[0].output_path, "r") as handle:
        assert set(handle.keys()) == {"timelapsedata"}
        assert handle["timelapsedata"].dtype == movie.dtype
        assert handle["timelapsedata"].shape == (2, 3, 4, 4)
        np.testing.assert_array_equal(handle["timelapsedata"][0], movie[:3])
        np.testing.assert_array_equal(handle["timelapsedata"][1], movie[2:5])
        assert handle.attrs["source_file"].endswith("movie.tif")


def test_split_movie_file_aligns_edge_tiles_to_reduce_padding(tmp_path):
    movie = np.arange(3 * 4 * 6, dtype=np.float32).reshape(3, 4, 6)
    input_path = tmp_path / "movie.h5"
    _write_hdf5(input_path, "timelapsedata", movie)

    result = split_movie_file(
        input_path,
        output_dir=tmp_path / "tiles",
        block_shape=(3, 4, 4),
        overlap=(0, 0, 0),
        dtype="none",
    )

    x_starts = sorted({tile.x_start for tile in result.tiles})
    assert x_starts == [0, 2]
    assert result.tiles[-1].padded_shape == movie.shape
    with h5py.File(tmp_path / "tiles" / "movie_x002_y001.h5", "r") as handle:
        assert handle.attrs["x_start"] == 2
        np.testing.assert_array_equal(handle["timelapsedata"][0], movie[:, :, 2:6])


def test_stitch_movie_tiles_averages_overlap_and_crops(tmp_path):
    movie = np.arange(4 * 3 * 3, dtype=np.float32).reshape(4, 3, 3)
    input_path = tmp_path / "movie.h5"
    _write_hdf5(input_path, "timelapsedata", movie)
    result = split_movie_file(
        input_path,
        output_dir=tmp_path / "tiles",
        block_shape=(3, 2, 2),
        overlap=(2, 1, 1),
        dtype="none",
    )

    stitched = stitch_movie_tiles(
        [tile.output_path for tile in result.tiles],
        tmp_path / "stitched.h5",
        source_shape=movie.shape,
    )

    np.testing.assert_array_equal(stitched, movie)


def test_inference_dataset_reads_all_temporal_clips_from_segmentation_tile(tmp_path):
    pytest.importorskip("torch")
    from sptnet.data.inference_dataset import FileSampleDataset

    movie = np.arange(5 * 4 * 4, dtype=np.float32).reshape(5, 4, 4)
    input_path = tmp_path / "movie.h5"
    _write_hdf5(input_path, "timelapsedata", movie)
    result = split_movie_file(
        input_path,
        output_dir=tmp_path / "tiles",
        block_shape=(3, 4, 4),
        overlap=(1, 0, 0),
        dtype="none",
    )

    dataset = FileSampleDataset([str(result.tiles[0].output_path)])

    assert len(dataset) == 2
    np.testing.assert_array_equal(dataset[0]["video"], movie[:3])
    np.testing.assert_array_equal(dataset[1]["video"], movie[2:5])


def test_parse_tile_location_supports_result_prefix_and_legacy_names(tmp_path):
    loc = parse_tile_location(tmp_path / "result_movie_n003_x003_y004.h5", stride=(30, 64, 64))
    assert (loc.sample_index, loc.t_start, loc.y_start, loc.x_start) == (2, 0, 192, 128)

    loc = parse_tile_location(tmp_path / "result_movie_n003_t002_x003_y004.h5", stride=(30, 64, 64))
    assert (loc.sample_index, loc.t_start, loc.y_start, loc.x_start) == (2, 30, 192, 128)

    loc = parse_tile_location(tmp_path / "result_movie__n002__t0030_y0064_x0128.h5")
    assert (loc.sample_index, loc.t_start, loc.y_start, loc.x_start) == (2, 30, 64, 128)

    legacy = parse_tile_location(tmp_path / "resultblock001_x2_y3_t4.mat", stride=(30, 64, 64))
    assert (legacy.sample_index, legacy.t_start, legacy.y_start, legacy.x_start) == (0, 90, 64, 128)


def test_stitch_inference_results_globalizes_and_deduplicates_tracks(tmp_path):
    result_a = tmp_path / "result_movie_x001_y001.h5"
    result_b = tmp_path / "result_movie_x002_y001.h5"

    records = {
        "obj_estimation": [np.ones((1, 1, 6), dtype=np.float32)],
        "estimation_xy": [np.zeros((1, 1, 6, 2), dtype=np.float32)],
        "estimation_H": [np.array([[0.5]], dtype=np.float32)],
        "estimation_C": [np.array([[0.25]], dtype=np.float32)],
    }
    arrays = stack_result_arrays(records)
    write_inference_result_file(result_a, arrays, source_file="tile_a.h5")
    write_inference_result_file(result_b, arrays, source_file="tile_b.h5")

    tracks = stitch_inference_results(
        [result_a, result_b],
        score_threshold=0.9,
        min_track_len=5,
        dedup_overlap=5,
        dedup_distance=1.1,
        stride=(1, 1, 1),
    )

    assert len(tracks) == 1
    assert tracks[0].track_id == 0
    np.testing.assert_array_equal(tracks[0].points[:, 0], np.arange(6, dtype=np.float32))
    np.testing.assert_allclose(tracks[0].points[:, 1], 32.0)
    np.testing.assert_allclose(tracks[0].points[:, 2], 32.0)


def test_stitch_inference_results_uses_temporal_starts_from_tile_source(tmp_path):
    movie = np.zeros((5, 4, 4), dtype=np.float32)
    input_path = tmp_path / "movie.h5"
    _write_hdf5(input_path, "timelapsedata", movie)
    split = split_movie_file(
        input_path,
        output_dir=tmp_path / "tiles",
        block_shape=(3, 4, 4),
        overlap=(1, 0, 0),
        dtype="none",
    )
    tile_path = split.tiles[0].output_path
    result_path = tmp_path / f"result_{tile_path.name}"
    records = {
        "obj_estimation": [np.ones((1, 1, 3), dtype=np.float32), np.ones((1, 1, 3), dtype=np.float32)],
        "estimation_xy": [np.zeros((1, 1, 3, 2), dtype=np.float32), np.zeros((1, 1, 3, 2), dtype=np.float32)],
        "estimation_H": [np.array([[0.5]], dtype=np.float32), np.array([[0.6]], dtype=np.float32)],
        "estimation_C": [np.array([[0.25]], dtype=np.float32), np.array([[0.35]], dtype=np.float32)],
    }
    write_inference_result_file(result_path, stack_result_arrays(records), source_file=tile_path)

    tracks = stitch_inference_results(
        [result_path],
        score_threshold=0.9,
        min_track_len=3,
        deduplicate=False,
        tile_shape_yx=(4, 4),
    )

    assert len(tracks) == 2
    np.testing.assert_array_equal(tracks[0].points[:, 0], np.array([0, 1, 2], dtype=np.float32))
    np.testing.assert_array_equal(tracks[1].points[:, 0], np.array([2, 3, 4], dtype=np.float32))


def test_stitch_inference_results_recovers_stale_source_file_from_sibling_tile(tmp_path):
    tile_dir = tmp_path / "tiles"
    result_dir = tile_dir / "inference_results"
    tile_dir.mkdir()
    result_dir.mkdir()
    tile_path = tile_dir / "full_realdata_x002_y001.h5"
    with h5py.File(tile_path, "w") as handle:
        handle.create_dataset("timelapsedata", data=np.zeros((1, 3, 4, 4), dtype=np.float32))
        handle.attrs["format"] = "sptnet-segmentation-tile"
        handle.attrs["sample_index"] = 0
        handle.attrs["t_start"] = 0
        handle.attrs["t_starts"] = np.array([0])
        handle.attrs["y_start"] = 0
        handle.attrs["x_start"] = 4
        handle.attrs["source_shape_tyx"] = np.array([3, 4, 8])

    result_path = result_dir / "result_full_realdata_x002_y001.h5"
    records = {
        "obj_estimation": [np.ones((1, 1, 3), dtype=np.float32)],
        "estimation_xy": [np.zeros((1, 1, 3, 2), dtype=np.float32)],
        "estimation_H": [np.array([[0.5]], dtype=np.float32)],
        "estimation_C": [np.array([[0.25]], dtype=np.float32)],
    }
    write_inference_result_file(result_path, stack_result_arrays(records), source_file=tmp_path / "old_tiles" / tile_path.name)

    tracks = stitch_inference_results(
        [result_path],
        score_threshold=0.9,
        min_track_len=3,
        deduplicate=False,
        tile_shape_yx=(4, 4),
    )

    assert len(tracks) == 1
    np.testing.assert_allclose(tracks[0].points[:, 2], 6.0)


def test_stitch_inference_results_recovers_missing_tile_from_manifest(tmp_path):
    tile_dir = tmp_path / "tiles"
    result_dir = tmp_path / "results"
    tile_dir.mkdir()
    result_dir.mkdir()
    tile_name = "movie_x002_y001.h5"
    manifest_path = tile_dir / "movie__segmentation_manifest.csv"
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
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for t_index, t_start in enumerate([0, 3]):
            writer.writerow(
                {
                    "output_path": tile_dir / tile_name,
                    "source_path": tmp_path / "movie.tif",
                    "dataset_name": "timelapsedata",
                    "sample_index": 0,
                    "tile_index": t_index,
                    "t_index": t_index,
                    "y_index": 0,
                    "x_index": 1,
                    "t_start": t_start,
                    "y_start": 0,
                    "x_start": 2,
                    "block_t": 3,
                    "block_y": 4,
                    "block_x": 4,
                    "stride_t": 3,
                    "stride_y": 4,
                    "stride_x": 4,
                    "source_t": 6,
                    "source_y": 4,
                    "source_x": 6,
                    "padded_t": 6,
                    "padded_y": 4,
                    "padded_x": 6,
                    "skipped": False,
                }
            )

    result_path = result_dir / f"result_{tile_name}"
    records = {
        "obj_estimation": [np.ones((1, 1, 3), dtype=np.float32), np.ones((1, 1, 3), dtype=np.float32)],
        "estimation_xy": [np.zeros((1, 1, 3, 2), dtype=np.float32), np.zeros((1, 1, 3, 2), dtype=np.float32)],
        "estimation_H": [np.array([[0.5]], dtype=np.float32), np.array([[0.6]], dtype=np.float32)],
        "estimation_C": [np.array([[0.25]], dtype=np.float32), np.array([[0.35]], dtype=np.float32)],
    }
    write_inference_result_file(result_path, stack_result_arrays(records), source_file=tile_dir / tile_name)

    tracks = stitch_inference_results(
        [result_path],
        score_threshold=0.9,
        min_track_len=3,
        deduplicate=False,
        tile_shape_yx=(4, 4),
    )

    assert len(tracks) == 2
    np.testing.assert_array_equal(tracks[0].points[:, 0], np.array([0, 1, 2], dtype=np.float32))
    np.testing.assert_array_equal(tracks[1].points[:, 0], np.array([3, 4, 5], dtype=np.float32))
    np.testing.assert_allclose(tracks[0].points[:, 2], 4.0)


def test_stitch_inference_results_uses_yx_coordinate_order_by_default(tmp_path):
    tile_path = tmp_path / "movie_x001_y001.h5"
    movie = np.zeros((1, 3, 64, 64), dtype=np.float32)
    movie[:, :, 20, 10] = 100.0
    with h5py.File(tile_path, "w") as handle:
        handle.create_dataset("timelapsedata", data=movie)
        handle.attrs["format"] = "sptnet-segmentation-tile"
        handle.attrs["sample_index"] = 0
        handle.attrs["t_start"] = 0
        handle.attrs["t_starts"] = np.array([0])
        handle.attrs["y_start"] = 0
        handle.attrs["x_start"] = 0

    result_path = tmp_path / "result_movie_x001_y001.h5"
    raw_y = (20.0 - 32.0) / 32.0
    raw_x = (10.0 - 32.0) / 32.0
    records = {
        "obj_estimation": [np.ones((1, 1, 3), dtype=np.float32)],
        "estimation_xy": [np.array([[[[raw_y, raw_x], [raw_y, raw_x], [raw_y, raw_x]]]], dtype=np.float32)],
        "estimation_H": [np.array([[0.5]], dtype=np.float32)],
        "estimation_C": [np.array([[0.25]], dtype=np.float32)],
    }
    write_inference_result_file(result_path, stack_result_arrays(records), source_file=tile_path)

    tracks = stitch_inference_results(
        [result_path],
        score_threshold=0.9,
        min_track_len=3,
        deduplicate=False,
    )

    assert len(tracks) == 1
    np.testing.assert_allclose(tracks[0].points[:, 1], 20.0)
    np.testing.assert_allclose(tracks[0].points[:, 2], 10.0)


def test_stitch_inference_results_drops_predictions_in_padded_tile_region(tmp_path):
    tile_path = tmp_path / "movie_x002_y001.h5"
    with h5py.File(tile_path, "w") as handle:
        handle.create_dataset("timelapsedata", data=np.zeros((1, 3, 4, 4), dtype=np.float32))
        handle.attrs["format"] = "sptnet-segmentation-tile"
        handle.attrs["sample_index"] = 0
        handle.attrs["t_start"] = 0
        handle.attrs["t_starts"] = np.array([0])
        handle.attrs["y_start"] = 0
        handle.attrs["x_start"] = 4
        handle.attrs["source_shape_tyx"] = np.array([3, 4, 6])

    inside_y = (1.0 - 2.0) / 2.0
    inside_x = (1.0 - 2.0) / 2.0
    padded_y = inside_y
    padded_x = (3.0 - 2.0) / 2.0
    result_path = tmp_path / "result_movie_x002_y001.h5"
    records = {
        "obj_estimation": [np.ones((2, 1, 3), dtype=np.float32)],
        "estimation_xy": [
            np.array(
                [
                    [[[inside_y, inside_x], [inside_y, inside_x], [inside_y, inside_x]]],
                    [[[padded_y, padded_x], [padded_y, padded_x], [padded_y, padded_x]]],
                ],
                dtype=np.float32,
            )
        ],
        "estimation_H": [np.array([[0.5], [0.6]], dtype=np.float32)],
        "estimation_C": [np.array([[0.25], [0.35]], dtype=np.float32)],
    }
    write_inference_result_file(result_path, stack_result_arrays(records), source_file=tile_path)

    tracks = stitch_inference_results(
        [result_path],
        score_threshold=0.9,
        min_track_len=3,
        deduplicate=False,
        tile_shape_yx=(4, 4),
    )

    assert len(tracks) == 1
    np.testing.assert_allclose(tracks[0].points[:, 1], 1.0)
    np.testing.assert_allclose(tracks[0].points[:, 2], 5.0)


def test_deduplicate_tracks_keeps_highest_scoring_overlap():
    low = Track(
        points=np.column_stack([np.arange(5), np.zeros(5), np.zeros(5), np.full(5, 0.9)]),
        h=0.1,
        diffusion=0.2,
        query_index=0,
        sample_index=0,
        tile_path="low.h5",
    )
    high = Track(
        points=np.column_stack([np.arange(5), np.full(5, 0.5), np.full(5, 0.5), np.full(5, 0.99)]),
        h=0.1,
        diffusion=0.2,
        query_index=1,
        sample_index=0,
        tile_path="high.h5",
    )

    kept = deduplicate_tracks([low, high], min_overlap=5, distance_threshold=1.0)

    assert len(kept) == 1
    assert kept[0].query_index == 1
    assert kept[0].track_id == 0


def test_deduplicate_tracks_merges_nearby_overlapping_tile_tracks():
    left_tile = Track(
        points=np.column_stack(
            [
                np.arange(6),
                np.full(6, 10.0),
                np.full(6, 12.0),
                np.full(6, 0.92),
            ]
        ).astype(np.float32),
        h=0.4,
        diffusion=0.1,
        query_index=0,
        sample_index=0,
        tile_path="left.h5",
    )
    right_tile = Track(
        points=np.column_stack(
            [
                np.arange(3, 9),
                np.full(6, 11.5),
                np.full(6, 13.5),
                np.full(6, 0.95),
            ]
        ).astype(np.float32),
        h=0.6,
        diffusion=0.3,
        query_index=1,
        sample_index=0,
        tile_path="right.h5",
    )

    kept = deduplicate_tracks([left_tile, right_tile], min_overlap=3, distance_threshold=3.0)

    assert len(kept) == 1
    np.testing.assert_array_equal(kept[0].points[:, 0], np.arange(9, dtype=np.float32))
    np.testing.assert_allclose(kept[0].points[3:6, 1], 11.5)
    np.testing.assert_allclose(kept[0].points[3:6, 2], 13.5)


def test_stitch_cli_csv_shape_is_simple_for_downstream_tools(tmp_path):
    csv_path = tmp_path / "tracks.csv"
    track = Track(
        points=np.array([[1, 2, 3, 0.95]], dtype=np.float32),
        h=0.4,
        diffusion=0.1,
        query_index=2,
        sample_index=0,
        tile_path=tmp_path / "tile.h5",
        track_id=7,
    )

    from sptnet.segmentation import write_tracks_csv

    write_tracks_csv([track], csv_path)

    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["track_id"] == "7"
    assert rows[0]["frame"] == "1"
    assert rows[0]["query_index"] == "2"
