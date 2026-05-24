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
    )

    assert len(tracks) == 2
    np.testing.assert_array_equal(tracks[0].points[:, 0], np.array([0, 1, 2], dtype=np.float32))
    np.testing.assert_array_equal(tracks[1].points[:, 0], np.array([2, 3, 4], dtype=np.float32))


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
