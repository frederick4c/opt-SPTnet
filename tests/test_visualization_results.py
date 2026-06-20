from pathlib import Path

import numpy as np
import h5py
from matplotlib.animation import FuncAnimation

from sptnet.segmentation.stitch import Track
from sptnet.visualization.results import (
    build_stitched_tracks_animation,
    find_tiff_result_pairs,
    load_trackmate_ground_truth_for_stitched_results,
    tracks_from_csv,
)


def test_find_tiff_result_pairs_prefers_h5_over_legacy_mat(tmp_path):
    tiff_dir = tmp_path / "tiffs"
    result_dir = tmp_path / "results"
    tiff_dir.mkdir()
    result_dir.mkdir()

    tiff_path = tiff_dir / "movie_000.tif"
    h5_result = result_dir / "result_movie_000.h5"
    mat_result = result_dir / "result_movie_000.mat"
    tiff_path.touch()
    h5_result.touch()
    mat_result.touch()

    pairs = find_tiff_result_pairs(test_data_dir=tiff_dir, results_dir=result_dir)

    assert pairs == [(str(tiff_path), str(h5_result))]


def test_build_stitched_tracks_animation_returns_animation():
    video = np.zeros((4, 8, 8), dtype=np.float32)
    track = Track(
        points=np.array(
            [
                [0, 2, 3, 0.95],
                [1, 2.5, 3.5, 0.96],
                [2, 3, 4, 0.97],
            ],
            dtype=np.float32,
        ),
        h=0.5,
        diffusion=0.1,
        query_index=0,
        sample_index=0,
        tile_path=Path("tile.h5"),
        track_id=7,
    )

    gt = np.full((1, 4, 2), np.nan, dtype=np.float32)
    gt[0, 0] = [3.0, 2.0]
    gt[0, 1] = [3.5, 2.5]

    ani = build_stitched_tracks_animation(
        video,
        [track],
        num_frames=3,
        interval=10,
        ground_truth_positions=gt,
        show_predicted_constants=True,
    )

    assert isinstance(ani, FuncAnimation)
    # Use the public frame-sequence API rather than the private ``_save_count``
    # attribute, which is not part of matplotlib's stable interface.
    assert len(list(ani.new_frame_seq())) == 3
    ani._draw_next_frame(0, blit=False)


def test_load_trackmate_ground_truth_for_stitched_results_reads_optional_dataset(tmp_path):
    path = tmp_path / "movie.h5"
    gt = np.zeros((2, 3, 2), dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("timelapsedata", data=np.zeros((3, 4, 4), dtype=np.float32))
        handle.create_dataset("trackmate_positions", data=gt)

    loaded = load_trackmate_ground_truth_for_stitched_results(path)

    np.testing.assert_array_equal(loaded, gt)


def test_tracks_from_csv_loads_stitched_tracks(tmp_path):
    path = tmp_path / "tracks.csv"
    path.write_text(
        "\n".join(
            [
                "track_id,frame,y,x,score,h,diffusion,query_index,sample_index,tile_path",
                "2,1,4.0,5.0,0.91,0.5,0.1,7,0,tile_a.h5",
                "2,0,3.0,4.0,0.90,0.5,0.1,7,0,tile_a.h5",
                "5,3,8.0,9.0,0.95,0.6,0.2,8,1,tile_b.h5",
            ]
        )
        + "\n"
    )

    tracks = tracks_from_csv(path)

    assert [track.track_id for track in tracks] == [2, 5]
    np.testing.assert_array_equal(tracks[0].points[:, 0], np.array([0, 1], dtype=np.float32))
    np.testing.assert_allclose(tracks[0].points[1], np.array([1, 4, 5, 0.91], dtype=np.float32))
    assert tracks[0].query_index == 7
    assert tracks[1].sample_index == 1
