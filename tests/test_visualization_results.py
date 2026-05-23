from pathlib import Path

from sptnet.visualization.results import find_tiff_result_pairs


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
