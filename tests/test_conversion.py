import h5py
import numpy as np
import pytest
import tifffile

from sptnet.data.conversion import convert_mat_file_to_tiff, expand_file_patterns


def _write_mat(path, name, data):
    with h5py.File(path, "w") as handle:
        handle.create_dataset(name, data=data)


def test_convert_3d_mat_movie_to_imagej_tiff(tmp_path):
    mat_path = tmp_path / "movie.mat"
    output_dir = tmp_path / "tiffs"
    data = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
    _write_mat(mat_path, "timelapsedata", data)

    results = convert_mat_file_to_tiff(mat_path, output_dir=output_dir, dtype="uint16")

    assert len(results) == 1
    result = results[0]
    assert result.dataset_name == "timelapsedata"
    assert result.source_shape == (2, 3, 4)
    assert result.output_shape == (2, 3, 4)
    assert result.sample_index is None
    assert not result.skipped
    np.testing.assert_array_equal(tifffile.imread(result.output_path), data)


def test_convert_4d_mat_movie_splits_samples_and_respects_axes(tmp_path):
    mat_path = tmp_path / "batch.mat"
    data = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    _write_mat(mat_path, "ims", data)

    results = convert_mat_file_to_tiff(mat_path, output_dir=tmp_path, input_axes="NTXY")

    assert [result.sample_index for result in results] == [0, 1]
    assert [result.output_shape for result in results] == [(3, 5, 4), (3, 5, 4)]
    np.testing.assert_array_equal(tifffile.imread(results[1].output_path), np.transpose(data[1], (0, 2, 1)))


def test_convert_mat_file_to_tiff_rejects_bad_axis_metadata(tmp_path):
    mat_path = tmp_path / "movie.mat"
    _write_mat(mat_path, "ims", np.zeros((2, 3, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="duplicate axis"):
        convert_mat_file_to_tiff(mat_path, input_axes="TYY")


def test_expand_file_patterns_deduplicates_and_sorts(tmp_path):
    first = tmp_path / "b.mat"
    second = tmp_path / "a.mat"
    first.touch()
    second.touch()

    assert expand_file_patterns([str(first), str(tmp_path / "*.mat")]) == [second, first]
