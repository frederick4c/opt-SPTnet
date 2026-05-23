import h5py
import numpy as np
import scipy.io as sio

from sptnet.inference.results_io import (
    result_output_path,
    stack_result_arrays,
    write_inference_result_file,
)
from sptnet.visualization import load_inference_results


def _records():
    return {
        "obj_estimation": [np.ones((1, 2, 3), dtype=np.float32)],
        "estimation_xy": [np.zeros((1, 2, 3, 2), dtype=np.float32)],
        "estimation_H": [np.full((2, 1), 0.5, dtype=np.float32)],
        "estimation_C": [np.full((2, 1), 0.25, dtype=np.float32)],
    }


def test_inference_result_path_uses_mat_only_for_mat_inputs(tmp_path):
    assert result_output_path(tmp_path, "movie.mat").endswith("result_movie.mat")
    assert result_output_path(tmp_path, "movie.h5").endswith("result_movie.h5")
    assert result_output_path(tmp_path, "movie.hdf5").endswith("result_movie.h5")
    assert result_output_path(tmp_path, "movie.tif").endswith("result_movie.h5")


def test_write_inference_result_file_uses_native_hdf5_for_h5_inputs(tmp_path):
    arrays = stack_result_arrays(_records())
    output_path = result_output_path(tmp_path, "movie.h5")

    write_inference_result_file(output_path, arrays, source_file="movie.h5")

    with h5py.File(output_path, "r") as handle:
        assert handle.attrs["format"] == "sptnet-inference-results"
        assert handle.attrs["source_file"] == "movie.h5"
        assert handle["obj_estimation"].shape == (1, 1, 2, 3)
        assert handle["estimation_xy"].shape == (1, 2, 3, 2)
        assert handle["estimation_H"].shape == (1, 2)
        assert handle["estimation_C"].shape == (1, 2)

    obj_est, xy_est, est_h, est_c = load_inference_results(output_path)
    assert obj_est.shape == (1, 3, 2)
    assert xy_est.shape == (1, 3, 2, 2)
    assert est_h.shape == (1, 2)
    assert est_c.shape == (1, 2)


def test_write_inference_result_file_keeps_mat_for_mat_inputs(tmp_path):
    arrays = stack_result_arrays(_records())
    output_path = result_output_path(tmp_path, "movie.mat")

    write_inference_result_file(output_path, arrays, source_file="movie.mat")

    data = sio.loadmat(output_path)
    assert data["obj_estimation"].shape == (1, 1, 2, 3)
    assert data["estimation_xy"].shape == (1, 2, 3, 2)
    assert data["estimation_H"].shape == (1, 2)
    assert data["estimation_C"].shape == (1, 2)
