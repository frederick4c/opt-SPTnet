import h5py
import numpy as np
import pytest

from sptnet.training.crlb import (
    compute_crlb_matrix,
    load_or_generate_crlb_matrix,
    save_crlb_matrix,
    validate_crlb_matrix,
)


def _reference_crlb_block(steps, h_idx, d_idx, diff_step=0.01):
    hurst = (h_idx + 1) * 0.01
    diffusion = (d_idx + 1) * diff_step
    t = np.arange(1, steps + 1, dtype=np.float64)[:, None]
    s = np.arange(1, steps + 1, dtype=np.float64)[None, :]
    tau = np.abs(t - s)
    tau_mask = tau > 0

    covariance_kernel = t ** (2 * hurst) + s ** (2 * hurst) - tau ** (2 * hurst)
    hurst_derivative = 2 * np.log(t) * t ** (2 * hurst) + 2 * np.log(s) * s ** (2 * hurst)
    hurst_derivative = np.array(hurst_derivative, copy=True)
    hurst_derivative[tau_mask] -= 2 * tau[tau_mask] ** (2 * hurst) * np.log(tau[tau_mask])

    covariance = diffusion * covariance_kernel
    covariance += 1e-8 * np.max(np.diag(covariance)) * np.eye(steps)
    cholesky = np.linalg.cholesky(covariance)
    solve = lambda matrix: np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, matrix))

    x = solve(diffusion * hurst_derivative)
    w = solve(covariance_kernel)
    fisher = np.array(
        [
            [0.5 * np.sum(x * x.T), 0.5 * np.sum(x * w.T)],
            [0.5 * np.sum(x * w.T), 0.5 * np.sum(w * w.T)],
        ]
    )
    return np.linalg.solve(fisher, np.eye(2))


def test_compute_crlb_matrix_shape_and_reference_samples():
    matrix = compute_crlb_matrix(frame_number=6, diff_max=0.05, diff_step=0.01)

    assert matrix.shape == (2, 2, 5, 99, 6)
    assert np.all(matrix[:, :, :, :, 0] == 0)
    assert np.isfinite(matrix[:, :, :, :, 1:]).all()

    sample_points = [
        (2, 0, 0),
        (4, 49, 2),
        (6, 98, 4),
    ]
    for steps, h_idx, d_idx in sample_points:
        np.testing.assert_allclose(
            matrix[:, :, d_idx, h_idx, steps - 1],
            _reference_crlb_block(steps, h_idx, d_idx),
            rtol=1e-10,
            atol=1e-10,
        )


def test_save_crlb_matrix_writes_training_dataset(tmp_path):
    matrix = compute_crlb_matrix(frame_number=3, diff_max=0.02, diff_step=0.01)
    output_path = tmp_path / "CRLB_H_D_frame.mat"

    save_crlb_matrix(output_path, matrix)

    with h5py.File(output_path, "r") as handle:
        dataset = handle["CRLB_matrix_HD_frame"]
        assert dataset.shape == (2, 2, 2, 99, 3)
        assert dataset.dtype == np.float64
        assert dataset.attrs["MATLAB_class"] == b"double"
        np.testing.assert_allclose(dataset[()], matrix)


def test_load_or_generate_crlb_matrix_creates_missing_file(tmp_path):
    output_path = tmp_path / "missing_crlb.mat"

    matrix = load_or_generate_crlb_matrix(
        output_path,
        frame_number=3,
        diff_max=0.02,
        progress=False,
    )

    assert output_path.is_file()
    assert matrix.shape == (2, 2, 2, 99, 3)
    with h5py.File(output_path, "r") as handle:
        np.testing.assert_allclose(handle["CRLB_matrix_HD_frame"][()], matrix)


def test_load_or_generate_crlb_matrix_reuses_compatible_file(tmp_path):
    output_path = tmp_path / "compatible_crlb.mat"
    existing = compute_crlb_matrix(frame_number=4, diff_max=0.03, diff_step=0.01)
    save_crlb_matrix(output_path, existing)

    loaded = load_or_generate_crlb_matrix(
        output_path,
        frame_number=3,
        diff_max=0.02,
        progress=False,
    )

    np.testing.assert_allclose(loaded, existing)


def test_load_or_generate_crlb_matrix_regenerates_too_small_file(tmp_path):
    output_path = tmp_path / "small_crlb.mat"
    existing = compute_crlb_matrix(frame_number=3, diff_max=0.02, diff_step=0.01)
    save_crlb_matrix(output_path, existing)

    loaded = load_or_generate_crlb_matrix(
        output_path,
        frame_number=4,
        diff_max=0.02,
        progress=False,
    )

    assert loaded.shape == (2, 2, 2, 99, 4)
    with h5py.File(output_path, "r") as handle:
        assert handle["CRLB_matrix_HD_frame"].shape == (2, 2, 2, 99, 4)


def test_validate_crlb_matrix_rejects_wrong_values():
    matrix = compute_crlb_matrix(frame_number=4, diff_max=0.03, diff_step=0.01)
    matrix[:, :, 1, 49, 3] = 0

    with pytest.raises(ValueError, match="do not match"):
        validate_crlb_matrix(matrix, frame_number=4, diff_max=0.03)


def test_load_or_generate_crlb_matrix_regenerates_wrong_values(tmp_path):
    output_path = tmp_path / "stale_crlb.mat"
    stale = compute_crlb_matrix(frame_number=4, diff_max=0.03, diff_step=0.01)
    stale[:, :, 1, 49, 3] = 0
    save_crlb_matrix(output_path, stale)

    loaded = load_or_generate_crlb_matrix(
        output_path,
        frame_number=4,
        diff_max=0.03,
        progress=False,
    )

    validate_crlb_matrix(loaded, frame_number=4, diff_max=0.03)
    with h5py.File(output_path, "r") as handle:
        validate_crlb_matrix(handle["CRLB_matrix_HD_frame"][()], frame_number=4, diff_max=0.03)
