from types import SimpleNamespace

import h5py
import numpy as np

from sptnet.training.data_generator import (
    PSFParams,
    SimulationParams,
    ZernikePSF,
    generate_training_file,
    make_otf_rescale_kernel,
    perlin_noise,
)


def _tiny_sim_params(**overrides):
    params = {
        "num_files": 1,
        "videos_per_file": 2,
        "frames": 3,
        "image_dims": 8,
        "p_num_min": 1,
        "p_num_max": 2,
    }
    params.update(overrides)
    return SimulationParams(**params)


def _read_first_sample(path):
    torch = __import__("pytest").importorskip("torch")
    assert torch is not None
    from sptnet.data.mat_dataset import TransformerMatDataset

    with TransformerMatDataset(SimpleNamespace(num_queries=4, image_size=8), path) as dataset:
        sample = dataset[0]
    return sample


def test_zernike_psf_outputs_finite_nonnegative_frames():
    psf = ZernikePSF(PSFParams(psf_size=32), box_size=8)

    psfs = psf.generate(
        np.array([0.0, np.nan, 1.25]),
        np.array([0.0, np.nan, -0.5]),
        np.zeros(3),
    )

    assert psfs.shape == (3, 8, 8)
    assert np.isfinite(psfs).all()
    assert np.all(psfs >= 0)
    assert psfs[0].sum() > 0
    assert psfs[1].sum() == 0
    assert psf.norm_parameter > 0


def test_otf_kernel_and_perlin_noise_are_bounded():
    kernel = make_otf_rescale_kernel(8, 0.157, 0.95, 0.95)
    noise = perlin_noise(8, np.random.default_rng(123))

    assert kernel.shape == (8, 8)
    assert np.isfinite(kernel).all()
    assert np.all(kernel >= 0)
    assert noise.shape == (8, 8)
    assert np.min(noise) >= 0
    assert np.max(noise) <= 1


def test_generator_is_deterministic_for_same_seed(tmp_path):
    sim = _tiny_sim_params()
    psf = PSFParams(psf_size=32)
    path_a = tmp_path / "a" / "trainingvideos_1.mat"
    path_b = tmp_path / "b" / "trainingvideos_1.mat"

    generate_training_file(path_a, sim_params=sim, psf_params=psf, seed=42, file_index=1)
    generate_training_file(path_b, sim_params=sim, psf_params=psf, seed=42, file_index=1)

    with h5py.File(path_a, "r") as a, h5py.File(path_b, "r") as b:
        np.testing.assert_array_equal(a["timelapsedata"][()], b["timelapsedata"][()])
        np.testing.assert_array_equal(a["bglabel"][()], b["bglabel"][()])
        assert a.attrs["seed"] == 42
        assert b.attrs["seed"] == 42

    sample_a = _read_first_sample(path_a)
    sample_b = _read_first_sample(path_b)
    np.testing.assert_array_equal(sample_a["video"], sample_b["video"])
    np.testing.assert_allclose(sample_a["Hlabel"], sample_b["Hlabel"])
    np.testing.assert_allclose(sample_a["Clabel"], sample_b["Clabel"])
    np.testing.assert_allclose(sample_a["position"], sample_b["position"], equal_nan=True)


def test_generator_changes_for_different_seed(tmp_path):
    sim = _tiny_sim_params()
    psf = PSFParams(psf_size=32)
    path_a = tmp_path / "trainingvideos_1.mat"
    path_b = tmp_path / "trainingvideos_2.mat"

    generate_training_file(path_a, sim_params=sim, psf_params=psf, seed=42, file_index=1)
    generate_training_file(path_b, sim_params=sim, psf_params=psf, seed=43, file_index=2)

    with h5py.File(path_a, "r") as a, h5py.File(path_b, "r") as b:
        assert not np.array_equal(a["timelapsedata"][()], b["timelapsedata"][()])


def test_generated_hdf5_schema_matches_training_loader_expectations(tmp_path):
    sim = _tiny_sim_params()
    path = tmp_path / "trainingvideos_1.mat"

    generate_training_file(path, sim_params=sim, psf_params=PSFParams(psf_size=32), seed=7, file_index=1)

    with h5py.File(path, "r") as handle:
        assert handle["timelapsedata"].shape == (2, 3, 8, 8)
        assert handle["timelapsedata"].dtype == np.float32
        assert handle["Hlabel"].shape == (2, 2)
        assert handle["Hlabel"].attrs["MATLAB_class"] == b"cell"
        first_h = handle[handle["Hlabel"][0, 0]]
        first_pos = handle[handle["traceposition"][0, 0]]
        assert first_h.shape == (1, 1)
        assert first_h.dtype == np.float32
        assert first_pos.shape == (2, 3)
        assert handle.attrs["generator"] == "sptnet-python-training-data"

    sample = _read_first_sample(path)
    assert sample["video"].shape == (3, 8, 8)
    assert sample["position"].shape == (3, 4, 2)
    assert sample["class_label"].shape == (3, 4)
