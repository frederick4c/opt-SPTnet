"""SPTnet training data generation.

This module is a Python port of the SPTnet MATLAB training-data generator by
Cheng Bi, Huang Lab, Purdue University. The port was written by Fred Lawrence,
University of Cambridge, for the refactored optSPTnet package.

The implementation preserves the MATLAB simulation model as closely as
practical in pure Python: fractional Brownian motion trajectories, Wyant
Zernike pupil PSFs, OTF rescaling, Perlin background, Poisson noise, optional
motion blur, and HDF5 output compatible with the existing SPTnet loaders.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import convolve2d


@dataclass(frozen=True)
class SimulationParams:
    num_files: int = 10
    videos_per_file: int = 100
    frames: int = 30
    image_dims: int = 64
    p_num_min: int = 1
    p_num_max: int = 10
    hurst_min: float = 0.0001
    hurst_max: float = 0.9999
    d_min: float = 0.001
    d_max: float = 0.5
    motion_blur: bool = False
    file_start: int = 1
    oversampling: int = 10


@dataclass(frozen=True)
class PSFParams:
    na: float = 1.49
    wavelength: float = 0.69
    refractive_index: float = 1.518
    otf_sigma_x: float = 0.95
    otf_sigma_y: float = 0.95
    pixel_size: float = 0.157
    psf_size: int = 128
    n_med: float = 1.33
    photon_min: float = 300.0
    photon_max: float = 5000.0
    bg_min: float = 1.0
    bg_max: float = 25.0
    perlin_bg_min: float = 0.0
    perlin_bg_max: float = 10.0


def _env_number(name: str, default: float | int) -> float | int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        if isinstance(default, int):
            return int(float(raw))
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be numeric, got {raw!r}.") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def params_from_environment() -> tuple[SimulationParams, PSFParams, int | None, Path]:
    """Read generator parameters from the MATLAB-compatible environment."""
    sim = SimulationParams(
        num_files=_env_number("SPT_NUM_FILES", 10),
        videos_per_file=_env_number("SPT_VIDEOS_PER_FILE", 100),
        frames=_env_number("SPT_FRAMES", 30),
        image_dims=_env_number("SPT_IMAGE_DIMS", 64),
        p_num_min=_env_number("SPT_P_NUM_MIN", 1),
        p_num_max=_env_number("SPT_P_NUM_MAX", 10),
        hurst_min=_env_number("SPT_HURST_MIN", 0.0001),
        hurst_max=_env_number("SPT_HURST_MAX", 0.9999),
        d_min=_env_number("SPT_D_MIN", 0.001),
        d_max=_env_number("SPT_D_MAX", 0.5),
        motion_blur=_env_bool("SPT_MOTION_BLUR", False),
        file_start=_env_number("SPT_FILE_START", 1),
    )
    psf = PSFParams(
        na=_env_number("SPT_NA", 1.49),
        wavelength=_env_number("SPT_LAMBDA", 0.69),
        refractive_index=_env_number("SPT_REFRACTIVE_INDEX", 1.518),
        otf_sigma_x=_env_number("SPT_OTF_SIGMA_X", 0.95),
        otf_sigma_y=_env_number("SPT_OTF_SIGMA_Y", 0.95),
        pixel_size=_env_number("SPT_PIXELSIZE", 0.157),
        psf_size=_env_number("SPT_PSF_SIZE", 128),
        n_med=_env_number("SPT_N_MED", 1.33),
        photon_min=_env_number("SPT_PHOTON_MIN", 300.0),
        photon_max=_env_number("SPT_PHOTON_MAX", 5000.0),
        bg_min=_env_number("SPT_BG_MIN", 1.0),
        bg_max=_env_number("SPT_BG_MAX", 25.0),
        perlin_bg_min=_env_number("SPT_PERLIN_BG_MIN", 0.0),
        perlin_bg_max=_env_number("SPT_PERLIN_BG_MAX", 10.0),
    )
    seed_raw = os.getenv("SPT_SEED")
    seed = None if seed_raw in {None, ""} else int(seed_raw)
    output_dir = Path(os.getenv("SPT_OUTPUT_DIR") or Path.cwd() / "TestData" / "training_data")
    return sim, psf, seed, output_dir


def _wyant_zernike_matrix(
    phi: np.ndarray,
    k_r: np.ndarray,
    na: float,
    wavelength: float,
    coefficient_count: int,
) -> np.ndarray:
    n_max = int(round(np.sqrt(coefficient_count) - 1))
    if (n_max + 1) ** 2 != coefficient_count:
        raise ValueError("Wyant Zernike coefficient count must be a perfect square.")

    freq_max = na / wavelength
    aperture = k_r < freq_max
    rho = (k_r * aperture) / freq_max
    theta = phi * aperture
    terms = np.zeros((*rho.shape, coefficient_count), dtype=np.float64)
    terms[:, :, 0] = aperture.astype(np.float64)

    idx = 1
    for n in range(1, n_max + 1):
        rho_powers = [np.ones_like(rho)]
        for _ in range(2 * n):
            rho_powers.append(rho_powers[-1] * rho)

        for m in range(n, -1, -1):
            coeffs = []
            for k in range(0, n - m + 1):
                numerator = 1.0
                for value in range(n - m - k + 1, 2 * n - m - k + 1):
                    numerator *= value
                denominator = float(math.factorial(k) * math.factorial(n - k))
                coeffs.append(((-1) ** k) * numerator / denominator)

            radial = np.zeros_like(rho)
            powers = range(2 * n - m, m - 1, -2)
            for coeff, power in zip(coeffs, powers):
                radial = radial + coeff * rho_powers[power]
            radial = radial * aperture

            if m == 0:
                terms[:, :, idx] = radial
                idx += 1
            else:
                terms[:, :, idx] = radial * np.cos(m * theta)
                terms[:, :, idx + 1] = radial * np.sin(m * theta)
                idx += 2
    return terms


class ZernikePSF:
    """Python port of the MATLAB ``PSF_zernike`` path used by the generator."""

    def __init__(
        self,
        params: PSFParams,
        box_size: int,
        *,
        zernike_phase: np.ndarray | None = None,
        zernike_mag: np.ndarray | None = None,
    ) -> None:
        self.params = params
        self.box_size = int(box_size)
        self.psf_size = int(params.psf_size)
        self.zernike_phase = np.zeros(25, dtype=np.float64) if zernike_phase is None else zernike_phase
        self.zernike_mag = np.zeros(25, dtype=np.float64) if zernike_mag is None else zernike_mag
        if zernike_mag is None:
            self.zernike_mag[0] = 1.0

        grid = np.arange(-self.psf_size / 2, self.psf_size / 2, dtype=np.float64)
        x, y = np.meshgrid(grid, grid)
        self.radius = np.sqrt(x * x + y * y)
        self.phi = np.arctan2(y, x)
        self.k_r = self.radius / (self.psf_size * params.pixel_size)
        self.na_constrain = self.k_r < (params.na / params.wavelength)
        kz_squared = (params.refractive_index / params.wavelength) ** 2 - self.k_r**2
        self.k_z = np.sqrt(np.maximum(kz_squared, 0.0)) * self.na_constrain

        z_terms = _wyant_zernike_matrix(
            self.phi,
            self.k_r,
            params.na,
            params.wavelength,
            len(self.zernike_phase),
        )
        self.pupil_phase = np.tensordot(z_terms, self.zernike_phase, axes=([2], [0]))
        self.pupil_mag = np.tensordot(z_terms, self.zernike_mag, axes=([2], [0]))
        self.norm_parameter = float(np.sum(self.pupil_mag))

    def generate(self, x_positions: np.ndarray, y_positions: np.ndarray, z_positions: np.ndarray) -> np.ndarray:
        n_frames = len(x_positions)
        psfs = np.zeros((n_frames, self.box_size, self.box_size), dtype=np.float64)
        start = self.psf_size // 2 - self.box_size // 2
        end = start + self.box_size
        base_phase = np.exp(1j * self.pupil_phase)
        kx = self.k_r * np.cos(self.phi)
        ky = self.k_r * np.sin(self.phi)

        for idx, (x_pos, y_pos, z_pos) in enumerate(zip(x_positions, y_positions, z_positions)):
            if not (np.isfinite(x_pos) and np.isfinite(y_pos) and np.isfinite(z_pos)):
                continue
            shift = -kx * x_pos * self.params.pixel_size - ky * y_pos * self.params.pixel_size
            shift_phase = np.exp(-2j * np.pi * shift)
            defocus_phase = np.exp(2j * np.pi * z_pos * self.k_z)
            pupil_complex = self.pupil_mag * base_phase * shift_phase * defocus_phase
            psf_amp = np.abs(np.fft.fftshift(np.fft.fft2(pupil_complex)))
            psfs[idx] = (psf_amp[start:end, start:end] ** 2) / (self.psf_size**2)
        return psfs


def make_otf_rescale_kernel(size: int, pixel_size: float, sigma_x: float, sigma_y: float) -> np.ndarray:
    cropsize = min(29, size)
    sigma_xr = 1.0 / (2.0 * np.pi * sigma_x)
    sigma_yr = 1.0 / (2.0 * np.pi * sigma_y)
    grid = np.arange(-size / 2, size / 2, dtype=np.float64)
    x, y = np.meshgrid(grid, grid)
    xx = x * pixel_size
    yy = y * pixel_size
    kernel_full = (
        2
        * np.pi
        * sigma_x
        * sigma_y
        * np.exp(-(xx**2) / (2 * sigma_xr**2))
        * np.exp(-(yy**2) / (2 * sigma_yr**2))
    )
    start = size // 2 - cropsize // 2
    kernel = kernel_full[start : start + cropsize, start : start + cropsize] * pixel_size**2
    return kernel


def apply_otf_rescale(psfs: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    output = np.empty_like(psfs, dtype=np.float64)
    for idx in range(psfs.shape[0]):
        output[idx] = convolve2d(psfs[idx], kernel, mode="same", boundary="fill", fillvalue=0)
    return output


def fractional_brownian_motion_2d(
    hurst: float,
    steps: int,
    diffusion: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(1, steps + 1, dtype=np.float64)[:, None]
    s = np.arange(1, steps + 1, dtype=np.float64)[None, :]
    covariance = 0.5 * (t ** (2 * hurst) + s ** (2 * hurst) - np.abs(t - s) ** (2 * hurst))
    try:
        chol = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        jitter = 1e-12 * max(float(np.max(np.diag(covariance))), 1.0)
        chol = np.linalg.cholesky(covariance + jitter * np.eye(steps))
    scale = np.sqrt(2.0 * diffusion)
    return chol @ (scale * rng.standard_normal(steps)), chol @ (scale * rng.standard_normal(steps))


def uniform_duration_and_present(
    frames: int,
    num_particles: int,
    min_duration: int,
    max_duration: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    occupancy = np.zeros(frames, dtype=np.float64)
    durations = rng.integers(min_duration, max_duration + 1, size=num_particles)
    durations = np.minimum(durations, frames)
    order = np.argsort(-durations, kind="stable")
    sorted_tracks: list[tuple[int, int, int]] = []

    for duration in durations[order]:
        best_score = np.inf
        best_starts: list[int] = []
        for start in range(0, frames - int(duration) + 1):
            trial = occupancy.copy()
            trial[start : start + int(duration)] += 1
            score = float(np.sum((trial - np.mean(trial)) ** 2))
            if score < best_score - 1e-12:
                best_score = score
                best_starts = [start]
            elif abs(score - best_score) < 1e-12:
                best_starts.append(start)
        start = int(rng.choice(best_starts))
        end = start + int(duration) - 1
        occupancy[start : end + 1] += 1
        sorted_tracks.append((start, end, int(duration)))

    tracks = [None] * num_particles
    for sorted_idx, original_idx in enumerate(order):
        tracks[int(original_idx)] = sorted_tracks[sorted_idx]
    return tracks  # type: ignore[return-value]


def _random_unit_square_disk(shape: tuple[int, int], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    total = shape[0] * shape[1]
    accepted_x: list[np.ndarray] = []
    accepted_y: list[np.ndarray] = []
    count = 0
    while count < total:
        candidate_count = max(total - count, total // 2, 8)
        xy = rng.random((candidate_count, 2))
        mask = (xy[:, 0] - 0.5) ** 2 + (xy[:, 1] - 0.5) ** 2 <= 0.25
        accepted = xy[mask]
        if accepted.size == 0:
            continue
        accepted_x.append(accepted[:, 0])
        accepted_y.append(accepted[:, 1])
        count += accepted.shape[0]
    ux = (np.concatenate(accepted_x)[:total].reshape(shape) - 0.5) * 2.0
    uy = (np.concatenate(accepted_y)[:total].reshape(shape) - 0.5) * 2.0
    return ux, uy


def _perlin_noise_generate(size: int, x_grid: int, y_grid: int, rng: np.random.Generator) -> np.ndarray:
    x_vals = np.linspace(0, x_grid, size + 1, dtype=np.float64)
    y_vals = np.linspace(0, y_grid, size + 1, dtype=np.float64)
    ux, uy = _random_unit_square_disk((y_grid + 2, x_grid + 2), rng)
    out = np.zeros((size + 1, size + 1), dtype=np.float64)

    for j, x in enumerate(x_vals):
        gx = int(np.floor(x))
        dx = x - np.floor(x)
        for k, y in enumerate(y_vals):
            gy = int(np.floor(y))
            dy = y - np.floor(y)
            n00 = ux[gy, gx] * dx + uy[gy, gx] * dy
            n10 = ux[gy, gx + 1] * (dx - 1) + uy[gy, gx + 1] * dy
            n01 = ux[gy + 1, gx] * dx + uy[gy + 1, gx] * (dy - 1)
            n11 = ux[gy + 1, gx + 1] * (dx - 1) + uy[gy + 1, gx + 1] * (dy - 1)
            wx = 6 * dx**5 - 15 * dx**4 + 10 * dx**3
            wy = 6 * dy**5 - 15 * dy**4 + 10 * dy**3
            n0 = (1 - wx) * n00 + wx * n10
            n1 = (1 - wx) * n01 + wx * n11
            out[k, j] = (1 - wy) * n0 + wy * n1
    return np.clip(out[:size, :size] + 0.5, 0.0, 1.0)


def perlin_noise(size: int, rng: np.random.Generator) -> np.ndarray:
    zmat = np.zeros((size, size), dtype=np.float64)
    for level in range(1, int(np.floor(np.log2(size)))):
        weight = rng.random()
        zmat += weight * _perlin_noise_generate(size, 2**level, 2**level, rng)
    z_min = float(np.min(zmat))
    z_max = float(np.max(zmat))
    if z_max == z_min:
        return np.zeros_like(zmat)
    return (zmat - z_min) / (z_max - z_min)


def _rand_range(rng: np.random.Generator, min_value: float, max_value: float) -> float:
    return float(min_value + (max_value - min_value) * rng.random())


def _ref_dataset(
    group: h5py.Group,
    name: str,
    data: np.ndarray | float | int,
    matlab_class: bytes,
) -> h5py.Reference:
    if matlab_class == np.bytes_("single"):
        array = np.asarray(data, dtype=np.float32)
    elif matlab_class == np.bytes_("double"):
        array = np.asarray(data, dtype=np.float64)
    else:
        array = np.asarray(data)
    dataset = group.create_dataset(name, data=array)
    dataset.attrs["MATLAB_class"] = matlab_class
    return dataset.ref


def _write_cell_refs(
    handle: h5py.File,
    name: str,
    refs: np.ndarray,
) -> None:
    dataset = handle.create_dataset(name, data=refs, dtype=h5py.ref_dtype)
    dataset.attrs["MATLAB_class"] = np.bytes_("cell")


def write_training_file(
    output_path: str | Path,
    *,
    timelapsedata: np.ndarray,
    hlabel: list[list[float]],
    clabel: list[list[float]],
    photonlabel: list[list[float]],
    traceposition: list[list[np.ndarray]],
    moleculeid: list[list[int]],
    duration: list[list[int]],
    bglabel: np.ndarray,
    perlinbglabel: np.ndarray,
    sim_params: SimulationParams,
    psf_params: PSFParams,
    seed: int | None,
    file_index: int,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        refs_group = handle.create_group("#refs#")
        handle.attrs["generator"] = "sptnet-python-training-data"
        handle.attrs["seed"] = -1 if seed is None else int(seed)
        handle.attrs["file_index"] = int(file_index)
        handle.attrs["simulation_params_json"] = json.dumps(asdict(sim_params), sort_keys=True)
        handle.attrs["psf_params_json"] = json.dumps(asdict(psf_params), sort_keys=True)

        video_ds = handle.create_dataset("timelapsedata", data=timelapsedata.astype(np.float32))
        video_ds.attrs["MATLAB_class"] = np.bytes_("single")
        bg_ds = handle.create_dataset("bglabel", data=bglabel.reshape(-1, 1).astype(np.float32))
        bg_ds.attrs["MATLAB_class"] = np.bytes_("single")
        perlin_ds = handle.create_dataset("Perlinbglabel", data=perlinbglabel.reshape(-1, 1).astype(np.float32))
        perlin_ds.attrs["MATLAB_class"] = np.bytes_("single")

        shape = (sim_params.p_num_max, sim_params.videos_per_file)
        h_refs = np.empty(shape, dtype=h5py.ref_dtype)
        c_refs = np.empty(shape, dtype=h5py.ref_dtype)
        photon_refs = np.empty(shape, dtype=h5py.ref_dtype)
        pos_refs = np.empty(shape, dtype=h5py.ref_dtype)
        id_refs = np.empty(shape, dtype=h5py.ref_dtype)
        duration_refs = np.empty(shape, dtype=h5py.ref_dtype)

        counter = 0
        for video_idx in range(sim_params.videos_per_file):
            for particle_idx in range(sim_params.p_num_max):
                active = particle_idx < len(hlabel[video_idx])
                suffix = f"{counter:08d}"
                counter += 1
                h_refs[particle_idx, video_idx] = _ref_dataset(
                    refs_group,
                    f"h_{suffix}",
                    [[hlabel[video_idx][particle_idx] if active else 0.0]],
                    np.bytes_("single"),
                )
                c_refs[particle_idx, video_idx] = _ref_dataset(
                    refs_group,
                    f"c_{suffix}",
                    [[clabel[video_idx][particle_idx] if active else 0.0]],
                    np.bytes_("single"),
                )
                photon_refs[particle_idx, video_idx] = _ref_dataset(
                    refs_group,
                    f"photons_{suffix}",
                    [[photonlabel[video_idx][particle_idx] if active else 0.0]],
                    np.bytes_("single"),
                )
                id_refs[particle_idx, video_idx] = _ref_dataset(
                    refs_group,
                    f"id_{suffix}",
                    [[moleculeid[video_idx][particle_idx] if active else 0.0]],
                    np.bytes_("double"),
                )
                duration_refs[particle_idx, video_idx] = _ref_dataset(
                    refs_group,
                    f"duration_{suffix}",
                    [[duration[video_idx][particle_idx] if active else 0.0]],
                    np.bytes_("double"),
                )
                pos = (
                    traceposition[video_idx][particle_idx].T.astype(np.float32)
                    if active
                    else np.zeros((2, sim_params.frames), dtype=np.float32)
                )
                pos_refs[particle_idx, video_idx] = _ref_dataset(
                    refs_group,
                    f"position_{suffix}",
                    pos,
                    np.bytes_("single"),
                )

        _write_cell_refs(handle, "Hlabel", h_refs)
        _write_cell_refs(handle, "Clabel", c_refs)
        _write_cell_refs(handle, "photonlabel", photon_refs)
        _write_cell_refs(handle, "traceposition", pos_refs)
        _write_cell_refs(handle, "moleculeid", id_refs)
        _write_cell_refs(handle, "duration", duration_refs)


def generate_training_file(
    output_path: str | Path,
    *,
    sim_params: SimulationParams,
    psf_params: PSFParams,
    seed: int | None,
    file_index: int = 1,
) -> Path:
    rng = np.random.default_rng(seed)
    psf_model = ZernikePSF(psf_params, sim_params.image_dims)
    otf_kernel = make_otf_rescale_kernel(
        sim_params.image_dims,
        psf_params.pixel_size,
        psf_params.otf_sigma_x,
        psf_params.otf_sigma_y,
    )

    videos = np.zeros(
        (sim_params.videos_per_file, sim_params.frames, sim_params.image_dims, sim_params.image_dims),
        dtype=np.float32,
    )
    bglabel = np.zeros(sim_params.videos_per_file, dtype=np.float32)
    perlinbglabel = np.zeros(sim_params.videos_per_file, dtype=np.float32)
    hlabel: list[list[float]] = []
    clabel: list[list[float]] = []
    photonlabel: list[list[float]] = []
    traceposition: list[list[np.ndarray]] = []
    moleculeid: list[list[int]] = []
    duration: list[list[int]] = []

    for video_idx in range(sim_params.videos_per_file):
        bg_level = _rand_range(rng, psf_params.bg_min, psf_params.bg_max)
        perlin_bg = (
            0.0
            if psf_params.perlin_bg_min < 1
            else _rand_range(rng, psf_params.perlin_bg_min, psf_params.perlin_bg_max)
        )
        psf_frame_count = sim_params.frames * sim_params.oversampling if sim_params.motion_blur else sim_params.frames
        psf_all = np.zeros((psf_frame_count, sim_params.image_dims, sim_params.image_dims), dtype=np.float64)

        num_particles = int(rng.integers(sim_params.p_num_min, sim_params.p_num_max + 1))
        tracks = uniform_duration_and_present(sim_params.frames, num_particles, 2, sim_params.frames, rng)
        video_h: list[float] = []
        video_c: list[float] = []
        video_photons: list[float] = []
        video_positions: list[np.ndarray] = []
        video_ids: list[int] = []
        video_durations: list[int] = []

        for particle_idx, (start, end, track_duration) in enumerate(tracks, start=1):
            hurst = _rand_range(rng, sim_params.hurst_min, sim_params.hurst_max)
            diffusion = _rand_range(rng, sim_params.d_min, sim_params.d_max)
            photons = _rand_range(rng, psf_params.photon_min, psf_params.photon_max)
            x_offset = _rand_range(rng, -(sim_params.image_dims / 2) + 4, (sim_params.image_dims / 2) - 4)
            y_offset = _rand_range(rng, -(sim_params.image_dims / 2) + 4, (sim_params.image_dims / 2) - 4)

            if sim_params.motion_blur:
                os_duration = track_duration * sim_params.oversampling
                os_start = start * sim_params.oversampling
                traj_x, traj_y = fractional_brownian_motion_2d(hurst, os_duration, diffusion, rng)
                scale = (1.0 / sim_params.oversampling) ** hurst
                xpos = np.full(psf_frame_count, np.nan, dtype=np.float64)
                ypos = np.full(psf_frame_count, np.nan, dtype=np.float64)
                xpos[os_start : os_start + os_duration] = scale * traj_x + x_offset
                ypos[os_start : os_start + os_duration] = scale * traj_y + y_offset
                zpos = np.zeros(psf_frame_count, dtype=np.float64)
                with np.errstate(invalid="ignore"):
                    trace_x = np.nanmean(xpos.reshape(sim_params.frames, sim_params.oversampling), axis=1)
                    trace_y = np.nanmean(ypos.reshape(sim_params.frames, sim_params.oversampling), axis=1)
                particle_psf = apply_otf_rescale(psf_model.generate(xpos, ypos, zpos), otf_kernel)
                psf_all += np.nan_to_num(particle_psf / psf_model.norm_parameter * (photons / sim_params.oversampling))
                trace = np.stack([trace_x, trace_y], axis=1)
            else:
                traj_x, traj_y = fractional_brownian_motion_2d(hurst, track_duration, diffusion, rng)
                xpos = np.full(sim_params.frames, np.nan, dtype=np.float64)
                ypos = np.full(sim_params.frames, np.nan, dtype=np.float64)
                xpos[start : end + 1] = traj_x + x_offset
                ypos[start : end + 1] = traj_y + y_offset
                zpos = np.zeros(sim_params.frames, dtype=np.float64)
                particle_psf = apply_otf_rescale(psf_model.generate(xpos, ypos, zpos), otf_kernel)
                psf_all += np.nan_to_num(particle_psf / psf_model.norm_parameter * photons)
                trace = np.stack([xpos, ypos], axis=1)

            video_h.append(hurst)
            video_c.append(diffusion)
            video_photons.append(photons)
            video_positions.append(trace.astype(np.float32))
            video_ids.append(particle_idx)
            video_durations.append(track_duration)

        if sim_params.motion_blur:
            psf_movie = psf_all.reshape(sim_params.frames, sim_params.oversampling, sim_params.image_dims, sim_params.image_dims)
            psf_movie = np.sum(psf_movie, axis=1)
        else:
            psf_movie = psf_all
        background = bg_level + perlin_bg * perlin_noise(sim_params.image_dims, rng)
        lam = np.maximum(psf_movie + background[None, :, :], 0.0)
        videos[video_idx] = rng.poisson(lam).astype(np.float32)
        bglabel[video_idx] = bg_level
        perlinbglabel[video_idx] = perlin_bg
        hlabel.append(video_h)
        clabel.append(video_c)
        photonlabel.append(video_photons)
        traceposition.append(video_positions)
        moleculeid.append(video_ids)
        duration.append(video_durations)

    write_training_file(
        output_path,
        timelapsedata=videos,
        hlabel=hlabel,
        clabel=clabel,
        photonlabel=photonlabel,
        traceposition=traceposition,
        moleculeid=moleculeid,
        duration=duration,
        bglabel=bglabel,
        perlinbglabel=perlinbglabel,
        sim_params=sim_params,
        psf_params=psf_params,
        seed=seed,
        file_index=file_index,
    )
    return Path(output_path)


def generate_training_data(
    output_dir: str | Path,
    *,
    sim_params: SimulationParams,
    psf_params: PSFParams,
    seed: int | None,
    progress: bool = True,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for file_offset in range(sim_params.num_files):
        file_index = sim_params.file_start + file_offset
        path = output_dir / f"trainingvideos_{file_index}.mat"
        start = time.time()
        if progress:
            print(f"Generating file {file_index} ({file_offset + 1}/{sim_params.num_files})...")
        file_seed = None if seed is None else seed + file_offset
        paths.append(
            generate_training_file(
                path,
                sim_params=sim_params,
                psf_params=psf_params,
                seed=file_seed,
                file_index=file_index,
            )
        )
        if progress:
            print(f"Saved {path} in {time.time() - start:.1f} seconds.")
    return paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate deterministic SPTnet training data.")
    parser.add_argument("--output-dir", default=None, help="Directory for generated HDF5 .mat files.")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic seed. Overrides SPT_SEED.")
    parser.add_argument("--num-files", type=int, default=None, help="Number of files to generate.")
    parser.add_argument("--videos-per-file", type=int, default=None, help="Videos per generated file.")
    parser.add_argument("--frames", type=int, default=None, help="Frames per video.")
    parser.add_argument("--image-dims", type=int, default=None, help="Square image size in pixels.")
    parser.add_argument("--p-num-min", type=int, default=None, help="Minimum particles per video.")
    parser.add_argument("--p-num-max", type=int, default=None, help="Maximum particles per video.")
    parser.add_argument("--file-start", type=int, default=None, help="First file index.")
    parser.add_argument("--photon-min", type=float, default=None, help="Minimum photons per particle.")
    parser.add_argument("--photon-max", type=float, default=None, help="Maximum photons per particle.")
    parser.add_argument("--bg-min", type=float, default=None, help="Minimum uniform background level.")
    parser.add_argument("--bg-max", type=float, default=None, help="Maximum uniform background level.")
    parser.add_argument("--perlin-bg-min", type=float, default=None, help="Minimum Perlin background multiplier.")
    parser.add_argument("--perlin-bg-max", type=float, default=None, help="Maximum Perlin background multiplier.")
    parser.add_argument("--motion-blur", action="store_true", help="Enable oversampled motion blur.")
    parser.add_argument("--no-progress", action="store_true", help="Suppress progress output.")
    return parser


def main(argv: list[str] | None = None) -> None:
    sim_params, psf_params, env_seed, output_dir = params_from_environment()
    parser = _build_parser()
    args = parser.parse_args(argv)
    updates = {
        "num_files": args.num_files,
        "videos_per_file": args.videos_per_file,
        "frames": args.frames,
        "image_dims": args.image_dims,
        "p_num_min": args.p_num_min,
        "p_num_max": args.p_num_max,
        "file_start": args.file_start,
    }
    sim_dict = asdict(sim_params)
    for key, value in updates.items():
        if value is not None:
            sim_dict[key] = value
    if args.motion_blur:
        sim_dict["motion_blur"] = True
    sim_params = SimulationParams(**sim_dict)

    psf_dict = asdict(psf_params)
    for key, value in {
        "photon_min": args.photon_min,
        "photon_max": args.photon_max,
        "bg_min": args.bg_min,
        "bg_max": args.bg_max,
        "perlin_bg_min": args.perlin_bg_min,
        "perlin_bg_max": args.perlin_bg_max,
    }.items():
        if value is not None:
            psf_dict[key] = value
    psf_params = PSFParams(**psf_dict)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    seed = env_seed if args.seed is None else args.seed

    print("SPTnet Python training-data generator starting.")
    print(f"Files: {sim_params.num_files}, videos/file: {sim_params.videos_per_file}, frames: {sim_params.frames}.")
    print(f"Image dims: {sim_params.image_dims}, seed: {'shuffle' if seed is None else seed}.")
    print(f"Saving to {output_dir}")
    generate_training_data(
        output_dir,
        sim_params=sim_params,
        psf_params=psf_params,
        seed=seed,
        progress=not args.no_progress,
    )
    print("Simulation Completed!")


if __name__ == "__main__":
    main()
