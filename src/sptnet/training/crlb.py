"""Generate CRLB matrices for SPTnet training."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


DEFAULT_DATASET_NAME = "CRLB_matrix_HD_frame"
DEFAULT_DIFF_STEP = 0.01
DEFAULT_CRLB_STEM = "CRLB_H_D_frame"
DEFAULT_CRLB_EXTENSION = ".h5"


def _diffusion_count(diff_max: float, diff_step: float) -> int:
    """Return the number of diffusion grid points, matching MATLAB's 1:D_step."""
    if diff_max <= 0:
        raise ValueError("diff_max must be positive.")
    if diff_step <= 0:
        raise ValueError("diff_step must be positive.")

    count = int(round(diff_max / diff_step))
    if not np.isclose(count * diff_step, diff_max, rtol=0.0, atol=1e-12):
        raise ValueError("diff_max must be an integer multiple of diff_step.")
    return count


def crlb_extension_for_training_data(training_files: Iterable[str | Path]) -> str:
    """Return the default CRLB extension for a set of training data files."""
    paths = [Path(path) for path in training_files]
    if paths and all(path.suffix.lower() == ".mat" for path in paths):
        return ".mat"
    return DEFAULT_CRLB_EXTENSION


def default_crlb_path_for_training_data(
    training_files: Iterable[str | Path],
    *,
    search_dirs: Iterable[str | Path] | None = None,
) -> Path:
    """Choose the default CRLB path from the training data file type.

    Native HDF5 training data defaults to ``CRLB_H_D_frame.h5``. MATLAB v7.3
    ``.mat`` training data keeps the legacy ``CRLB_H_D_frame.mat`` name.
    """
    extension = crlb_extension_for_training_data(training_files)
    if search_dirs is None:
        search_dirs = [
            Path.cwd(),
            Path(__file__).resolve().parents[3],
        ]

    candidates = [Path(directory) / f"{DEFAULT_CRLB_STEM}{extension}" for directory in search_dirs]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _crlb_for_frame(
    steps: int,
    diffusion_values: np.ndarray,
    *,
    jitter_scale: float,
) -> np.ndarray:
    """Compute CRLB values for one frame count.

    The returned array is shaped ``(2, 2, D, 99)``. The diffusion axis is
    vectorized by using the fact that the jittered covariance matrix scales
    linearly with the diffusion coefficient.
    """
    t = np.arange(1, steps + 1, dtype=np.float64)[:, None]
    s = np.arange(1, steps + 1, dtype=np.float64)[None, :]
    tau = np.abs(t - s)
    tau_mask = tau > 0
    h_values = 0.01 * np.arange(1, 100, dtype=np.float64)
    eye = np.eye(steps, dtype=np.float64)

    frame_crlb = np.zeros((2, 2, diffusion_values.size, h_values.size), dtype=np.float64)
    d = diffusion_values
    d_squared = d * d

    for h_idx, hurst in enumerate(h_values):
        t2h = t ** (2 * hurst)
        s2h = s ** (2 * hurst)
        tau2h = tau ** (2 * hurst)

        covariance_kernel = t2h + s2h - tau2h
        hurst_derivative = 2 * np.log(t) * t2h + 2 * np.log(s) * s2h
        hurst_derivative = np.array(hurst_derivative, copy=True)
        hurst_derivative[tau_mask] -= (
            2 * tau2h[tau_mask] * np.log(tau[tau_mask])
        )

        jitter = jitter_scale * np.max(np.diag(covariance_kernel))
        jittered_kernel = covariance_kernel + jitter * eye
        cholesky = np.linalg.cholesky(jittered_kernel)

        inv_kernel_hurst = np.linalg.solve(
            cholesky.T,
            np.linalg.solve(cholesky, hurst_derivative),
        )
        inv_kernel_diffusion = np.linalg.solve(
            cholesky.T,
            np.linalg.solve(cholesky, covariance_kernel),
        )

        fisher_hh = 0.5 * np.sum(inv_kernel_hurst * inv_kernel_hurst.T)
        fisher_hd_base = 0.5 * np.sum(inv_kernel_hurst * inv_kernel_diffusion.T)
        fisher_dd_base = 0.5 * np.sum(inv_kernel_diffusion * inv_kernel_diffusion.T)
        determinant_base = fisher_hh * fisher_dd_base - fisher_hd_base * fisher_hd_base

        frame_crlb[0, 0, :, h_idx] = fisher_dd_base / determinant_base
        frame_crlb[0, 1, :, h_idx] = -fisher_hd_base * d / determinant_base
        frame_crlb[1, 0, :, h_idx] = frame_crlb[0, 1, :, h_idx]
        frame_crlb[1, 1, :, h_idx] = fisher_hh * d_squared / determinant_base

    return frame_crlb


def compute_crlb_matrix(
    frame_number: int = 100,
    diff_max: float = 10.0,
    diff_step: float = 0.01,
    *,
    jitter_scale: float = 1e-8,
    progress: bool = False,
) -> np.ndarray:
    """Compute the CRLB matrix used by SPTnet training.

    Parameters
    ----------
    frame_number:
        Maximum number of movie frames. Index 0 is left as zeros to preserve
        the MATLAB convention where calculations start at frame count 2.
    diff_max:
        Maximum generalized diffusion coefficient on the grid.
    diff_step:
        Spacing between diffusion grid points.
    jitter_scale:
        Scale factor for the covariance diagonal jitter used before Cholesky.
    progress:
        Print a short progress line for each computed frame count.

    Returns
    -------
    np.ndarray
        CRLB matrix shaped ``(2, 2, D_step, 99, frame_number)``.
    """
    if frame_number < 2:
        raise ValueError("frame_number must be at least 2.")
    if jitter_scale < 0:
        raise ValueError("jitter_scale must be non-negative.")

    diffusion_count = _diffusion_count(diff_max, diff_step)
    diffusion_values = diff_step * np.arange(1, diffusion_count + 1, dtype=np.float64)
    crlb_matrix = np.zeros((2, 2, diffusion_count, 99, frame_number), dtype=np.float64)

    for steps in range(2, frame_number + 1):
        if progress:
            print(f"Computing CRLB for {steps}/{frame_number} frames")
        crlb_matrix[:, :, :, :, steps - 1] = _crlb_for_frame(
            steps,
            diffusion_values,
            jitter_scale=jitter_scale,
        )

    return crlb_matrix


def save_crlb_matrix(
    output_path: str | Path,
    crlb_matrix: np.ndarray,
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
) -> None:
    """Write a CRLB matrix to an HDF5 file readable by training code."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        dataset = handle.create_dataset(dataset_name, data=crlb_matrix, dtype="float64")
        dataset.attrs["MATLAB_class"] = np.bytes_("double")


def _expected_shape(frame_number: int, diff_max: float, diff_step: float) -> tuple[int, int, int, int, int]:
    return (2, 2, _diffusion_count(diff_max, diff_step), 99, frame_number)


def _sample_grid_indices(count: int) -> list[int]:
    return sorted({0, count // 2, count - 1})


def validate_crlb_matrix(
    crlb_matrix: np.ndarray,
    *,
    frame_number: int,
    diff_max: float,
    diff_step: float = DEFAULT_DIFF_STEP,
    jitter_scale: float = 1e-8,
    rtol: float = 1e-7,
    atol: float = 1e-7,
) -> None:
    """Validate that a CRLB matrix is usable for a training dataset."""
    required_shape = _expected_shape(frame_number, diff_max, diff_step)
    if len(crlb_matrix.shape) != 5:
        raise ValueError(f"Unexpected CRLB matrix shape: {crlb_matrix.shape}.")
    if crlb_matrix.shape[0:2] != (2, 2) or crlb_matrix.shape[3] != 99:
        raise ValueError(f"Unexpected CRLB matrix shape: {crlb_matrix.shape}.")
    if crlb_matrix.shape[2] < required_shape[2] or crlb_matrix.shape[4] < required_shape[4]:
        raise ValueError(
            f"CRLB matrix has shape {crlb_matrix.shape}, but training requires at least {required_shape}."
        )

    required_diffusion_count = required_shape[2]
    if not np.all(np.isfinite(crlb_matrix[:, :, :required_diffusion_count, :, :frame_number])):
        raise ValueError("CRLB matrix contains non-finite values in the training-required region.")
    if not np.allclose(crlb_matrix[:, :, :required_diffusion_count, :, 0], 0.0, rtol=rtol, atol=atol):
        raise ValueError("CRLB matrix frame index 0 does not match the expected all-zero sentinel.")

    diffusion_values = diff_step * np.arange(1, required_diffusion_count + 1, dtype=np.float64)
    frame_samples = sorted({2, max(2, frame_number // 2), frame_number})
    diffusion_samples = _sample_grid_indices(required_diffusion_count)
    hurst_samples = _sample_grid_indices(99)

    for steps in frame_samples:
        expected_frame = _crlb_for_frame(steps, diffusion_values, jitter_scale=jitter_scale)
        frame_index = steps - 1
        for diffusion_index in diffusion_samples:
            for hurst_index in hurst_samples:
                actual = crlb_matrix[:, :, diffusion_index, hurst_index, frame_index]
                expected = expected_frame[:, :, diffusion_index, hurst_index]
                if not np.allclose(actual, expected, rtol=rtol, atol=atol):
                    raise ValueError(
                        "CRLB matrix values do not match the current generator at "
                        f"frame={steps}, diffusion_index={diffusion_index}, hurst_index={hurst_index}."
                    )


def load_or_generate_crlb_matrix(
    crlb_path: str | Path,
    *,
    frame_number: int,
    diff_max: float,
    diff_step: float = DEFAULT_DIFF_STEP,
    jitter_scale: float = 1e-8,
    progress: bool = True,
) -> np.ndarray:
    """Load a CRLB matrix, generating it once if the file is missing."""
    crlb_path = Path(crlb_path)
    generation_reason = f"CRLB matrix not found at {crlb_path}."

    if crlb_path.is_file():
        try:
            with h5py.File(crlb_path, "r") as handle:
                if DEFAULT_DATASET_NAME not in handle:
                    raise ValueError(f"missing dataset {DEFAULT_DATASET_NAME!r}")
                crlb_matrix = handle[DEFAULT_DATASET_NAME][()]
            validate_crlb_matrix(
                crlb_matrix,
                frame_number=frame_number,
                diff_max=diff_max,
                diff_step=diff_step,
                jitter_scale=jitter_scale,
            )
        except (OSError, ValueError) as exc:
            generation_reason = (
                f"Existing CRLB matrix at {crlb_path} is not valid for this training data: {exc}"
            )
        else:
            print(f"Loading CRLB matrix from {crlb_path}")
            return crlb_matrix

    print(f"{generation_reason} Computing it once and saving it.")
    return generate_crlb_file(
        crlb_path,
        frame_number=frame_number,
        diff_max=diff_max,
        diff_step=diff_step,
        jitter_scale=jitter_scale,
        progress=progress,
    )


def plot_crlb_surfaces(
    crlb_matrix: np.ndarray,
    output_dir: str | Path,
    *,
    frames: Iterable[int] | None = None,
) -> None:
    """Save optional CRLB_H and CRLB_D surface plots."""
    matplotlib_config_dir = Path(tempfile.gettempdir()) / "sptnet-matplotlib"
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_number = crlb_matrix.shape[-1]
    diffusion_count = crlb_matrix.shape[2]
    if frames is None:
        upper_frame = min(frame_number, 60)
        frames = range(5, upper_frame + 1) if upper_frame >= 5 else range(2, frame_number + 1)
    selected_frames = [frame for frame in frames if 2 <= frame <= frame_number]

    x_grid, y_grid = np.meshgrid(
        np.arange(1, diffusion_count + 1),
        np.arange(1, 100),
    )
    plot_specs = [
        ("CRLB_H", 0, 0, "crlb_h_surface.png"),
        ("CRLB_D", 1, 1, "crlb_d_surface.png"),
    ]

    for z_label, row, col, filename in plot_specs:
        figure = plt.figure(figsize=(8, 8))
        axes = figure.add_subplot(111, projection="3d")
        colors = plt.cm.turbo(np.linspace(0, 1, max(len(selected_frames), 1)))
        for color, frame in zip(colors, selected_frames):
            z_grid = crlb_matrix[row, col, :, :, frame - 1].T
            axes.plot_surface(
                x_grid,
                y_grid,
                z_grid,
                color=color,
                edgecolor="k",
                linewidth=0.2,
                alpha=0.85,
            )
        axes.set_xlabel("Generalized diffusion coefficient index")
        axes.set_ylabel("Hurst exponent index")
        axes.set_zlabel(z_label)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=150)
        plt.close(figure)


def generate_crlb_file(
    output_path: str | Path = f"{DEFAULT_CRLB_STEM}{DEFAULT_CRLB_EXTENSION}",
    *,
    frame_number: int = 100,
    diff_max: float = 10.0,
    diff_step: float = 0.01,
    jitter_scale: float = 1e-8,
    progress: bool = False,
) -> np.ndarray:
    """Compute and save a CRLB matrix file.

    The matrix is returned so callers can inspect it without reopening the file.
    """
    crlb_matrix = compute_crlb_matrix(
        frame_number=frame_number,
        diff_max=diff_max,
        diff_step=diff_step,
        jitter_scale=jitter_scale,
        progress=progress,
    )
    save_crlb_matrix(output_path, crlb_matrix)
    return crlb_matrix


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the CRLB_H_D_frame matrix used for SPTnet training."
    )
    parser.add_argument(
        "-o",
        "--output",
        default=f"{DEFAULT_CRLB_STEM}{DEFAULT_CRLB_EXTENSION}",
        help=f"Output HDF5 path. Defaults to ./{DEFAULT_CRLB_STEM}{DEFAULT_CRLB_EXTENSION}.",
    )
    parser.add_argument(
        "--frame-number",
        type=int,
        default=100,
        help="Maximum number of frames to compute. Defaults to 100.",
    )
    parser.add_argument(
        "--diff-max",
        type=float,
        default=10.0,
        help="Maximum generalized diffusion coefficient. Defaults to 10.0.",
    )
    parser.add_argument(
        "--diff-step",
        type=float,
        default=0.01,
        help="Diffusion grid step. Defaults to 0.01.",
    )
    parser.add_argument(
        "--jitter-scale",
        type=float,
        default=1e-8,
        help="Diagonal jitter scale used before Cholesky. Defaults to 1e-8.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print progress while computing frame counts.",
    )
    parser.add_argument(
        "--plot-dir",
        default="",
        help="Optional directory for CRLB_H and CRLB_D surface PNGs.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    crlb_matrix = generate_crlb_file(
        args.output,
        frame_number=args.frame_number,
        diff_max=args.diff_max,
        diff_step=args.diff_step,
        jitter_scale=args.jitter_scale,
        progress=args.progress,
    )
    if args.plot_dir:
        plot_crlb_surfaces(crlb_matrix, args.plot_dir)
    print(f"Wrote CRLB matrix to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
