#!/usr/bin/env python
"""Generate one physics-faithful spatial-diffusion visual movie.

This is a standalone thesis/demo script. It does not modify or replace the
training-data generator, but it deliberately reuses the same core simulation
components: fractional Brownian motion, Zernike PSF rendering, OTF rescaling,
background, and Poisson camera noise.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.patches import Circle

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sptnet.training.data_generator import (  # noqa: E402
    PSFParams,
    ZernikePSF,
    apply_otf_rescale,
    fractional_brownian_motion_2d,
    make_otf_rescale_kernel,
    perlin_noise,
)


def _inside_circle(y: np.ndarray | float, x: np.ndarray | float, center_y: float, center_x: float, radius: float):
    return (np.asarray(y) - center_y) ** 2 + (np.asarray(x) - center_x) ** 2 <= radius**2


def _sample_inside_circle(rng: np.random.Generator, center_y: float, center_x: float, radius: float) -> tuple[float, float]:
    angle = rng.uniform(0.0, 2.0 * np.pi)
    distance = radius * np.sqrt(rng.uniform(0.0, 0.75))
    return center_y + distance * np.sin(angle), center_x + distance * np.cos(angle)


def _sample_outside_circle(
    rng: np.random.Generator,
    *,
    size: int,
    center_y: float,
    center_x: float,
    radius: float,
    margin: float,
) -> tuple[float, float]:
    for _ in range(10_000):
        y = rng.uniform(margin, size - margin)
        x = rng.uniform(margin, size - margin)
        if not _inside_circle(y, x, center_y, center_x, radius + margin):
            return y, x
    raise RuntimeError("Could not place a low-diffusion particle outside the high-D region.")


def diffusion_map(size: int, low_d: float, high_d: float, center_y: float, center_x: float, radius: float) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    return np.where(_inside_circle(yy, xx, center_y, center_x, radius), high_d, low_d).astype(np.float32)


def build_tracks(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic fBM tracks with per-particle diffusion labels.

    Positions returned to the renderer are in x/y coordinates relative to image
    centre, which is exactly what ``ZernikePSF.generate`` expects.
    """
    rng = np.random.default_rng(args.seed)
    particle_count = args.high_particles + args.low_particles
    positions_xy = np.full((args.frames, particle_count, 2), np.nan, dtype=np.float64)
    positions_yx_pixels = np.full((args.frames, particle_count, 2), np.nan, dtype=np.float32)
    diffusion = np.zeros(particle_count, dtype=np.float32)
    region = np.zeros(particle_count, dtype=np.int8)

    for idx in range(particle_count):
        is_high = idx < args.high_particles
        if is_high:
            y0, x0 = _sample_inside_circle(
                rng,
                args.high_center_y,
                args.high_center_x,
                args.high_radius,
            )
            particle_diffusion = args.high_d
            region[idx] = 1
        else:
            y0, x0 = _sample_outside_circle(
                rng,
                size=args.size,
                center_y=args.high_center_y,
                center_x=args.high_center_x,
                radius=args.high_radius,
                margin=args.margin,
            )
            particle_diffusion = args.low_d

        hurst = args.hurst
        traj_x, traj_y = fractional_brownian_motion_2d(hurst, args.frames, particle_diffusion, rng)
        x_centered = traj_x + (x0 - args.size / 2.0)
        y_centered = traj_y + (y0 - args.size / 2.0)

        positions_xy[:, idx, 0] = x_centered
        positions_xy[:, idx, 1] = y_centered
        positions_yx_pixels[:, idx, 0] = y_centered + args.size / 2.0
        positions_yx_pixels[:, idx, 1] = x_centered + args.size / 2.0
        diffusion[idx] = particle_diffusion

    return positions_xy, positions_yx_pixels, diffusion, region


def render_physics_movie(args: argparse.Namespace, positions_xy: np.ndarray) -> np.ndarray:
    """Render tracks using the same PSF, OTF, background, and Poisson path."""
    if args.psf_size < args.size:
        raise ValueError("--psf-size must be >= --size because the PSF renderer crops from the pupil image.")

    rng = np.random.default_rng(args.seed + 1)
    psf_params = PSFParams(
        psf_size=args.psf_size,
        photon_min=args.photons,
        photon_max=args.photons,
        bg_min=args.background,
        bg_max=args.background,
        perlin_bg_min=args.perlin_background,
        perlin_bg_max=args.perlin_background,
    )
    psf_model = ZernikePSF(psf_params, args.size)
    otf_kernel = make_otf_rescale_kernel(
        args.size,
        psf_params.pixel_size,
        psf_params.otf_sigma_x,
        psf_params.otf_sigma_y,
    )

    movie_signal = np.zeros((args.frames, args.size, args.size), dtype=np.float64)
    zpos = np.zeros(args.frames, dtype=np.float64)

    for particle_idx in range(positions_xy.shape[1]):
        particle_psf = psf_model.generate(
            positions_xy[:, particle_idx, 0],
            positions_xy[:, particle_idx, 1],
            zpos,
        )
        particle_psf = apply_otf_rescale(particle_psf, otf_kernel)
        movie_signal += particle_psf / psf_model.norm_parameter * args.photons
        print(f"Rendered particle {particle_idx + 1}/{positions_xy.shape[1]}")

    background = args.background + args.perlin_background * perlin_noise(args.size, rng)
    lam = np.maximum(movie_signal + background[None, :, :], 0.0)
    return rng.poisson(lam).astype(np.float32)


def save_preview(
    path: Path,
    *,
    movie: np.ndarray,
    positions_yx_pixels: np.ndarray,
    region: np.ndarray,
    d_map: np.ndarray,
    args: argparse.Namespace,
) -> None:
    max_projection = np.max(movie, axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    axes[0].imshow(max_projection, cmap="gray")
    axes[0].add_patch(
        Circle(
            (args.high_center_x, args.high_center_y),
            args.high_radius,
            fill=False,
            edgecolor="tab:red",
            linewidth=2.0,
        )
    )
    for idx in range(positions_yx_pixels.shape[1]):
        color = "tab:red" if region[idx] else "tab:cyan"
        alpha = 0.75 if region[idx] else 0.45
        axes[0].plot(
            positions_yx_pixels[:, idx, 1],
            positions_yx_pixels[:, idx, 0],
            color=color,
            linewidth=0.6,
            alpha=alpha,
        )
    axes[0].set_title("SPTnet-style simulated movie")
    axes[0].set_axis_off()

    image = axes[1].imshow(d_map, cmap="magma")
    axes[1].set_title("Assigned diffusion regions")
    axes[1].set_axis_off()
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04, label="D")

    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_outputs(
    args: argparse.Namespace,
    *,
    movie: np.ndarray,
    positions_xy: np.ndarray,
    positions_yx_pixels: np.ndarray,
    diffusion: np.ndarray,
    region: np.ndarray,
    d_map: np.ndarray,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = args.output_dir / f"{args.basename}.h5"
    tif_path = args.output_dir / f"{args.basename}.tif"
    preview_path = args.output_dir / f"{args.basename}_preview.png"

    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("timelapsedata", data=movie, compression="gzip")
        handle.create_dataset("positions_xy_centered", data=positions_xy.astype(np.float32), compression="gzip")
        handle.create_dataset("positions_yx_pixels", data=positions_yx_pixels, compression="gzip")
        handle.create_dataset("particle_diffusion", data=diffusion)
        handle.create_dataset("particle_region", data=region)
        handle.create_dataset("diffusion_map", data=d_map, compression="gzip")
        handle.attrs["description"] = "Physics-faithful spatial diffusion visual using SPTnet generator components."
        for key, value in vars(args).items():
            handle.attrs[key] = str(value) if isinstance(value, Path) else value

    tifffile.imwrite(tif_path, movie, imagej=True)
    save_preview(
        preview_path,
        movie=movie,
        positions_yx_pixels=positions_yx_pixels,
        region=region,
        d_map=d_map,
        args=args,
    )

    print(f"Wrote {h5_path}")
    print(f"Wrote {tif_path}")
    print(f"Wrote {preview_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a physics-faithful spatial-diffusion visual movie.")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/spatial_diffusion_visual"))
    parser.add_argument("--basename", default="spatial_diffusion_visual")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--psf-size", type=int, default=512)
    parser.add_argument("--high-particles", type=int, default=28)
    parser.add_argument("--low-particles", type=int, default=44)
    parser.add_argument("--hurst", type=float, default=0.5)
    parser.add_argument("--low-d", type=float, default=0.15)
    parser.add_argument("--high-d", type=float, default=0.45)
    parser.add_argument("--high-center-y", type=float, default=256.0)
    parser.add_argument("--high-center-x", type=float, default=340.0)
    parser.add_argument("--high-radius", type=float, default=115.0)
    parser.add_argument("--margin", type=float, default=24.0)
    parser.add_argument("--photons", type=float, default=1800.0)
    parser.add_argument("--background", type=float, default=5.0)
    parser.add_argument("--perlin-background", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=1458)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    positions_xy, positions_yx_pixels, diffusion, region = build_tracks(args)
    movie = render_physics_movie(args, positions_xy)
    d_map = diffusion_map(
        args.size,
        args.low_d,
        args.high_d,
        args.high_center_y,
        args.high_center_x,
        args.high_radius,
    )
    write_outputs(
        args,
        movie=movie,
        positions_xy=positions_xy,
        positions_yx_pixels=positions_yx_pixels,
        diffusion=diffusion,
        region=region,
        d_map=d_map,
    )


if __name__ == "__main__":
    main()
