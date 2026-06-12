"""Background subtraction and contrast enhancement for dim real SPT movies.

The ER single-molecule movies in ``RealData/ER_SPT`` (e.g. the KDEL and sec61b
cells) are acquired close to the noise floor: in a typical 64x64 tile-clip the
raw counts span only ~83-145, so a particle sits barely ~13 counts above a
background of ~100. SPTnet was trained/finetuned on bright, high-contrast
synthetic particles, so its detector works on bright real movies but fires on
background texture here, and overlaying predictions on the raw movie is hard to
read. These helpers lift the dim particles out of the noise floor.

The processing is a configurable three-stage pipeline (each stage optional):

1. Temporal background subtraction -- removes the static/slowly-varying part of
   each pixel by subtracting a per-pixel temporal percentile. With sparse moving
   particles the per-pixel median (or a low percentile) over time estimates the
   pixel value with no particle on it.
2. Spatial background removal -- flattens residual large-scale shading per frame,
   either a high-pass (subtract a large-sigma Gaussian; spots stay natural) or a
   difference-of-Gaussians band-pass (also denoises at the PSF scale).
3. Contrast normalization -- robust percentile clip + rescale so the now
   background-free particles use the full output range.

The output array is the same ``T,H,W`` shape as the input and can be fed
straight into the segmentation/inference pipeline or the result visualizers.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


__all__ = [
    "subtract_temporal_background",
    "remove_spatial_background",
    "normalize_contrast",
    "remove_background",
    "frame_snr",
    "stack_snr",
]


# ─── Quality metric ──────────────────────────────────────────────────────────
def frame_snr(frame: np.ndarray) -> float:
    """Rough particle contrast-to-noise for one frame.

    The background level and noise (robust MAD) are estimated from non-zero
    pixels only, so the metric stays meaningful after background subtraction
    clips most of the frame to exactly zero (a plain MAD would be 0 there and
    blow the ratio up). The brightest pixels (99.9th percentile) stand in for
    particle peaks. Higher means particles stand out more from the noise.
    """
    frame = np.asarray(frame, dtype=np.float32)
    background = frame[frame > 0]
    if background.size < 16:
        background = frame.ravel()
    med = float(np.median(background))
    mad = float(np.median(np.abs(background - med)))
    sigma = max(1.4826 * mad, 1e-6 * (float(frame.max()) - med) + 1e-9)
    peak = float(np.percentile(frame, 99.9))
    return (peak - med) / sigma


def stack_snr(stack: np.ndarray, sample: int = 25) -> float:
    """Mean :func:`frame_snr` over a subsample of frames."""
    stack = np.asarray(stack)
    idx = np.linspace(0, stack.shape[0] - 1, num=min(sample, stack.shape[0])).astype(int)
    return float(np.mean([frame_snr(stack[i]) for i in idx]))


# ─── Stage 1: temporal background ────────────────────────────────────────────
def _percentile_over_time(block: np.ndarray, q: float) -> np.ndarray:
    """Per-pixel temporal percentile over axis 0, ``(T,h,w) -> (h,w)``."""
    if q == 50.0:
        return np.median(block, axis=0)
    return np.percentile(block, q, axis=0)


def subtract_temporal_background(
    stack: np.ndarray,
    *,
    q: float = 50.0,
    window: int = 0,
    row_chunk: int = 48,
    clip: bool = True,
    copy: bool = True,
) -> np.ndarray:
    """Subtract a per-pixel temporal background.

    ``window == 0`` uses one global per-pixel percentile (fast, best when the
    background is static). ``window > 0`` estimates the background separately on
    consecutive blocks of ``window`` frames (piecewise constant in time), which
    tracks slow drift / dynamic background at modest cost. Negative residuals are
    clipped to zero when ``clip``. Rows are processed in chunks to bound peak
    memory. With ``copy=False`` the input array is modified in place.
    """
    stack = np.array(stack, dtype=np.float32, copy=copy)
    T, H, _ = stack.shape
    if window and window > 0:
        for t0 in range(0, T, window):
            t1 = min(T, t0 + window)
            stack[t0:t1] -= _percentile_over_time(stack[t0:t1], q)
    else:
        for r0 in range(0, H, row_chunk):
            r1 = min(H, r0 + row_chunk)
            stack[:, r0:r1, :] -= _percentile_over_time(stack[:, r0:r1, :], q)
    if clip:
        np.clip(stack, 0, None, out=stack)
    return stack


# ─── Stage 2: spatial background ─────────────────────────────────────────────
def remove_spatial_background(
    stack: np.ndarray,
    *,
    mode: str = "highpass",
    sigma_small: float = 1.0,
    sigma_large: float = 7.0,
    clip: bool = True,
    copy: bool = True,
) -> np.ndarray:
    """Flatten large-scale shading per frame.

    ``highpass``: ``frame - gaussian(frame, sigma_large)`` -- keeps particles as
    natural positive spots. ``dog``: ``gaussian(frame, sigma_small) -
    gaussian(frame, sigma_large)`` -- additionally smooths pixel noise at the PSF
    scale. Negative residuals are clipped to zero when ``clip``.
    """
    if mode not in {"highpass", "dog"}:
        raise ValueError(f"spatial mode must be 'highpass' or 'dog', got {mode!r}.")
    stack = np.array(stack, dtype=np.float32, copy=copy)
    for t in range(stack.shape[0]):
        frame = stack[t]
        large = gaussian_filter(frame, sigma_large)
        if mode == "dog":
            small = gaussian_filter(frame, sigma_small) if sigma_small > 0 else frame
            stack[t] = small - large
        else:
            stack[t] = frame - large
    if clip:
        np.clip(stack, 0, None, out=stack)
    return stack


# ─── Stage 3: contrast normalization ─────────────────────────────────────────
def normalize_contrast(
    stack: np.ndarray,
    *,
    mode: str = "global",
    p_low: float = 1.0,
    p_high: float = 99.9,
    sample_frames: int = 200,
    copy: bool = True,
) -> np.ndarray:
    """Robust percentile clip + rescale to ``[0, 1]``.

    ``global`` uses one pair of percentiles for the whole movie (preserves
    relative brightness between frames, recommended for tracking). ``perframe``
    rescales each frame independently. ``none`` leaves values unchanged. Global
    percentiles are estimated from a frame subsample to keep the sort cheap.
    """
    if mode == "none":
        return np.asarray(stack, dtype=np.float32)
    stack = np.array(stack, dtype=np.float32, copy=copy)
    if mode == "global":
        idx = np.linspace(0, stack.shape[0] - 1, num=min(sample_frames, stack.shape[0])).astype(int)
        lo, hi = np.percentile(stack[idx], [p_low, p_high])
        scale = float(hi - lo) if hi > lo else 1.0
        stack -= lo
        stack /= scale
        np.clip(stack, 0.0, 1.0, out=stack)
    elif mode == "perframe":
        for t in range(stack.shape[0]):
            frame = stack[t]
            lo, hi = np.percentile(frame, [p_low, p_high])
            scale = float(hi - lo) if hi > lo else 1.0
            stack[t] = np.clip((frame - lo) / scale, 0.0, 1.0)
    else:
        raise ValueError(f"normalize mode must be 'global', 'perframe' or 'none', got {mode!r}.")
    return stack


# ─── Combined pipeline ───────────────────────────────────────────────────────
def remove_background(
    stack: np.ndarray,
    *,
    temporal: str = "median",
    temporal_q: float = 20.0,
    temporal_window: int = 0,
    spatial: str = "dog",
    sigma_small: float = 1.0,
    sigma_large: float = 7.0,
    normalize: str = "global",
    p_low: float = 1.0,
    p_high: float = 99.9,
) -> np.ndarray:
    """Run the configured background-removal pipeline on a ``T,H,W`` movie.

    Returns a new float32 stack in ``[0, 1]`` (unless ``normalize="none"``).

    On a synthetic-dim-spot injection test over the ``ER_SPT`` KDEL and sec61b
    cells (recover faint Gaussian spots planted on the real background; higher
    spot/background d-prime is better), the ``spatial dog`` band-pass is by far
    the dominant stage -- it raises detectability ~20-50% over the raw movie,
    while temporal/high-pass stages alone barely move it. The default therefore
    keeps ``spatial dog``; ``temporal median`` is retained as an optional
    robustness stage that removes bright *static* structures (tubules/puncta)
    that DoG's large sigma leaves behind, at a small detectability cost. Use
    ``temporal="none"`` for the pure best-detectability setting.

    Parameters
    ----------
    temporal:
        ``"median"``, ``"percentile"`` (uses ``temporal_q``) or ``"none"``.
    spatial:
        ``"highpass"``, ``"dog"`` or ``"none"``.
    normalize:
        ``"global"``, ``"perframe"`` or ``"none"``.
    """
    out = np.asarray(stack, dtype=np.float32)
    first = True
    if temporal != "none":
        q = 50.0 if temporal == "median" else temporal_q
        out = subtract_temporal_background(out, q=q, window=temporal_window, copy=first)
        first = False
    if spatial != "none":
        out = remove_spatial_background(
            out, mode=spatial, sigma_small=sigma_small, sigma_large=sigma_large, copy=first
        )
        first = False
    out = normalize_contrast(out, mode=normalize, p_low=p_low, p_high=p_high, copy=first)
    return out
