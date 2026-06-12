"""``sptnet-preprocess`` CLI: background-subtract / contrast-normalize movies.

Thin command-line wrapper around :mod:`sptnet.visualization.background`. It
reads raw movie TIFF(s), runs :func:`~sptnet.visualization.background.remove_background`,
and writes a processed TIFF that ``sptnet-segment`` consumes exactly like the raw
movie, so the tiling/inference pipeline is unchanged. See the ``background``
module docstring for the rationale (the ER cells sit close to the noise floor)
and the per-stage descriptions.

Example
-------
    sptnet-preprocess \
        "RealData/ER_SPT/240709_KDEL_cell3_uniPA-3.tif" \
        "RealData/ER_SPT/240723_sec61b_cell11_uniPA-3.tif" \
        --output-dir RealData/ER_SPT/preprocessed --qc-png

then tile + infer the processed movie as usual (``sptnet-segment`` -> inference
-> ``sptnet-stitch``).
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import tifffile

from sptnet.visualization.background import frame_snr, remove_background, stack_snr


def to_output_dtype(stack: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "float32":
        return stack.astype(np.float32, copy=False)
    if dtype == "uint16":
        return np.clip(stack * 65535.0, 0, 65535).astype(np.uint16)
    if dtype == "uint8":
        return np.clip(stack * 255.0, 0, 255).astype(np.uint8)
    raise ValueError(f"Unknown output dtype {dtype!r}.")


def save_qc_png(raw: np.ndarray, processed: np.ndarray, out_path: Path, frame: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = int(min(frame, raw.shape[0] - 1))
    rf = raw[frame].astype(np.float32)
    pf = processed[frame].astype(np.float32)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    axes[0].imshow(rf, cmap="gray", vmin=np.percentile(rf, 1), vmax=np.percentile(rf, 99.7))
    axes[0].set_title(f"raw (frame {frame}, SNR~{frame_snr(rf):.1f})")
    axes[1].imshow(pf, cmap="gray", vmin=np.percentile(pf, 1), vmax=np.percentile(pf, 99.7))
    axes[1].set_title(f"preprocessed (SNR~{frame_snr(pf):.1f})")
    for ax in axes:
        ax.axis("off")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def expand_inputs(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for item in patterns:
        matches = glob.glob(item) if glob.has_magic(item) else [item]
        paths.extend(Path(m) for m in matches)
    return sorted(set(paths))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sptnet-preprocess",
        description="Background subtraction + contrast normalization for dim real SPT movies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("inputs", nargs="+", help="Raw movie TIFF(s) or glob patterns.")
    p.add_argument("-o", "--output-dir", default=None,
                   help="Directory for processed TIFFs (default: alongside each input).")
    p.add_argument("--suffix", default="_preprocessed", help="Appended to each output stem.")
    p.add_argument("--dtype", choices=("uint16", "float32", "uint8"), default="uint16",
                   help="Output TIFF dtype. Inference min-max normalizes, so uint16 is fine.")

    p.add_argument("--temporal", choices=("median", "percentile", "none"), default="median",
                   help="Per-pixel temporal background estimate to subtract.")
    p.add_argument("--temporal-q", type=float, default=20.0,
                   help="Percentile for --temporal percentile (lower = more conservative background).")
    p.add_argument("--temporal-window", type=int, default=0,
                   help="Frames per temporal background block (0 = one global estimate).")

    p.add_argument("--spatial", choices=("highpass", "dog", "none"), default="dog",
                   help="Per-frame spatial background removal (dog band-pass scores best on ER_SPT).")
    p.add_argument("--sigma-small", type=float, default=1.0, help="Small Gaussian sigma (PSF scale) for --spatial dog.")
    p.add_argument("--sigma-large", type=float, default=7.0, help="Large Gaussian sigma (background scale).")

    p.add_argument("--normalize", choices=("global", "perframe", "none"), default="global",
                   help="Contrast normalization after background removal.")
    p.add_argument("--p-low", type=float, default=1.0, help="Lower percentile for clip/rescale.")
    p.add_argument("--p-high", type=float, default=99.9, help="Upper percentile for clip/rescale.")

    p.add_argument("--max-frames", type=int, default=None, help="Process only the first N frames (quick test).")
    p.add_argument("--qc-png", action="store_true", help="Write a raw-vs-processed comparison PNG per movie.")
    p.add_argument("--qc-frame", type=int, default=0, help="Frame index used for the QC PNG.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    inputs = expand_inputs(args.inputs)
    if not inputs:
        raise SystemExit("No input movies matched.")

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    for path in inputs:
        raw = np.asarray(tifffile.imread(str(path)))
        if raw.ndim != 3:
            print(f"Skipping {path}: expected a T,Y,X stack, got shape {raw.shape}.")
            continue
        if args.max_frames is not None:
            raw = raw[: args.max_frames]

        snr_before = stack_snr(raw)
        processed = remove_background(
            raw,
            temporal=args.temporal,
            temporal_q=args.temporal_q,
            temporal_window=args.temporal_window,
            spatial=args.spatial,
            sigma_small=args.sigma_small,
            sigma_large=args.sigma_large,
            normalize=args.normalize,
            p_low=args.p_low,
            p_high=args.p_high,
        )
        snr_after = stack_snr(processed)
        out_stack = to_output_dtype(processed, args.dtype)

        dest_dir = out_dir if out_dir is not None else path.parent
        out_path = dest_dir / f"{path.stem}{args.suffix}.tif"
        tifffile.imwrite(str(out_path), out_stack)
        print(
            f"{path.name}: {raw.shape} {raw.dtype} -> {out_path.name} {out_stack.dtype} "
            f"| particle SNR {snr_before:.1f} -> {snr_after:.1f}"
        )
        if args.qc_png:
            qc_path = dest_dir / f"{path.stem}{args.suffix}_qc.png"
            save_qc_png(raw, processed, qc_path, args.qc_frame)
            print(f"  wrote QC image {qc_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
