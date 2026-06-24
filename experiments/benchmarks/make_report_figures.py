#!/usr/bin/env python
"""Generate the report figure for the runtime benchmark.

Reads the per-epoch timing CSVs from the two headline configs, drops epoch 1 as
warm-up, and writes a clean per-epoch training-time comparison figure to
`report/figures/`. Reproducible from the committed `epoch_timing.csv` files.
"""

import argparse
import csv
import glob
import os

import numpy as np

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "notebooks"))
from plot_style import COLORS, apply_house_style  # noqa: E402


def load_steady(config_dir, metric, warmup=1):
    """Return a flat array of steady-state per-epoch values for one config."""
    vals = []
    for path in sorted(glob.glob(os.path.join(config_dir, "run_*", "epoch_timing.csv"))):
        rows = sorted(
            (r for r in csv.DictReader(open(path))),
            key=lambda r: int(float(r["epoch"])),
        )
        vals.extend(float(r[metric]) for r in rows[warmup:])
    return np.asarray(vals, dtype=float)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-dir", default=os.path.join(here, "new_fresh_good_runs"))
    ap.add_argument("--old-dir", default=os.path.join(here, "old_good_8_runs"))
    ap.add_argument("--out", default=os.path.join(repo, "report", "figures", "benchmark_per_epoch_train.pdf"))
    args = ap.parse_args()

    apply_house_style()

    new = load_steady(args.new_dir, "train_seconds")
    old = load_steady(args.old_dir, "train_seconds")
    speedup = old.mean() / new.mean()

    def boot_ci(arr, n=10000, seed=0):
        gen = np.random.default_rng(seed)
        m = arr[gen.integers(0, arr.size, size=(n, arr.size))].mean(axis=1)
        return np.percentile(m, 2.5), np.percentile(m, 97.5)

    # Mean bar per implementation with a 95% bootstrap CI, the individual
    # steady-state epochs overlaid as a jittered strip (so the bar is not a flat
    # block), and the mean labelled above. The title states the speed-up.
    rng = np.random.default_rng(0)
    configs = [("Original\nSPTnet", old, COLORS["baseline"]),
               ("opt-SPTnet", new, COLORS["opt"])]
    x = [1, 2]
    means = [arr.mean() for _, arr, _ in configs]
    cis = [boot_ci(arr) for _, arr, _ in configs]
    yerr = [[m - lo for m, (lo, _) in zip(means, cis)],
            [hi - m for m, (_, hi) in zip(means, cis)]]
    colours = [c for _, _, c in configs]

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.bar(x, means, width=0.58, color=colours, alpha=0.9, edgecolor="black",
           linewidth=0.7, yerr=yerr, capsize=4,
           error_kw=dict(ecolor="0.25", lw=1.1), zorder=2)
    for xi, (_, arr, colour) in zip(x, configs):
        jitter = (rng.random(arr.size) - 0.5) * 0.26
        ax.scatter(np.full(arr.size, xi) + jitter, arr, s=12, color=colour,
                   edgecolor="white", linewidth=0.4, alpha=0.95, zorder=3)
    for xi, m, (_, arr, colour) in zip(x, means, configs):
        ax.text(xi, arr.max() + max(old) * 0.03, rf"\textbf{{{m:.0f}\,s}}",
                ha="center", va="bottom", fontsize=13, color=colour)

    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in configs])
    ax.set_xlim(0.4, 2.6)
    ax.set_ylim(0, max(old) * 1.15)
    ax.set_ylabel("Training time per epoch (s)")
    ax.set_title(rf"Per-epoch training time: {speedup:.2f}$\times$ faster")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"new mean {new.mean():.2f}s/epoch, old mean {old.mean():.2f}s/epoch, "
          f"speedup {speedup:.2f}x")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
