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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

    new = load_steady(args.new_dir, "train_seconds")
    old = load_steady(args.old_dir, "train_seconds")
    speedup = old.mean() / new.mean()

    def boot_ci(arr, n=10000, seed=0):
        rng = np.random.default_rng(seed)
        means = arr[rng.integers(0, arr.size, size=(n, arr.size))].mean(axis=1)
        return np.percentile(means, 2.5), np.percentile(means, 97.5)

    data = [old, new]
    means = [old.mean(), new.mean()]
    cis = [boot_ci(old), boot_ci(new)]
    yerr = [[m - lo for m, (lo, hi) in zip(means, cis)],
            [hi - m for m, (lo, hi) in zip(means, cis)]]
    labels = ["Original\nSPTnet", "opt-SPTnet"]
    colors = ["#780115", "#06527e"]
    x = [1, 2]

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    # Mean as bar height, with 95% bootstrap CI error bars.
    ax.bar(x, means, width=0.6, color=colors, alpha=0.85, edgecolor=colors,
           linewidth=1.2, yerr=yerr, capsize=5,
           error_kw=dict(ecolor="0.25", lw=1.2), zorder=2)
    # Individual steady-state epochs overlaid as a jittered scatter.
    for xi, arr, c in zip(x, data, colors):
        jitter = (np.random.default_rng(xi).random(arr.size) - 0.5) * 0.28
        ax.scatter(np.full(arr.size, xi) + jitter, arr, s=16, color=c,
                   alpha=0.85, zorder=3, edgecolor="white", linewidth=0.4)
    # Mean value labels, placed just above each group's highest scatter point so
    # they never overlap the points.
    for xi, m, arr in zip(x, means, data):
        ax.text(xi, arr.max() + max(old) * 0.025, f"{m:.0f}s", ha="center",
                va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Training time per epoch (s)")
    ax.set_ylim(0, max(old) * 1.18)
    ax.set_title("Per-epoch training time (8 runs, steady-state)")
    ax.annotate(f"{speedup:.2f}× faster", xy=(1.5, max(old) * 1.10),
                ha="center", va="center", fontsize=12, fontweight="bold")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"new mean {new.mean():.2f}s/epoch, old mean {old.mean():.2f}s/epoch, "
          f"speedup {speedup:.2f}x")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
