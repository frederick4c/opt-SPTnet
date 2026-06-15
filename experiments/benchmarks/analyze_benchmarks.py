#!/usr/bin/env python
"""Bootstrap analysis of SPTnet training-speed benchmarks.

Consumes the per-epoch ``epoch_timing.csv`` files written by the training
scripts (one per run, laid out as ``<root>/<config>/run_NN/epoch_timing.csv``)
and produces:

  * per-config steady-state per-epoch training time with a 95% bootstrap CI,
  * the headline speedup ratio (old / new) with a 95% bootstrap CI,
  * a decomposition table (each disabled-optimization config vs the full new
    config) reported as a % per-epoch slowdown with a CI,
  * a Markdown summary table and a per-config distribution plot.

The bootstrap is *hierarchical*: it resamples runs with replacement and then
epochs within each resampled run, so both between-run and within-run variance
propagate into every interval. Epoch 1 of each run is dropped as warmup
(dataloader spin-up, cudnn autotuning, CUDA allocator warmup).

Only numpy + matplotlib (Agg) + the stdlib are required.
"""

import argparse
import csv
import glob
import os

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # pragma: no cover - plotting is optional
    _HAVE_MPL = False


def _read_timing_csv(path, metric):
    """Return a list of (epoch, value) row dicts from one epoch_timing.csv.

    `metric` selects the timed quantity: ``train_seconds`` (compute only) or
    ``epoch_total_seconds`` (full epoch incl. plotting/logging/checkpointing).
    Falls back to train_seconds if the requested column is absent (older CSVs).
    """
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                value = row.get(metric)
                if value in (None, ""):
                    value = row["train_seconds"]
                rows.append({"epoch": int(float(row["epoch"])), "value": float(value)})
            except (KeyError, ValueError):
                continue
    rows.sort(key=lambda r: r["epoch"])
    return rows


def discover(results_root, warmup_epochs, metric):
    """Map config -> list of per-run steady-state train_seconds arrays.

    Config is the directory two levels above epoch_timing.csv (the run-name
    folder); the run is the immediate parent. Runs that have no steady-state
    epochs after dropping warmup are skipped with a warning.
    """
    paths = sorted(glob.glob(os.path.join(results_root, "**", "epoch_timing.csv"), recursive=True))
    if not paths:
        raise SystemExit(f"No epoch_timing.csv files found under {results_root!r}.")

    configs = {}
    for path in paths:
        run_dir = os.path.dirname(path)
        config = os.path.basename(os.path.dirname(run_dir)) or os.path.basename(run_dir)
        rows = _read_timing_csv(path, metric)
        steady = [r["value"] for r in rows[warmup_epochs:]]
        if not steady:
            print(f"  (skip) {path}: no epochs left after dropping {warmup_epochs} warmup epoch(s)")
            continue
        configs.setdefault(config, []).append(np.asarray(steady, dtype=float))
    return configs


def hierarchical_bootstrap_means(runs, n_boot, rng):
    """Bootstrap distribution of the grand mean for one config.

    ``runs`` is a list of 1-D arrays (one per run). Each iteration resamples
    runs with replacement, then epochs within each chosen run with replacement,
    and records the mean of the pooled resample.
    """
    n_runs = len(runs)
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        chosen = rng.integers(0, n_runs, size=n_runs)
        pooled = []
        for idx in chosen:
            arr = runs[idx]
            pooled.append(arr[rng.integers(0, len(arr), size=len(arr))])
        means[b] = np.concatenate(pooled).mean()
    return means


def ci(samples, lo=2.5, hi=97.5):
    return float(np.percentile(samples, lo)), float(np.percentile(samples, hi))


def point_mean(runs):
    return float(np.concatenate(runs).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default=os.path.dirname(os.path.abspath(__file__)),
                    help="directory tree containing <config>/run_NN/epoch_timing.csv")
    ap.add_argument("--new-config", default="new_full", help="config name of the fully-optimized new system")
    ap.add_argument("--old-config", default="old", help="config name of the original baseline")
    ap.add_argument("--old-noplot-config", default="old_no_plot",
                    help="config name of the original baseline with plotting disabled")
    ap.add_argument("--metric", default="train_seconds",
                    choices=["train_seconds", "epoch_total_seconds"],
                    help="train_seconds = compute only (AMP/TF32/cudnn); "
                         "epoch_total_seconds = full epoch incl. plotting (use for old_no_plot)")
    ap.add_argument("--warmup-epochs", type=int, default=1, help="epochs dropped per run as warmup")
    ap.add_argument("--bootstrap", type=int, default=10000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None, help="defaults to <results-root>/results")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.results_root, "results")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Scanning {args.results_root} (metric={args.metric}; "
          f"dropping {args.warmup_epochs} warmup epoch(s) per run)...")
    configs = discover(args.results_root, args.warmup_epochs, args.metric)

    # Per-config bootstrap of the mean per-epoch train time.
    boot = {}
    summary_rows = []
    for config, runs in sorted(configs.items()):
        means = hierarchical_bootstrap_means(runs, args.bootstrap, rng)
        boot[config] = means
        pooled = np.concatenate(runs)
        lo, hi = ci(means)
        summary_rows.append(
            {
                "config": config,
                "n_runs": len(runs),
                "n_epochs": int(pooled.size),
                "mean_train_s": point_mean(runs),
                "median_train_s": float(np.median(pooled)),
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
        print(f"  {config}: {len(runs)} runs, {pooled.size} steady epochs, "
              f"mean {point_mean(runs):.2f}s/epoch [{lo:.2f}, {hi:.2f}]")

    # Write per-config summary CSV (metric-suffixed so train vs total don't clash).
    summary_csv = os.path.join(out_dir, f"summary_{args.metric}.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    md = [f"# SPTnet training-speed benchmark ({args.metric})", "",
          f"Steady-state per-epoch `{args.metric}` (warmup epochs dropped: {args.warmup_epochs}; "
          f"{args.bootstrap} hierarchical bootstrap resamples). "
          f"train_seconds = compute only; epoch_total_seconds = full epoch incl. plotting.", "",
          "| Config | Runs | Epochs | Mean s/epoch | 95% CI |",
          "|---|---|---|---|---|"]
    for r in summary_rows:
        md.append(f"| {r['config']} | {r['n_runs']} | {r['n_epochs']} | "
                  f"{r['mean_train_s']:.2f} | [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}] |")

    # Headline speedup old / new (independent resamples already in `boot`).
    have_headline = args.old_config in boot and args.new_config in boot
    if have_headline:
        ratio = boot[args.old_config] / boot[args.new_config]
        r_lo, r_hi = ci(ratio)
        point = point_mean(configs[args.old_config]) / point_mean(configs[args.new_config])
        md += ["", "## Headline speedup",
               f"**{args.old_config} / {args.new_config} = {point:.2f}x** "
               f"(95% CI [{r_lo:.2f}, {r_hi:.2f}])."]
        print(f"\nHEADLINE speedup {args.old_config}/{args.new_config}: "
              f"{point:.2f}x  95% CI [{r_lo:.2f}, {r_hi:.2f}]")
    else:
        md += ["", "## Headline speedup",
               f"_Skipped: need both '{args.old_config}' and '{args.new_config}' configs._"]
        print(f"\nHEADLINE skipped: missing '{args.old_config}' and/or '{args.new_config}'.")

    # Plotting contribution: old vs old_no_plot on the chosen metric (only
    # meaningful on epoch_total_seconds, since train_seconds excludes plotting).
    if args.old_config in boot and args.old_noplot_config in boot:
        rel = boot[args.old_config] / boot[args.old_noplot_config]
        lo, hi = ci(rel)
        point = point_mean(configs[args.old_config]) / point_mean(configs[args.old_noplot_config])
        md += ["", "## Plotting contribution (old baseline)",
               f"`{args.old_config}` / `{args.old_noplot_config}` on `{args.metric}` = "
               f"**{point:.3f}x** (95% CI [{lo:.3f}, {hi:.3f}]); i.e. per-epoch plotting "
               f"adds {(point - 1) * 100:+.1f}% to the old baseline. "
               f"Only meaningful on `epoch_total_seconds`."]
        print(f"\nPLOTTING old/{args.old_noplot_config} on {args.metric}: "
              f"{point:.3f}x ({(point - 1) * 100:+.1f}%) 95% CI [{lo:.3f}, {hi:.3f}]")

    # Decomposition: each disabled-optimization NEW config vs full-new. Excludes
    # the old baselines (old, old_no_plot), which are not new-system variants.
    decomp = [c for c in sorted(boot)
              if c not in (args.old_config, args.new_config, args.old_noplot_config)]
    if decomp and args.new_config in boot:
        md += ["", "## Decomposition (marginal, non-additive)",
               "Per-epoch slowdown when one optimization is disabled, relative to "
               f"`{args.new_config}`. Effects are marginal and need not sum to the total.", "",
               "| Disabled config | Mean s/epoch | Slowdown vs full | 95% CI |",
               "|---|---|---|---|"]
        for config in decomp:
            rel = boot[config] / boot[args.new_config]
            lo, hi = ci(rel)
            point = point_mean(configs[config]) / point_mean(configs[args.new_config])
            md.append(f"| {config} | {point_mean(configs[config]):.2f} | "
                      f"{(point - 1) * 100:+.1f}% | [{(lo - 1) * 100:+.1f}%, {(hi - 1) * 100:+.1f}%] |")
            print(f"  decomp {config}: {point:.2f}x full ({(point - 1) * 100:+.1f}%) "
                  f"95% CI [{(lo - 1) * 100:+.1f}%, {(hi - 1) * 100:+.1f}%]")

    md_path = os.path.join(out_dir, f"summary_{args.metric}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")

    # Distribution plot.
    plot_path = None
    if _HAVE_MPL:
        labels = [r["config"] for r in summary_rows]
        data = [np.concatenate(configs[c]) for c in labels]
        fig, axp = plt.subplots(figsize=(max(6, 1.4 * len(labels)), 5))
        positions = np.arange(1, len(labels) + 1)
        axp.boxplot(data, positions=positions, showfliers=False)
        for i, arr in zip(positions, data):
            jitter = (np.random.default_rng(int(i)).random(arr.size) - 0.5) * 0.25
            axp.scatter(np.full(arr.size, i) + jitter, arr, s=10, alpha=0.5, color="tab:blue")
        axp.set_xticks(positions)
        axp.set_xticklabels(labels)
        axp.set_ylabel(f"{args.metric} / epoch (steady-state)")
        axp.set_title(f"SPTnet per-epoch time by config ({args.metric})")
        plt.setp(axp.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        plot_path = os.path.join(out_dir, f"per_epoch_{args.metric}.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    print(f"\nWrote:\n  {summary_csv}\n  {md_path}" + (f"\n  {plot_path}" if plot_path else ""))


if __name__ == "__main__":
    main()
