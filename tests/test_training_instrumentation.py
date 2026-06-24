"""Unit tests for the benchmark instrumentation in ``sptnet.training.cli``.

The runtime-optimisation claim (the headline result) rests on the per-epoch
``epoch_timing.csv`` log and the ``SPT_*`` optimisation toggles. These tests lock
the CSV schema and the toggle-resolution semantics so a refactor cannot silently
change what the benchmark analyzer consumes. Everything runs on CPU with no
training.
"""

import csv

import pytest

torch = pytest.importorskip("torch")

from sptnet.training import cli


def test_epoch_timing_columns_schema():
    # The analyzer (experiments/benchmarks/analyze_benchmarks.py) reads these by
    # name; locking the tuple guards the train/val/total + toggle contract.
    assert cli.EPOCH_TIMING_COLUMNS == (
        "epoch",
        "train_seconds",
        "val_seconds",
        "epoch_total_seconds",
        "n_train_batches",
        "n_val_batches",
        "amp",
        "tf32",
        "cudnn_benchmark",
    )


def test_loss_history_round_trip(tmp_path):
    csv_path = tmp_path / "loss_history.csv"
    fieldnames = [
        "epoch", "t_loss", "v_loss", "t_cls", "v_cls", "t_coor", "v_coor",
        "t_hurst", "v_hurst", "t_diff", "v_diff", "t_bg", "v_bg",
    ]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({name: (1 if name == "epoch" else 0.5) for name in fieldnames})
        writer.writerow({name: (2 if name == "epoch" else 0.25) for name in fieldnames})

    history = cli._load_loss_history(str(csv_path))
    assert history["epoch"] == [1, 2]
    assert history["v_loss"] == [0.5, 0.25]


def test_load_loss_history_missing_file_returns_empty():
    history = cli._load_loss_history("/no/such/file.csv")
    assert history["epoch"] == []
    assert all(history[key] == [] for key in history)


def test_copy_loss_history(tmp_path):
    src = tmp_path / "src.csv"
    dst = tmp_path / "dst.csv"
    src.write_text("epoch,v_loss\n1,0.5\n")
    cli._copy_loss_history(str(src), str(dst))
    assert dst.read_text() == "epoch,v_loss\n1,0.5\n"


def test_copy_loss_history_noop_when_same_path(tmp_path):
    path = tmp_path / "same.csv"
    path.write_text("epoch,v_loss\n1,0.5\n")
    # Must not truncate the file when src == dst.
    cli._copy_loss_history(str(path), str(path))
    assert path.read_text() == "epoch,v_loss\n1,0.5\n"


# --- optimisation toggle resolvers ---

def test_tf32_default_on_disabled_by_env():
    assert cli.resolve_tf32_allowed({}) is True
    assert cli.resolve_tf32_allowed({"SPT_DISABLE_TF32": "1"}) is False
    assert cli.resolve_tf32_allowed({"SPT_DISABLE_TF32": "0"}) is True


def test_amp_enabled_requires_cuda_and_not_disabled():
    assert cli.resolve_amp_enabled("cpu", {}) is False
    assert cli.resolve_amp_enabled("cuda", {}) is True
    assert cli.resolve_amp_enabled("cuda", {"SPT_DISABLE_AMP": "1"}) is False


def test_cudnn_benchmark_default_on():
    assert cli.resolve_cudnn_benchmark({}) is True
    assert cli.resolve_cudnn_benchmark({"SPT_CUDNN_BENCHMARK": "0"}) is False


def test_amp_dtype_selection_and_validation():
    assert cli.resolve_amp_dtype({}) is torch.bfloat16
    assert cli.resolve_amp_dtype({"SPT_AMP_DTYPE": "fp16"}) is torch.float16
    assert cli.resolve_amp_dtype({"SPT_AMP_DTYPE": "bf16"}) is torch.bfloat16
    with pytest.raises(ValueError):
        cli.resolve_amp_dtype({"SPT_AMP_DTYPE": "float8"})


def test_grad_scaler_follows_dtype_unless_overridden():
    # fp16 keeps the scaler, bf16 drops it by default.
    assert cli.resolve_use_grad_scaler(True, torch.float16, {}) is True
    assert cli.resolve_use_grad_scaler(True, torch.bfloat16, {}) is False
    # Explicit override wins.
    assert cli.resolve_use_grad_scaler(True, torch.bfloat16, {"SPT_DISABLE_GRAD_SCALER": "0"}) is True
    assert cli.resolve_use_grad_scaler(True, torch.float16, {"SPT_DISABLE_GRAD_SCALER": "1"}) is False
    # Disabled AMP always disables the scaler.
    assert cli.resolve_use_grad_scaler(False, torch.float16, {}) is False


def test_tf32_helper_drives_module_import_state(monkeypatch):
    # SPT_DISABLE_TF32 is consumed at import time; reloading under the env var
    # must flip the resolved module-level flag.
    import importlib

    monkeypatch.setenv("SPT_DISABLE_TF32", "1")
    reloaded = importlib.reload(cli)
    try:
        assert reloaded._ALLOW_TF32 is False
    finally:
        monkeypatch.delenv("SPT_DISABLE_TF32", raising=False)
        importlib.reload(reloaded)
