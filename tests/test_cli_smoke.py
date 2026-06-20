"""Smoke tests for the command-line entry points.

These guard against the most common breakages in a CLI-focused package: an
entry point that no longer imports, or an argument parser that fails to build.
They run on CPU, need no model weights, and exercise ``--help`` (which builds
the full parser) plus argument parsing for the two training/inference CLIs.
"""

import importlib
import sys
from pathlib import Path

import pytest
import tomllib

# Most CLI modules import torch transitively (model/dataset stack), so skip the
# whole suite in minimal environments rather than failing to import.
pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]


def _entry_points():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return data["project"]["scripts"]


ENTRY_POINTS = _entry_points()

# Modules that expose a public ``build_arg_parser()`` factory.
PARSER_FACTORY_MODULES = [
    "sptnet.data.conversion",
    "sptnet.data.trackmate",
    "sptnet.segmentation.split",
    "sptnet.segmentation.stitch",
    "sptnet.visualization.preprocess",
]


def test_entry_points_are_declared():
    # Sanity check that the pyproject scripts table is non-trivial so the
    # parametrized import test below actually covers something.
    assert len(ENTRY_POINTS) >= 8


@pytest.mark.parametrize("target", sorted(set(ENTRY_POINTS.values())))
def test_entry_point_callable_imports(target):
    """Every ``module:function`` entry point imports and is callable."""
    module_name, func_name = target.split(":")
    module = importlib.import_module(module_name)
    assert callable(getattr(module, func_name))


@pytest.mark.parametrize("module_name", PARSER_FACTORY_MODULES)
def test_build_arg_parser_help(module_name):
    module = importlib.import_module(module_name)
    parser = module.build_arg_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_data_generator_parser_help():
    module = importlib.import_module("sptnet.training.data_generator")
    with pytest.raises(SystemExit) as exc:
        module._build_parser().parse_args(["--help"])
    assert exc.value.code == 0


def test_crlb_help():
    crlb = importlib.import_module("sptnet.training.crlb")
    with pytest.raises(SystemExit) as exc:
        crlb.main(["--help"])
    assert exc.value.code == 0


@pytest.mark.parametrize("module_name", ["sptnet.training.cli", "sptnet.inference.cli"])
def test_parse_args_help(module_name, monkeypatch):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(sys, "argv", [module_name, "--help"])
    with pytest.raises(SystemExit) as exc:
        module.parse_args()
    assert exc.value.code == 0


def test_training_parse_args_defaults(monkeypatch):
    module = importlib.import_module("sptnet.training.cli")
    monkeypatch.setattr(sys, "argv", ["sptnet-train"])
    args = module.parse_args()
    assert args.batch_size == 16
    assert args.query == 20
    assert args.grad_clip == 1.0


def test_inference_parse_args_minimal(monkeypatch):
    module = importlib.import_module("sptnet.inference.cli")
    monkeypatch.setattr(
        sys, "argv",
        ["sptnet-inference", "--model-path", "model", "--data", "a.h5", "b.h5"],
    )
    args = module.parse_args()
    assert args.model_path == "model"
    assert args.data == ["a.h5", "b.h5"]
    assert args.batch_size == 8
