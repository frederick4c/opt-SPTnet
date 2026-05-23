"""Inference result file naming and serialization helpers."""

from __future__ import annotations

import os
import tempfile
from os.path import basename, dirname

import h5py
import numpy as np
import scipy.io as sio


def result_extension_for_input(file_path: os.PathLike | str) -> str:
    """Return the inference result extension implied by the source file."""
    input_ext = os.path.splitext(str(file_path))[1].lower()
    return ".mat" if input_ext == ".mat" else ".h5"


def result_output_path(save_dir: os.PathLike | str, file_path: os.PathLike | str) -> str:
    """Build the result path, preserving MATLAB output only for MATLAB input."""
    stem = os.path.splitext(basename(str(file_path)))[0]
    return os.path.join(str(save_dir), f"result_{stem}{result_extension_for_input(file_path)}")


def stack_result_arrays(records: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack per-sample inference records into arrays written to result files."""
    estimation_obj = np.vstack(records["obj_estimation"])
    estimation_obj = np.expand_dims(estimation_obj, axis=1)  # [N, 1, Q, T]
    return {
        "obj_estimation": estimation_obj,
        "estimation_xy": np.vstack(records["estimation_xy"]),
        "estimation_H": np.stack([np.asarray(value).squeeze(-1) for value in records["estimation_H"]], axis=0),
        "estimation_C": np.stack([np.asarray(value).squeeze(-1) for value in records["estimation_C"]], axis=0),
    }


def write_inference_result_file(
    output_path: os.PathLike | str,
    arrays: dict[str, np.ndarray],
    source_file: os.PathLike | str | None = None,
) -> None:
    """Atomically write one inference result as MATLAB or native HDF5."""
    output_path = str(output_path)
    output_ext = os.path.splitext(output_path)[1].lower()
    tmp_suffix = output_ext if output_ext in {".mat", ".h5", ".hdf5"} else ".tmp"

    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=tmp_suffix,
        prefix="tmp_result_",
        dir=dirname(output_path),
        delete=False,
    ) as tf:
        tmp_output_path = tf.name

    try:
        if output_ext == ".mat":
            sio.savemat(tmp_output_path, mdict=arrays)
        else:
            with h5py.File(tmp_output_path, "w") as handle:
                for name, value in arrays.items():
                    handle.create_dataset(name, data=value)
                handle.attrs["format"] = "sptnet-inference-results"
                if source_file is not None:
                    handle.attrs["source_file"] = str(source_file)
        os.replace(tmp_output_path, output_path)
    finally:
        if os.path.exists(tmp_output_path):
            os.unlink(tmp_output_path)
