"""Shared file IO helpers for segmentation workflows."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np
import scipy.io as sio


DEFAULT_DATASET_NAMES = ("timelapsedata", "ims")


def read_named_array(path: os.PathLike, dataset_names: Sequence[str] = DEFAULT_DATASET_NAMES):
    """Read the first matching array from HDF5/MATLAB files.

    HDF5, HDF5-backed MATLAB v7.3 ``.mat`` files, and older MATLAB v5 files
    written by ``scipy.io.savemat`` are supported.
    """
    path = Path(path)
    try:
        with h5py.File(path, "r") as handle:
            for name in dataset_names:
                if name in handle:
                    return np.asarray(handle[name]), name
    except OSError:
        pass

    data = sio.loadmat(path)
    for name in dataset_names:
        if name in data:
            return np.asarray(data[name]), name
    visible = sorted(name for name in data if not name.startswith("__"))
    expected = ", ".join(dataset_names)
    raise KeyError(f"{path} does not contain any expected dataset ({expected}); found {visible}.")


def write_hdf5_array(
    path: os.PathLike,
    dataset_name: str,
    array: np.ndarray,
    attrs: Mapping[str, object] | None = None,
    *,
    overwrite: bool = True,
) -> bool:
    """Write one array to an HDF5 file and return whether it was written."""
    path = Path(path)
    if path.exists() and not overwrite:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(dataset_name, data=array)
        if attrs:
            for key, value in attrs.items():
                if value is None:
                    continue
                if isinstance(value, Path):
                    value = str(value)
                handle.attrs[key] = value
    return True


def read_hdf5_attrs(path: os.PathLike) -> dict[str, object]:
    """Read HDF5 root attributes, returning an empty dict for non-HDF5 files."""
    try:
        with h5py.File(path, "r") as handle:
            return dict(handle.attrs.items())
    except OSError:
        return {}
