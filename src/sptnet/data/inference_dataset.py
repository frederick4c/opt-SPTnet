"""Datasets used by SPTnet inference."""

from collections import defaultdict
import os

import h5py
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset


class HDF5DatasetMixin:
    def close(self):
        if getattr(self, "dataset", None) is not None:
            self.dataset.close()
            self.dataset = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def __del__(self):
        self.close()


class InferenceSimulationDataset(HDF5DatasetMixin, Dataset):
    def __init__(self, config, dataset_path):
        super().__init__()
        self.dataset = h5py.File(dataset_path, "r")

    def __len__(self):
        return len(self.dataset["Hlabel"][0])

    def __getitem__(self, idx):
        return {"video": np.array(self.dataset["timelapsedata"][idx])}


class RunningWindowSimulationDataset(HDF5DatasetMixin, Dataset):
    def __init__(self, config, dataset_path, window_size=30):
        super().__init__()
        self.dataset = h5py.File(dataset_path, "r")
        self.window_size = window_size

    def __len__(self):
        return self.dataset["timelapsedata"].shape[0] - self.window_size + 1

    def __getitem__(self, idx):
        video = np.array(self.dataset["timelapsedata"])
        return {"video": video[idx : idx + self.window_size, :, :]}


class BeadsDataset(HDF5DatasetMixin, Dataset):
    def __init__(self, config, dataset_path):
        super().__init__()
        self.dataset = h5py.File(dataset_path, "r")

    def __len__(self):
        return len(self.dataset["beadsdata"])

    def __getitem__(self, idx):
        return {"video": np.array(self.dataset["beadsdata"])}


class ExperimentalDataset(HDF5DatasetMixin, Dataset):
    def __init__(self, config, dataset_path):
        super().__init__()
        self.dataset = h5py.File(dataset_path, "r")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {"video": np.array(self.dataset["ims"])}


class ERDataset(HDF5DatasetMixin, Dataset):
    def __init__(self, config, dataset_path):
        super().__init__()
        self.dataset = h5py.File(dataset_path, "r")

    def __len__(self):
        return len(self.dataset["ims"])

    def __getitem__(self, idx):
        return {"video": np.array(self.dataset["ims"][idx])}


class FileSampleDataset(Dataset):
    """Flatten MAT/TIFF files into inference samples grouped by video shape."""

    def __init__(self, file_list, mat_clip_index=0):
        self.records = []
        self.shape_groups = defaultdict(list)
        self.mat_clip_index = int(mat_clip_index)

        for file_path in file_list:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in [".tif", ".tiff"]:
                video = tifffile.imread(file_path)
                if video.ndim == 2:
                    raise ValueError(f"{file_path} contains only one frame; need a time series.")
                self._add_record(file_path, ext, 0, tuple(video.shape))
                continue

            with h5py.File(file_path, "r") as f:
                if "timelapsedata" not in f:
                    raise KeyError(f"Missing variable 'timelapsedata' in dataset: {file_path}")
                td = f["timelapsedata"]
                if td.ndim == 3:
                    self._add_record(file_path, ext, 0, tuple(td.shape))
                elif td.ndim == 4:
                    if self.mat_clip_index < 0 or self.mat_clip_index >= td.shape[0]:
                        raise IndexError(
                            f"mat_clip_index={self.mat_clip_index} out of range for {file_path} "
                            f"(N={td.shape[0]})."
                        )
                    self._add_record(file_path, ext, self.mat_clip_index, tuple(td.shape[1:]))
                else:
                    raise ValueError(f"'timelapsedata' must be 3D or 4D, got {td.shape} in {file_path}.")

    def _add_record(self, file_path, ext, sample_idx, shape_key):
        record = {
            "file_path": file_path,
            "ext": ext,
            "sample_idx": sample_idx,
            "shape_key": shape_key,
        }
        self.shape_groups[shape_key].append(len(self.records))
        self.records.append(record)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        file_path = rec["file_path"]
        ext = rec["ext"]
        sample_idx = rec["sample_idx"]

        if ext in [".tif", ".tiff"]:
            video = tifffile.imread(file_path).astype(np.float32)
        else:
            with h5py.File(file_path, "r") as f:
                td = f["timelapsedata"]
                if td.ndim == 3:
                    video = np.array(td, dtype=np.float32)
                else:
                    video = np.array(td[sample_idx], dtype=np.float32)

        return {
            "video": video,
            "file_path": file_path,
            "sample_idx": sample_idx,
        }


class SubsetByIndices(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def collate_inference(batch):
    videos = torch.stack([torch.from_numpy(item["video"]) for item in batch], dim=0)
    return {
        "video": videos,
        "file_path": [item["file_path"] for item in batch],
        "sample_idx": [item["sample_idx"] for item in batch],
    }
