"""MAT/HDF5 training dataset for SPTnet."""

import h5py
import numpy as np
import torch


class TransformerMatDataset(torch.utils.data.Dataset):
    def __init__(self, config, dataset_path):
        super().__init__()
        self.config = config
        self.dataset_path = dataset_path
        self.dataset = h5py.File(dataset_path, "r")

        self.has_labels = all(k in self.dataset for k in ["Hlabel", "Clabel", "traceposition"])
        if "timelapsedata" not in self.dataset:
            raise KeyError("Missing variable 'timelapsedata' in dataset.")
        self.td = self.dataset["timelapsedata"]

    def __len__(self):
        if self.has_labels:
            return len(self.dataset["Hlabel"][1])
        if self.td.ndim == 3:
            return 1
        if self.td.ndim == 4:
            return self.td.shape[0]
        raise ValueError(f"'timelapsedata' must be 3D or 4D, got {self.td.shape}.")

    def __getitem__(self, idx):
        if self.td.ndim == 3:
            if idx != 0:
                raise IndexError("Index out of range for single 3D movie.")
            video = np.array(self.td)
        else:
            video = np.array(self.td[idx])

        if not self.has_labels:
            return {"video": video}

        hlabel_ref = np.array(self.dataset["Hlabel"][:, idx])
        clabel_ref = np.array(self.dataset["Clabel"][:, idx])
        position_ref = np.array(self.dataset["traceposition"][:, idx])

        if len(hlabel_ref) > self.config.num_queries:
            raise ValueError(
                f"num_queries ({self.config.num_queries}) must be >= label slots ({len(hlabel_ref)})."
            )

        hlabel = np.zeros(len(hlabel_ref))
        clabel = np.zeros(len(hlabel_ref))
        position = np.full((video.shape[0], len(hlabel_ref), 2), np.nan)
        class_label = np.full((video.shape[0], len(hlabel_ref)), 0)

        j = 0
        for i in range(len(hlabel_ref)):
            if np.array(self.dataset[hlabel_ref[i]][0]) != 0:
                hlabel[j] = float(np.array(self.dataset[hlabel_ref[i]][0]).item())
                clabel[j] = float(np.array(self.dataset[clabel_ref[i]][0]).item())
                pos_arr = np.array(self.dataset[position_ref[i]]).T
                if pos_arr.size == video.shape[0] * 2:
                    position[:, j, :] = pos_arr
                    class_label[:, j] = np.multiply(~np.isnan(position[:, j, 0]), 1)
                    j += 1

        query_pad = self.config.num_queries - hlabel.shape[0]
        class_label_pd = np.pad(class_label, [(0, 0), (0, query_pad)], "constant", constant_values=0)
        position_pd = np.pad(
            position,
            ((0, 0), (0, query_pad), (0, 0)),
            "constant",
            constant_values=np.nan,
        )

        outfov_mask = np.any(
            (position_pd < -self.config.image_size / 2) | (position_pd > self.config.image_size / 2),
            axis=2,
        )
        class_label_pd[outfov_mask] = 0

        return {
            "video": video,
            "position": position_pd,
            "Hlabel": hlabel,
            "Clabel": clabel,
            "class_label": class_label_pd,
        }

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
