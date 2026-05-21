from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import tifffile

torch = pytest.importorskip("torch")

from sptnet.data.inference_dataset import FileSampleDataset, SubsetByIndices, collate_inference
from sptnet.data.loaders import default_num_workers
from sptnet.data.mat_dataset import TransformerMatDataset


def _write_hdf5(path, name, data):
    with h5py.File(path, "w") as handle:
        handle.create_dataset(name, data=data)


def test_file_sample_dataset_reads_hdf5_and_tiff_records(tmp_path):
    hdf5_path = tmp_path / "clips.h5"
    tif_path = tmp_path / "movie.tif"
    hdf5_data = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    tif_data = np.arange(3 * 2 * 2, dtype=np.uint16).reshape(3, 2, 2)
    _write_hdf5(hdf5_path, "timelapsedata", hdf5_data)
    tifffile.imwrite(tif_path, tif_data)

    dataset = FileSampleDataset([str(hdf5_path), str(tif_path)], mat_clip_index=1)

    assert len(dataset) == 2
    assert set(dataset.shape_groups) == {(3, 4, 5), (3, 2, 2)}
    np.testing.assert_array_equal(dataset[0]["video"], hdf5_data[1])
    np.testing.assert_array_equal(dataset[1]["video"], tif_data.astype(np.float32))


def test_file_sample_dataset_reads_legacy_mat_records(tmp_path):
    mat_path = tmp_path / "clips.mat"
    mat_data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    _write_hdf5(mat_path, "timelapsedata", mat_data)

    dataset = FileSampleDataset([str(mat_path)])

    assert len(dataset) == 1
    np.testing.assert_array_equal(dataset[0]["video"], mat_data)


def test_file_sample_dataset_rejects_single_frame_tiff(tmp_path):
    tif_path = tmp_path / "single.tif"
    tifffile.imwrite(tif_path, np.ones((4, 4), dtype=np.uint16))

    with pytest.raises(ValueError, match="only one frame"):
        FileSampleDataset([str(tif_path)])


def test_collate_inference_preserves_metadata_and_stacks_videos():
    batch = [
        {"video": np.ones((2, 3, 4), dtype=np.float32), "file_path": "a.mat", "sample_idx": 0},
        {"video": np.zeros((2, 3, 4), dtype=np.float32), "file_path": "b.mat", "sample_idx": 2},
    ]

    collated = collate_inference(batch)

    assert collated["video"].shape == (2, 2, 3, 4)
    assert collated["video"].dtype == torch.float32
    assert collated["file_path"] == ["a.mat", "b.mat"]
    assert collated["sample_idx"] == [0, 2]


def test_subset_by_indices_maps_requested_items():
    subset = SubsetByIndices(["zero", "one", "two"], [2, 0])

    assert len(subset) == 2
    assert subset[0] == "two"
    assert subset[1] == "zero"


def test_default_num_workers_honors_environment(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    monkeypatch.delenv("SPT_NUM_WORKERS", raising=False)
    assert default_num_workers() == 2

    monkeypatch.setenv("SPT_NUM_WORKERS", "0")
    assert default_num_workers() == 0


def test_transformer_mat_dataset_reads_unlabeled_single_movie(tmp_path):
    hdf5_path = tmp_path / "unlabeled.h5"
    video = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    _write_hdf5(hdf5_path, "timelapsedata", video)

    with TransformerMatDataset(SimpleNamespace(num_queries=4, image_size=8), hdf5_path) as dataset:
        assert len(dataset) == 1
        np.testing.assert_array_equal(dataset[0]["video"], video)
        with pytest.raises(IndexError):
            dataset[1]


def test_transformer_mat_dataset_reads_labels_and_masks_out_of_fov(tmp_path):
    mat_path = tmp_path / "labeled.mat"
    video = np.ones((1, 3, 4, 4), dtype=np.float32)

    with h5py.File(mat_path, "w") as handle:
        handle.create_dataset("timelapsedata", data=video)
        h0 = handle.create_dataset("h0", data=np.array([0.4]))
        h1 = handle.create_dataset("h1", data=np.array([0.0]))
        c0 = handle.create_dataset("c0", data=np.array([0.03]))
        c1 = handle.create_dataset("c1", data=np.array([0.0]))
        p0 = handle.create_dataset(
            "p0",
            data=np.array(
                [
                    [0.0, 3.0, np.nan],
                    [0.0, 0.0, np.nan],
                ]
            ),
        )
        p1 = handle.create_dataset("p1", data=np.zeros((2, 3)))

        h_refs = np.array([[h0.ref], [h1.ref]], dtype=h5py.ref_dtype)
        c_refs = np.array([[c0.ref], [c1.ref]], dtype=h5py.ref_dtype)
        p_refs = np.array([[p0.ref], [p1.ref]], dtype=h5py.ref_dtype)
        handle.create_dataset("Hlabel", data=h_refs)
        handle.create_dataset("Clabel", data=c_refs)
        handle.create_dataset("traceposition", data=p_refs)

    with TransformerMatDataset(SimpleNamespace(num_queries=4, image_size=4), mat_path) as dataset:
        sample = dataset[0]

    assert sample["video"].shape == (3, 4, 4)
    np.testing.assert_array_equal(sample["Hlabel"], np.array([0.4, 0.0]))
    np.testing.assert_array_equal(sample["Clabel"], np.array([0.03, 0.0]))
    assert sample["position"].shape == (3, 4, 2)
    assert sample["class_label"].shape == (3, 4)
    np.testing.assert_array_equal(sample["class_label"][:, 0], np.array([1, 0, 0]))
    assert np.all(sample["class_label"][:, 1:] == 0)


def test_transformer_mat_dataset_rejects_too_many_label_slots(tmp_path):
    mat_path = tmp_path / "too_many.mat"
    with h5py.File(mat_path, "w") as handle:
        handle.create_dataset("timelapsedata", data=np.ones((1, 3, 4, 4), dtype=np.float32))
        label = handle.create_dataset("h", data=np.array([0.5]))
        diff = handle.create_dataset("c", data=np.array([0.02]))
        pos = handle.create_dataset("p", data=np.zeros((2, 3)))
        handle.create_dataset("Hlabel", data=np.array([[label.ref], [label.ref]], dtype=h5py.ref_dtype))
        handle.create_dataset("Clabel", data=np.array([[diff.ref], [diff.ref]], dtype=h5py.ref_dtype))
        handle.create_dataset("traceposition", data=np.array([[pos.ref], [pos.ref]], dtype=h5py.ref_dtype))

    with TransformerMatDataset(SimpleNamespace(num_queries=1, image_size=4), mat_path) as dataset:
        with pytest.raises(ValueError, match="num_queries"):
            dataset[0]
