"""Dataset and DataLoader helpers for SPTnet."""

__all__ = [
    "BeadsDataset",
    "ERDataset",
    "ExperimentalDataset",
    "FileSampleDataset",
    "InferenceSimulationDataset",
    "RunningWindowSimulationDataset",
    "SubsetByIndices",
    "TransformerMatDataset",
    "TiffConversionResult",
    "collate_inference",
    "convert_hdf5_file_to_tiff",
    "convert_hdf5_files_to_tiff",
    "convert_mat_file_to_tiff",
    "convert_mat_files_to_tiff",
    "create_test_loader",
    "create_train_val_loaders",
    "expand_file_patterns",
    "find_movie_dataset",
]

_EXPORTS = {
    "BeadsDataset": ("sptnet.data.inference_dataset", "BeadsDataset"),
    "ERDataset": ("sptnet.data.inference_dataset", "ERDataset"),
    "ExperimentalDataset": ("sptnet.data.inference_dataset", "ExperimentalDataset"),
    "FileSampleDataset": ("sptnet.data.inference_dataset", "FileSampleDataset"),
    "InferenceSimulationDataset": ("sptnet.data.inference_dataset", "InferenceSimulationDataset"),
    "RunningWindowSimulationDataset": ("sptnet.data.inference_dataset", "RunningWindowSimulationDataset"),
    "SubsetByIndices": ("sptnet.data.inference_dataset", "SubsetByIndices"),
    "TransformerMatDataset": ("sptnet.data.mat_dataset", "TransformerMatDataset"),
    "TiffConversionResult": ("sptnet.data.conversion", "TiffConversionResult"),
    "collate_inference": ("sptnet.data.inference_dataset", "collate_inference"),
    "convert_hdf5_file_to_tiff": ("sptnet.data.conversion", "convert_hdf5_file_to_tiff"),
    "convert_hdf5_files_to_tiff": ("sptnet.data.conversion", "convert_hdf5_files_to_tiff"),
    "convert_mat_file_to_tiff": ("sptnet.data.conversion", "convert_mat_file_to_tiff"),
    "convert_mat_files_to_tiff": ("sptnet.data.conversion", "convert_mat_files_to_tiff"),
    "create_test_loader": ("sptnet.data.loaders", "create_test_loader"),
    "create_train_val_loaders": ("sptnet.data.loaders", "create_train_val_loaders"),
    "expand_file_patterns": ("sptnet.data.conversion", "expand_file_patterns"),
    "find_movie_dataset": ("sptnet.data.conversion", "find_movie_dataset"),
}


def __getattr__(name):
    """Lazily import public data helpers."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, symbol_name = _EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), symbol_name)
    globals()[name] = value
    return value
