"""DataLoader helpers for SPTnet."""

import os

import torch


def default_num_workers():
    """Return the DataLoader worker count from environment defaults.

    `SPT_NUM_WORKERS` overrides the value. Otherwise the helper uses a small
    conservative default, capped at two workers, to avoid HDF5 handle pressure
    on dense training jobs.
    """
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", min(os.cpu_count() or 2, 2)))
    default_workers = min(slurm_cpus, 2)
    return int(os.environ.get("SPT_NUM_WORKERS", default_workers))


def create_train_val_loaders(dataset, split_lengths, batch_size):
    """Split a dataset and build train/validation DataLoaders.

    Returns
    -------
    tuple
        `(train_loader, val_loader, train_set, val_set)`.
    """
    dataset_length = len(dataset)
    if dataset_length <= 0:
        raise ValueError("Cannot build training DataLoaders from an empty dataset.")
    if len(split_lengths) != 2:
        raise ValueError(f"Expected two split lengths, got {split_lengths!r}.")
    if sum(split_lengths) != dataset_length:
        raise ValueError(
            f"Split lengths {split_lengths!r} do not sum to dataset length {dataset_length}."
        )
    if split_lengths[0] == 0:
        split_lengths = [1, dataset_length - 1]
        print(
            "Adjusted train/validation split for tiny dataset: "
            f"train={split_lengths[0]}, val={split_lengths[1]}."
        )

    train_set, val_set = torch.utils.data.random_split(dataset, split_lengths)
    num_workers = default_num_workers()
    persistent_workers = num_workers > 0
    train_drop_last = len(train_set) >= batch_size
    val_drop_last = len(val_set) >= batch_size
    print(f"DataLoader workers: {num_workers} (persistent_workers={persistent_workers})")
    if not train_drop_last or not val_drop_last:
        print(
            "Tiny split detected; disabling drop_last for "
            f"train={len(train_set)} and val={len(val_set)} samples."
        )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=train_drop_last,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=len(val_set) > 0,
        num_workers=num_workers,
        drop_last=val_drop_last,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader, train_set, val_set


def create_test_loader(dataset, batch_size=1, shuffle=False):
    """Build a single-process DataLoader for evaluation or inference."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )
