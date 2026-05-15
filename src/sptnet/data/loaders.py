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
    train_set, val_set = torch.utils.data.random_split(dataset, split_lengths)
    num_workers = default_num_workers()
    persistent_workers = num_workers > 0
    print(f"DataLoader workers: {num_workers} (persistent_workers={persistent_workers})")

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
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
