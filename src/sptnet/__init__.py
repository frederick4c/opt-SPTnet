"""SPTnet package."""

__all__ = [
    "BackBone",
    "ResidualBlock",
    "SPTnet",
    "Transformer",
    "Transformer3d",
    "TransformerMatDataset",
]

_EXPORTS = {
    "BackBone": ("sptnet.models.backbone", "BackBone"),
    "ResidualBlock": ("sptnet.models.backbone", "ResidualBlock"),
    "SPTnet": ("sptnet.models.sptnet", "SPTnet"),
    "Transformer": ("sptnet.models.transformers", "Transformer"),
    "Transformer3d": ("sptnet.models.transformers", "Transformer3d"),
    "TransformerMatDataset": ("sptnet.data.mat_dataset", "TransformerMatDataset"),
}


def __getattr__(name):
    """Lazily import public symbols so lightweight utilities stay lightweight."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, symbol_name = _EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), symbol_name)
    globals()[name] = value
    return value
