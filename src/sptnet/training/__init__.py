"""Training helpers for SPTnet."""


def __getattr__(name):
    if name in {
        "compute_crlb_matrix",
        "generate_crlb_file",
        "load_or_generate_crlb_matrix",
        "plot_crlb_surfaces",
        "save_crlb_matrix",
        "validate_crlb_matrix",
    }:
        from sptnet.training import crlb

        return getattr(crlb, name)
    if name == "hungarian_matched_loss":
        from sptnet.training.losses import hungarian_matched_loss

        return hungarian_matched_loss
    if name == "normalize_training_inputs":
        from sptnet.training.trainer import normalize_training_inputs

        return normalize_training_inputs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "compute_crlb_matrix",
    "generate_crlb_file",
    "load_or_generate_crlb_matrix",
    "hungarian_matched_loss",
    "normalize_training_inputs",
    "plot_crlb_surfaces",
    "save_crlb_matrix",
    "validate_crlb_matrix",
]
