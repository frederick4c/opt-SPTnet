"""Prediction and checkpoint-loading helpers."""

import numpy as np
import torch


def extract_state_dict(ckpt_obj):
    """Return a plain model state dict from a checkpoint-like object."""
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
        return ckpt_obj["state_dict"]
    return ckpt_obj


def get_num_queries(ckpt_path=None, state_dict=None):
    """Infer the number of query slots from `query_embed.weight`.

    Provide either a checkpoint path or an already-loaded state dict.
    """
    if state_dict is None:
        if ckpt_path is None:
            raise ValueError("Either ckpt_path or state_dict must be provided.")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = extract_state_dict(ckpt)

    for key, value in state_dict.items():
        if "query_embed.weight" in key:
            return value.shape[0]
    raise ValueError("query_embed.weight not found")


def get_num_frames(ckpt_path=None, state_dict=None):
    """Infer the model frame count from `conv_temp.weight`.

    Provide either a checkpoint path or an already-loaded state dict. The
    current SPTnet architecture stores the expected number of frames as the
    output channel count of the temporal `1x1` convolution.
    """
    if state_dict is None:
        if ckpt_path is None:
            raise ValueError("Either ckpt_path or state_dict must be provided.")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = extract_state_dict(ckpt)

    for key, value in state_dict.items():
        if "conv_temp.weight" in key:
            return value.shape[0]
    raise ValueError("conv_temp.weight not found")


def normalize_state_dict_keys(state_dict, model):
    """Handle common DataParallel `module.` prefix mismatches."""
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected state_dict dict, got {type(state_dict)}")

    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())
    if not model_keys or not ckpt_keys:
        return state_dict

    ckpt_has_module = all(k.startswith("module.") for k in ckpt_keys)
    model_has_module = all(k.startswith("module.") for k in model_keys)
    if ckpt_has_module and not model_has_module:
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    if model_has_module and not ckpt_has_module:
        return {f"module.{k}": v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint_strict_enough(model, ckpt_path=None, device=None, state_dict=None, min_loaded_fraction=0.9):
    """Load model weights and fail if too few tensors match.

    This is intentionally stricter than `strict=False` alone: it still allows
    wrapper-prefix differences and minor non-critical mismatches, but catches
    accidental architecture/checkpoint mismatches early.
    """
    if state_dict is None:
        if ckpt_path is None:
            raise ValueError("Either ckpt_path or state_dict must be provided.")
        map_loc = device if device is not None else "cpu"
        ckpt = torch.load(ckpt_path, map_location=map_loc)
        state_dict = extract_state_dict(ckpt)

    state_dict = normalize_state_dict_keys(state_dict, model)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    total = len(model.state_dict())
    loaded = total - len(missing)
    loaded_fraction = loaded / max(total, 1)

    print(
        f"Checkpoint load summary: loaded {loaded}/{total} tensors "
        f"({loaded_fraction * 100:.1f}%), missing={len(missing)}, unexpected={len(unexpected)}"
    )
    if missing:
        print("  First missing keys:", missing[:8])
    if unexpected:
        print("  First unexpected keys:", unexpected[:8])

    if loaded == 0:
        raise RuntimeError(
            "No model weights were loaded from checkpoint. "
            "This usually means architecture/key mismatch."
        )
    if loaded_fraction < min_loaded_fraction:
        raise RuntimeError(
            f"Only {loaded}/{total} tensors loaded (<{min_loaded_fraction:.0%}). "
            "Checkpoint and inference model are likely incompatible."
        )
    return incompatible


def normalize_video_batch(inputs):
    """Apply per-sample min-max normalization to `[B, 1, T, H, W]` videos."""
    image_max = inputs.amax(dim=tuple(range(2, inputs.ndim)), keepdim=True)
    image_min = inputs.amin(dim=tuple(range(2, inputs.ndim)), keepdim=True)
    return (inputs - image_min) / (image_max - image_min).clamp_min(1e-8)


def run_batched_inference(model, dataloader, device):
    """Run inference over a DataLoader and return per-sample result records."""
    results = []
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            inputs = data["video"].unsqueeze(1).float().to(device)
            inputs = normalize_video_batch(inputs)
            class_out, center_out, h_out, d_out = model(inputs)

            class_out = class_out.detach().cpu().numpy()
            center_out = center_out.detach().cpu().numpy()
            h_out = h_out.detach().cpu().numpy()
            d_out = d_out.detach().cpu().numpy()

            for i in range(class_out.shape[0]):
                results.append(
                    {
                        "file_path": data["file_path"][i],
                        "sample_idx": data["sample_idx"][i],
                        "obj_estimation": class_out[i : i + 1],
                        "estimation_xy": center_out[i : i + 1],
                        "estimation_H": h_out[i],
                        "estimation_C": d_out[i],
                    }
                )
    return results


def run_inference_loop(model, dataloader, checkpoint_path, device):
    """Load a checkpoint and run the legacy-style inference accumulation loop."""
    load_checkpoint_strict_enough(model, ckpt_path=checkpoint_path, device=device, min_loaded_fraction=0.0)
    model.eval()
    total_obj_est = []
    total_xy_est = []
    total_h_est = []
    total_d_est = []

    with torch.no_grad():
        for data in dataloader:
            inputs = data["video"].unsqueeze(1).float().to(device)
            inputs = normalize_video_batch(inputs)
            class_out, center_out, h_out, d_out = model(inputs)
            total_obj_est.append(np.array(class_out.unsqueeze(0).cpu()))
            total_xy_est.append(np.array(center_out.cpu()))
            total_h_est.append(np.array(h_out.squeeze().cpu()))
            total_d_est.append(np.array(d_out.squeeze().cpu()))

    return total_obj_est, total_xy_est, total_h_est, total_d_est
