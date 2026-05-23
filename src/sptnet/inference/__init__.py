"""Inference helpers for SPTnet."""

__all__ = [
    "extract_state_dict",
    "get_num_frames",
    "get_num_queries",
    "load_checkpoint_strict_enough",
    "normalize_state_dict_keys",
    "normalize_video_batch",
    "result_extension_for_input",
    "result_output_path",
    "run_batched_inference",
    "run_inference_loop",
    "stack_result_arrays",
    "write_inference_result_file",
]

_EXPORTS = {
    "extract_state_dict": ("sptnet.inference.predict", "extract_state_dict"),
    "get_num_frames": ("sptnet.inference.predict", "get_num_frames"),
    "get_num_queries": ("sptnet.inference.predict", "get_num_queries"),
    "load_checkpoint_strict_enough": ("sptnet.inference.predict", "load_checkpoint_strict_enough"),
    "normalize_state_dict_keys": ("sptnet.inference.predict", "normalize_state_dict_keys"),
    "normalize_video_batch": ("sptnet.inference.predict", "normalize_video_batch"),
    "result_extension_for_input": ("sptnet.inference.results_io", "result_extension_for_input"),
    "result_output_path": ("sptnet.inference.results_io", "result_output_path"),
    "run_batched_inference": ("sptnet.inference.predict", "run_batched_inference"),
    "run_inference_loop": ("sptnet.inference.predict", "run_inference_loop"),
    "stack_result_arrays": ("sptnet.inference.results_io", "stack_result_arrays"),
    "write_inference_result_file": ("sptnet.inference.results_io", "write_inference_result_file"),
}


def __getattr__(name):
    """Lazily import inference helpers so result I/O does not require Torch."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, symbol_name = _EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), symbol_name)
    globals()[name] = value
    return value
