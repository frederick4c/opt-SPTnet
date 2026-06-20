"""Tests for the previously untested visualization modules:
``sptnet.visualization.background`` (dim-movie background removal) and the
``sptnet.visualization.preprocess`` CLI wrapper. CPU/numpy only, no torch.
"""

import numpy as np
import pytest
import tifffile

from sptnet.visualization import background, preprocess


def _planted_spot_stack(t=12, size=32, bg=100.0, amp=40.0):
    """A static background with one bright Gaussian spot per frame."""
    rng = np.random.default_rng(0)
    stack = bg + rng.normal(0, 2.0, size=(t, size, size)).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size // 2, size // 2
    spot = amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.5 ** 2))
    stack += spot.astype(np.float32)
    return stack


def test_subtract_temporal_background_removes_static_level():
    stack = np.full((8, 16, 16), 50.0, dtype=np.float32)
    out = background.subtract_temporal_background(stack, q=50.0)
    # A perfectly static background subtracts to (clipped) zero.
    assert np.allclose(out, 0.0)
    # copy=True must not mutate the input.
    assert np.allclose(stack, 50.0)


def test_normalize_contrast_global_in_unit_range():
    stack = _planted_spot_stack()
    out = background.normalize_contrast(stack, mode="global")
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert out.dtype == np.float32


def test_normalize_contrast_none_is_passthrough():
    stack = _planted_spot_stack()
    out = background.normalize_contrast(stack, mode="none")
    assert np.array_equal(out, stack.astype(np.float32))


def test_remove_spatial_background_invalid_mode():
    with pytest.raises(ValueError):
        background.remove_spatial_background(_planted_spot_stack(), mode="bogus")


def test_remove_background_preserves_shape_and_range():
    stack = _planted_spot_stack()
    # temporal="none" is the documented pure best-detectability setting; a
    # temporal median would (correctly) remove this deliberately static spot.
    out = background.remove_background(stack, temporal="none", spatial="dog", normalize="global")
    assert out.shape == stack.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0
    # The planted spot must survive: the brightest output pixel sits at the
    # spot centre, not lost in the (zeroed) background.
    cy, cx = stack.shape[1] // 2, stack.shape[2] // 2
    peak_frame = out[0]
    py, px = np.unravel_index(int(np.argmax(peak_frame)), peak_frame.shape)
    assert abs(py - cy) <= 2 and abs(px - cx) <= 2


def test_to_output_dtype_variants():
    unit = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    assert preprocess.to_output_dtype(unit, "float32").dtype == np.float32
    u16 = preprocess.to_output_dtype(unit, "uint16")
    assert u16.dtype == np.uint16 and u16.max() == 65535
    u8 = preprocess.to_output_dtype(unit, "uint8")
    assert u8.dtype == np.uint8 and u8.max() == 255
    with pytest.raises(ValueError):
        preprocess.to_output_dtype(unit, "float16")


def test_expand_inputs_explicit_paths(tmp_path):
    a = tmp_path / "a.tif"
    b = tmp_path / "b.tif"
    a.touch()
    b.touch()
    out = preprocess.expand_inputs([str(b), str(a)])
    assert out == sorted([a, b])


def test_preprocess_parser_help():
    with pytest.raises(SystemExit) as exc:
        preprocess.build_arg_parser().parse_args(["--help"])
    assert exc.value.code == 0


def test_preprocess_main_end_to_end(tmp_path):
    movie = tmp_path / "movie.tif"
    tifffile.imwrite(str(movie), _planted_spot_stack().astype(np.uint16))
    out_dir = tmp_path / "out"
    rc = preprocess.main(
        [str(movie), "--output-dir", str(out_dir), "--max-frames", "6", "--dtype", "uint8"]
    )
    assert rc == 0
    produced = list(out_dir.glob("*.tif"))
    assert len(produced) == 1
    result = tifffile.imread(str(produced[0]))
    assert result.shape[0] == 6
