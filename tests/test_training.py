import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sptnet.training.losses import hungarian_matched_loss
from sptnet.training.trainer import normalize_training_inputs


def _crlb_matrix(diff_count=5, hurst_count=99, frames=3):
    matrix = np.ones((2, 2, diff_count, hurst_count, frames), dtype=np.float64)
    matrix[:, :, :, :, 0] = 0.0
    return matrix


def test_normalize_training_inputs_operates_per_sample_in_place():
    inputs = torch.tensor(
        [
            [[[[0.0, 2.0], [4.0, 6.0]]]],
            [[[[3.0, 3.0], [3.0, 3.0]]]],
        ]
    )
    original_id = id(inputs)

    normalized = normalize_training_inputs(inputs)

    assert id(normalized) == original_id
    torch.testing.assert_close(normalized[0, 0].amin(), torch.tensor(0.0))
    torch.testing.assert_close(normalized[0, 0].amax(), torch.tensor(1.0))
    torch.testing.assert_close(normalized[1, 0], torch.zeros_like(normalized[1, 0]))
    assert torch.isfinite(normalized).all()


def test_hungarian_loss_handles_background_only_batches():
    pred_classes = torch.full((1, 3, 3), 0.1, requires_grad=True)
    pred_positions = torch.zeros((1, 3, 3, 2), requires_grad=True)
    pred_h = torch.full((1, 3), 0.5, requires_grad=True)
    pred_d = torch.full((1, 3), 0.5, requires_grad=True)
    gt_classes = torch.zeros((1, 3, 1))
    gt_positions = torch.full((1, 3, 1, 2), float("nan"))
    gt_h = torch.zeros((1, 1))
    gt_d = torch.zeros((1, 1))

    loss, class_loss, coord_loss, h_loss, d_loss, bg_loss = hungarian_matched_loss(
        pred_classes,
        pred_positions,
        pred_h,
        pred_d,
        gt_classes,
        gt_positions,
        gt_h,
        gt_d,
        num_queries=3,
        diff_max=0.05,
        num_frames=3,
        crlb_matrix=_crlb_matrix(),
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(bg_loss)
    assert class_loss == 0
    assert coord_loss == 0
    assert h_loss == 0
    assert d_loss == 0
    loss.backward()
    assert pred_classes.grad is not None


def test_hungarian_loss_matches_simple_single_track_and_remains_finite():
    pred_classes = torch.tensor([[[0.95, 0.95, 0.05], [0.05, 0.05, 0.05], [0.1, 0.1, 0.1]]], requires_grad=True)
    pred_positions = torch.zeros((1, 3, 3, 2), requires_grad=True)
    pred_h = torch.tensor([[0.4, 0.2, 0.7]], requires_grad=True)
    pred_d = torch.tensor([[0.4, 0.2, 0.7]], requires_grad=True)
    gt_classes = torch.tensor([[[1.0], [1.0], [0.0]]])
    gt_positions = torch.zeros((1, 3, 1, 2))
    gt_positions[:, 2, :, :] = float("nan")
    gt_h = torch.tensor([[0.4]])
    gt_d = torch.tensor([[0.4]])

    outputs = hungarian_matched_loss(
        pred_classes,
        pred_positions,
        pred_h,
        pred_d,
        gt_classes,
        gt_positions,
        gt_h,
        gt_d,
        num_queries=3,
        diff_max=0.05,
        num_frames=3,
        crlb_matrix=_crlb_matrix(),
    )

    loss = outputs[0]
    assert torch.isfinite(loss)
    assert loss.item() < 1.0
    loss.backward()
    assert pred_classes.grad is not None


def test_hungarian_loss_explicit_default_weights_match_default_behavior():
    pred_classes = torch.tensor([[[0.95, 0.95, 0.05], [0.05, 0.05, 0.05], [0.1, 0.1, 0.1]]])
    pred_positions = torch.zeros((1, 3, 3, 2))
    pred_h = torch.tensor([[0.4, 0.2, 0.7]])
    pred_d = torch.tensor([[0.4, 0.2, 0.7]])
    gt_classes = torch.tensor([[[1.0], [1.0], [0.0]]])
    gt_positions = torch.zeros((1, 3, 1, 2))
    gt_positions[:, 2, :, :] = float("nan")
    gt_h = torch.tensor([[0.4]])
    gt_d = torch.tensor([[0.4]])

    default_outputs = hungarian_matched_loss(
        pred_classes,
        pred_positions,
        pred_h,
        pred_d,
        gt_classes,
        gt_positions,
        gt_h,
        gt_d,
        num_queries=3,
        diff_max=0.05,
        num_frames=3,
        crlb_matrix=_crlb_matrix(),
    )
    explicit_outputs = hungarian_matched_loss(
        pred_classes,
        pred_positions,
        pred_h,
        pred_d,
        gt_classes,
        gt_positions,
        gt_h,
        gt_d,
        num_queries=3,
        diff_max=0.05,
        num_frames=3,
        crlb_matrix=_crlb_matrix(),
        match_h_weight=0.5,
        match_d_weight=0.5,
        loss_h_weight=0.5,
        loss_d_weight=0.5,
    )

    for default_value, explicit_value in zip(default_outputs, explicit_outputs):
        torch.testing.assert_close(default_value, explicit_value)


def test_hungarian_loss_can_remove_h_d_from_matching_but_keep_final_losses():
    pred_classes = torch.tensor([[[0.95, 0.95, 0.05], [0.05, 0.05, 0.05], [0.1, 0.1, 0.1]]], requires_grad=True)
    pred_positions = torch.zeros((1, 3, 3, 2), requires_grad=True)
    pred_h = torch.tensor([[0.1, 0.2, 0.7]], requires_grad=True)
    pred_d = torch.tensor([[0.1, 0.2, 0.7]], requires_grad=True)
    gt_classes = torch.tensor([[[1.0], [1.0], [0.0]]])
    gt_positions = torch.zeros((1, 3, 1, 2))
    gt_positions[:, 2, :, :] = float("nan")
    gt_h = torch.tensor([[0.8]])
    gt_d = torch.tensor([[0.8]])

    loss, _, _, h_loss, d_loss, _ = hungarian_matched_loss(
        pred_classes,
        pred_positions,
        pred_h,
        pred_d,
        gt_classes,
        gt_positions,
        gt_h,
        gt_d,
        num_queries=3,
        diff_max=0.05,
        num_frames=3,
        crlb_matrix=_crlb_matrix(),
        match_h_weight=0.0,
        match_d_weight=0.0,
        loss_h_weight=1.0,
        loss_d_weight=1.0,
    )

    assert torch.isfinite(loss)
    assert h_loss > 0
    assert d_loss > 0
    loss.backward()
    assert pred_h.grad.abs().sum() > 0
    assert pred_d.grad.abs().sum() > 0


def test_hungarian_loss_relative_diffusion_handles_near_zero_targets():
    pred_classes = torch.full((1, 3, 3), 0.8, requires_grad=True)
    pred_positions = torch.zeros((1, 3, 3, 2), requires_grad=True)
    pred_h = torch.full((1, 3), 0.5, requires_grad=True)
    pred_d = torch.full((1, 3), 0.5, requires_grad=True)
    gt_classes = torch.tensor([[[1.0], [1.0], [0.0]]])
    gt_positions = torch.zeros((1, 3, 1, 2))
    gt_positions[:, 2, :, :] = float("nan")
    gt_h = torch.tensor([[0.5]])
    gt_d = torch.tensor([[0.0]])

    loss, *components = hungarian_matched_loss(
        pred_classes,
        pred_positions,
        pred_h,
        pred_d,
        gt_classes,
        gt_positions,
        gt_h,
        gt_d,
        num_queries=3,
        diff_max=0.05,
        num_frames=3,
        crlb_matrix=_crlb_matrix(),
        diffusion_loss="relative",
        relative_d_eps=0.01,
    )

    assert torch.isfinite(loss)
    assert all(torch.isfinite(component) if torch.is_tensor(component) else np.isfinite(component) for component in components)
    loss.backward()
    assert pred_d.grad is not None


def test_hungarian_loss_log_hurst_and_diffusion_handles_near_zero_targets():
    pred_classes = torch.full((1, 3, 3), 0.8, requires_grad=True)
    pred_positions = torch.zeros((1, 3, 3, 2), requires_grad=True)
    pred_h = torch.full((1, 3), 0.5, requires_grad=True)
    pred_d = torch.full((1, 3), 0.5, requires_grad=True)
    gt_classes = torch.tensor([[[1.0], [1.0], [0.0]]])
    gt_positions = torch.zeros((1, 3, 1, 2))
    gt_positions[:, 2, :, :] = float("nan")
    gt_h = torch.tensor([[0.0]])
    gt_d = torch.tensor([[0.0]])

    loss, _, _, h_loss, d_loss, _ = hungarian_matched_loss(
        pred_classes,
        pred_positions,
        pred_h,
        pred_d,
        gt_classes,
        gt_positions,
        gt_h,
        gt_d,
        num_queries=3,
        diff_max=0.05,
        num_frames=3,
        crlb_matrix=_crlb_matrix(),
        hurst_loss="log",
        diffusion_loss="log",
        log_h_eps=0.01,
        log_d_eps=0.01,
    )

    assert torch.isfinite(loss)
    assert np.isfinite(h_loss)
    assert np.isfinite(d_loss)
    loss.backward()
    assert pred_h.grad is not None
    assert pred_d.grad is not None


def test_hungarian_loss_accepts_objectness_logits():
    pred_classes = torch.tensor([[[3.0, 3.0, -3.0], [-3.0, -3.0, -3.0], [-2.0, -2.0, -2.0]]], requires_grad=True)
    pred_positions = torch.zeros((1, 3, 3, 2), requires_grad=True)
    pred_h = torch.tensor([[0.4, 0.2, 0.7]], requires_grad=True)
    pred_d = torch.tensor([[0.4, 0.2, 0.7]], requires_grad=True)
    gt_classes = torch.tensor([[[1.0], [1.0], [0.0]]])
    gt_positions = torch.zeros((1, 3, 1, 2))
    gt_positions[:, 2, :, :] = float("nan")
    gt_h = torch.tensor([[0.4]])
    gt_d = torch.tensor([[0.4]])

    loss, *_ = hungarian_matched_loss(
        pred_classes,
        pred_positions,
        pred_h,
        pred_d,
        gt_classes,
        gt_positions,
        gt_h,
        gt_d,
        num_queries=3,
        diff_max=0.05,
        num_frames=3,
        crlb_matrix=_crlb_matrix(),
        objectness_loss="bce_logits",
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert pred_classes.grad is not None
    assert pred_classes.grad.abs().sum() > 0


def test_hungarian_loss_rejects_query_count_mismatches():
    with pytest.raises(ValueError, match="Expected 4 queries"):
        hungarian_matched_loss(
            torch.zeros((1, 3, 2)),
            torch.zeros((1, 3, 2, 2)),
            torch.zeros((1, 3)),
            torch.zeros((1, 3)),
            torch.zeros((1, 2, 1)),
            torch.zeros((1, 2, 1, 2)),
            torch.zeros((1, 1)),
            torch.zeros((1, 1)),
            num_queries=4,
            diff_max=0.05,
            num_frames=2,
            crlb_matrix=_crlb_matrix(frames=2),
        )
