"""Loss functions for SPTnet training."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def hungarian_matched_loss(
    pred_classes,
    pred_positions,
    pred_h,
    pred_d,
    gt_classes,
    gt_positions,
    gt_h,
    gt_d,
    *,
    num_queries,
    diff_max,
    num_frames,
    crlb_matrix,
):
    """Compute SPTnet's Hungarian-matched multi-object training loss.

    The loss matches predicted query slots to ground-truth trajectories using a
    cost that combines object confidence, coordinate error, Hurst error, and
    diffusion-coefficient error. Unmatched query slots are penalized as
    background.

    Parameters
    ----------
    pred_classes:
        Predicted object probabilities shaped `[B, Q, T]`.
    pred_positions:
        Predicted normalized coordinates shaped `[B, Q, T, 2]`.
    pred_h, pred_d:
        Predicted Hurst and diffusion values shaped `[B, Q]`.
    gt_classes:
        Ground-truth activity labels shaped `[B, T, N]`.
    gt_positions:
        Ground-truth coordinates shaped `[B, T, N, 2]`.
    gt_h, gt_d:
        Ground-truth Hurst and diffusion labels shaped `[B, N]`.
    num_queries:
        Number of model query slots.
    diff_max:
        Maximum diffusion coefficient used for label normalization.
    num_frames:
        Number of movie frames used to index the CRLB weighting matrix.
    crlb_matrix:
        Precomputed CRLB matrix used to weight Hurst/diffusion losses.
    """
    num_batches, num_queries_from_pred, num_frames_from_pred = pred_classes.shape
    if num_queries_from_pred != num_queries:
        raise ValueError(f"Expected {num_queries} queries, got {num_queries_from_pred}.")

    pred_classes = torch.nan_to_num(pred_classes, nan=0.5, posinf=1.0, neginf=0.0).clamp(1e-6, 1 - 1e-6)
    pred_positions = torch.nan_to_num(pred_positions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    pred_h = torch.nan_to_num(pred_h, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    pred_d = torch.nan_to_num(pred_d, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    gt_classes = torch.nan_to_num(gt_classes, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)

    loss_pb = 0
    total_class_pb = 0
    total_coordi_pb = 0
    total_hurst_pb = 0
    total_diffusion_pb = 0
    non_obj_loss_all = 0
    fullindex = np.arange(num_queries)
    gt_positions = gt_positions.permute(0, 2, 1, 3)

    if num_queries <= gt_h.shape[1]:
        raise ValueError(
            f"Number of queries ({num_queries}) must be greater than number of particles ({gt_h.shape[1]}). "
            "Please increase num_queries to be > max number of ground truth particles."
        )

    zeros_pd = torch.zeros(num_batches, num_queries - gt_h.shape[1], device=gt_h.device, dtype=gt_h.dtype)
    gt_h = torch.cat((gt_h, zeros_pd), dim=1)
    gt_d = torch.cat((gt_d, zeros_pd), dim=1)

    criterion_mae = torch.nn.L1Loss(reduction="none").to(pred_classes.device)
    pdist = torch.nn.PairwiseDistance(p=2)

    for b in range(num_batches):
        track_flag = sum(gt_classes[b, :]) >= 2
        num_tracks = int(sum(track_flag))
        if num_tracks != 0:
            gt_pos_track = gt_positions[b, :, :, :][track_flag, :, :].unsqueeze(0).repeat(num_queries, 1, 1, 1)
            gt_classes_pm = gt_classes[b, :][:, track_flag].permute(1, 0)
            class_loss_matrix = F.binary_cross_entropy(
                pred_classes[b, :, :].view(num_queries, 1, num_frames_from_pred).repeat(1, num_tracks, 1),
                gt_classes_pm.view(1, num_tracks, num_frames_from_pred).repeat(num_queries, 1, 1),
                reduction="none",
            )
            nan_mask = torch.isnan(gt_pos_track)
            gt_pos_track[nan_mask] = 0
            pred_masked = pred_positions[b, :, :, :].unsqueeze(1).repeat(1, num_tracks, 1, 1)
            pred_masked[nan_mask] = 0
            pos_loss_matrix = pdist(pred_masked, gt_pos_track)
            pos_loss_matrix = torch.nansum(pos_loss_matrix, dim=2)
            cost_matrix_class_pf = torch.mean(class_loss_matrix, dim=2)
            duration = sum(gt_classes[b, :])[track_flag]
            pos_loss_matrix_allfrm_pf = pos_loss_matrix / duration

            gt_h_nonzero = gt_h[b][track_flag]
            gt_d_nonzero = gt_d[b][track_flag]
            h_idx = torch.clamp((gt_h_nonzero * 100).round() - 1, min=0, max=98).cpu().numpy().astype(int)
            d_idx = (
                torch.clamp((gt_d_nonzero * diff_max * 100).round() - 1, min=0, max=diff_max * 100 - 1)
                .cpu()
                .numpy()
                .astype(int)
            )
            stepidx = duration.cpu().numpy().astype(int) - 1
            crlbweight_h = crlb_matrix[0, 0, d_idx, h_idx, stepidx] / (
                crlb_matrix[0, 0, d_idx, h_idx, num_frames - 1] + 1e-8
            )
            crlbweight_d = crlb_matrix[1, 1, d_idx, h_idx, stepidx] / (
                crlb_matrix[1, 1, d_idx, h_idx, num_frames - 1] + 1e-8
            )
            crlbweight_h = torch.as_tensor(crlbweight_h, device=pred_h.device, dtype=pred_h.dtype)
            crlbweight_d = torch.as_tensor(crlbweight_d, device=pred_d.device, dtype=pred_d.dtype)
            crlbweight_h = torch.nan_to_num(crlbweight_h, nan=1.0, posinf=1.0, neginf=1.0).clamp(min=1e-4)
            crlbweight_d = torch.nan_to_num(crlbweight_d, nan=1.0, posinf=1.0, neginf=1.0).clamp(min=1e-4)

            h_loss_matrix = criterion_mae(
                pred_h[b].view(-1, 1).repeat(1, gt_h_nonzero.shape[-1]),
                gt_h_nonzero.view(1, -1).repeat(pred_h.shape[-1], 1),
            ) / crlbweight_h.repeat(pred_h.shape[-1], 1)
            d_loss_matrix = criterion_mae(
                pred_d[b].view(-1, 1).repeat(1, gt_d_nonzero.shape[-1]),
                gt_d_nonzero.view(1, -1).repeat(pred_d.shape[-1], 1),
            ) / crlbweight_d.repeat(pred_h.shape[-1], 1)
            cost_matrix_all_pf = (
                cost_matrix_class_pf
                + 2 * pos_loss_matrix_allfrm_pf
                + 0.5 * h_loss_matrix
                + 0.5 * d_loss_matrix
            ).t()
            cost_matrix_safe = torch.nan_to_num(cost_matrix_all_pf, nan=1e4, posinf=1e4, neginf=1e4)

            row_indices, col_indices = linear_sum_assignment(cost_matrix_safe.cpu().detach().numpy())
            cost_matrix_all_pf = cost_matrix_safe[row_indices, col_indices].sum()

            total_class = (
                torch.nan_to_num(cost_matrix_class_pf.t(), nan=0.0, posinf=1e4, neginf=0.0)
                .cpu()
                .detach()
                .numpy()[row_indices, col_indices]
                .sum()
            ) / num_tracks
            total_coordi = (
                2
                * torch.nan_to_num(pos_loss_matrix_allfrm_pf.t(), nan=0.0, posinf=1e4, neginf=0.0)
                .cpu()
                .detach()
                .numpy()[row_indices, col_indices]
                .sum()
            ) / num_tracks
            total_hurst = (
                0.5
                * torch.nan_to_num(h_loss_matrix.t(), nan=0.0, posinf=1e4, neginf=0.0)
                .cpu()
                .detach()
                .numpy()[row_indices, col_indices]
                .sum()
            ) / num_tracks
            total_diffusion = (
                0.5
                * torch.nan_to_num(d_loss_matrix.t(), nan=0.0, posinf=1e4, neginf=0.0)
                .cpu()
                .detach()
                .numpy()[row_indices, col_indices]
                .sum()
            ) / num_tracks

            non_obj_pre = pred_classes[b, :, :][np.setdiff1d(fullindex, col_indices), :]
            non_obj_pre = torch.nan_to_num(non_obj_pre, nan=0.5, posinf=1.0, neginf=0.0).clamp(1e-6, 1 - 1e-6)
            non_obj_loss = F.binary_cross_entropy(non_obj_pre, torch.zeros_like(non_obj_pre), reduction="mean")
            loss_pv = (cost_matrix_all_pf / num_tracks) + non_obj_loss
            loss_pb += loss_pv
            if torch.isnan(loss_pv):
                print("Tracks", num_tracks)
                continue
        else:
            non_obj_pre = pred_classes[b, :, :]
            non_obj_pre = torch.nan_to_num(non_obj_pre, nan=0.5, posinf=1.0, neginf=0.0).clamp(1e-6, 1 - 1e-6)
            non_obj_loss = F.binary_cross_entropy(non_obj_pre, torch.zeros_like(non_obj_pre), reduction="mean")
            loss_pb += non_obj_loss
            total_class = 0
            total_coordi = 0
            total_hurst = 0
            total_diffusion = 0

        non_obj_loss_all += non_obj_loss
        total_class_pb += total_class
        total_coordi_pb += total_coordi
        total_hurst_pb += total_hurst
        total_diffusion_pb += total_diffusion

    return (
        loss_pb / num_batches,
        total_class_pb / num_batches,
        total_coordi_pb / num_batches,
        total_hurst_pb / num_batches,
        total_diffusion_pb / num_batches,
        non_obj_loss_all / num_batches,
    )
