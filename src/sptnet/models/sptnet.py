"""SPTnet model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encodings.torch_encodings import (
    PositionalEncodingPermute2D,
    PositionalEncodingPermute3D,
)

from sptnet.models.backbone import BackBone


class SPTnet(nn.Module):
    """SPTnet detector/regressor model.

    The model combines a 3D convolutional backbone with spatial and temporal
    DETR-style transformer decoders. It predicts per-query object confidence,
    normalized xy coordinates, Hurst exponent, and diffusion coefficient.

    Parameters
    ----------
    num_classes:
        Retained for compatibility with the original constructor; the current
        model uses a single object-confidence head.
    num_queries:
        Number of trajectory query slots decoded by the transformer.
    num_frames:
        Number of frames expected in each input movie.
    spatial_t:
        Transformer used over per-frame spatial feature maps.
    temporal_t:
        Transformer used over the full spatiotemporal feature volume.
    input_channel:
        Retained for compatibility with the original constructor.
    """

    def __init__(self, num_classes, num_queries, num_frames, spatial_t, temporal_t, input_channel):
        super().__init__()
        self.input_channel = input_channel
        self.backbone = BackBone()
        self.conv_temp = nn.Conv1d(1, num_frames, kernel_size=1, stride=1, padding=0)

        d_model = temporal_t.d_model
        self.num_queries = num_queries
        self.transformer = spatial_t
        self.transformer3d = temporal_t
        self.query_embed = nn.Embedding(num_queries, d_model)

        self.fc1 = nn.Linear(256, 32)
        self.fc1_1 = nn.Linear(32, 2)
        self.fc2 = nn.Linear(256, 1)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):
        """Run SPTnet on normalized videos.

        Parameters
        ----------
        x:
            Tensor shaped `[B, 1, T, H, W]`.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            `(class_logits, center_pred, h_est, d_est)` with shapes
            `[B, Q, T]`, `[B, Q, T, 2]`, `[B, Q, 1]`, and `[B, Q, 1]`.
        """
        features = F.relu(self.backbone(x))
        batch_size, channels, num_frames, height, width = features.shape
        device = features.device

        pos = PositionalEncodingPermute3D(channels).to(device)(features)
        sp_pos_encoder = PositionalEncodingPermute2D(channels).to(device)

        queries = self.query_embed.weight
        mask = torch.zeros((batch_size, num_frames, height, width), dtype=torch.bool, device=device)
        sp_mask = torch.zeros((batch_size * num_frames, height, width), dtype=torch.bool, device=device)

        sp_features = features.permute(0, 2, 1, 3, 4).flatten(0, 1)
        sp_pos = sp_pos_encoder(sp_features)
        sp_hs = self.transformer(sp_features, sp_mask, queries, sp_pos)[0]
        sp_hs = sp_hs.view(batch_size, num_frames, self.num_queries, channels)

        hs1 = self.transformer3d(features, mask, queries, pos)[0]
        ts_hs = hs1.permute(1, 2, 0, 3).flatten(0, 1)
        ts_hs = self.conv_temp(ts_hs)
        ts_hs = ts_hs.view(batch_size, num_frames, self.num_queries, -1)

        deco_comb = ts_hs + sp_hs

        center_pred = F.relu(self.fc1(deco_comb))
        center_pred = torch.tanh(self.fc1_1(center_pred))
        center_pred = center_pred.permute(0, 2, 1, 3)

        class_logits = torch.sigmoid(self.fc2(deco_comb)).squeeze(-1)
        class_logits = class_logits.permute(0, 2, 1)

        xf_hd = F.relu(self.fc3(hs1.squeeze(0)))
        xf_hd = torch.sigmoid(self.fc4(xf_hd))
        h_est = xf_hd[:, :, 0].unsqueeze(-1)
        d_est = xf_hd[:, :, 1].unsqueeze(-1)

        return class_logits, center_pred, h_est, d_est
