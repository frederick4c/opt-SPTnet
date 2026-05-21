import pytest

torch = pytest.importorskip("torch")

from sptnet.models.backbone import BackBone, ResidualBlock
from sptnet.models.transformers import Transformer, Transformer3d


def test_residual_block_preserves_shape_when_channels_match():
    block = ResidualBlock(4, 4)
    x = torch.randn(2, 4, 3, 5, 5)

    y = block(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_backbone_preserves_time_and_downsamples_spatial_dimensions():
    model = BackBone()
    x = torch.randn(1, 1, 3, 16, 16)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 256, 3, 1, 1)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("transformer_cls,src_shape,mask_shape", [
    (Transformer, (2, 4, 3, 5), (2, 3, 5)),
    (Transformer3d, (2, 4, 2, 3, 5), (2, 2, 3, 5)),
])
def test_flattened_transformers_return_decoder_states_and_memory(transformer_cls, src_shape, mask_shape):
    transformer = transformer_cls(
        d_model=4,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=8,
        dropout=0,
    )
    src = torch.randn(*src_shape)
    mask = torch.zeros(mask_shape, dtype=torch.bool)
    query_embed = torch.randn(6, 4)
    pos_embed = torch.zeros_like(src)

    hs, memory = transformer(src, mask, query_embed, pos_embed)

    assert hs.shape == (1, src_shape[0], 6, 4)
    assert memory.shape == src.shape
    assert torch.isfinite(hs).all()
    assert torch.isfinite(memory).all()


def test_flattened_transformer_validates_shape_mismatches():
    transformer = Transformer(d_model=4, nhead=2, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=8)
    src = torch.randn(2, 4, 3, 5)
    mask = torch.zeros(2, 3, 5, dtype=torch.bool)
    query_embed = torch.randn(6, 5)
    pos_embed = torch.zeros_like(src)

    with pytest.raises(ValueError, match="query_embed"):
        transformer(src, mask, query_embed, pos_embed)
