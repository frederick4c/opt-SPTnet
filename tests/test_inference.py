import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sptnet.inference.predict import (
    extract_state_dict,
    get_num_frames,
    get_num_queries,
    load_checkpoint_strict_enough,
    normalize_state_dict_keys,
    normalize_video_batch,
    run_batched_inference,
)


def test_checkpoint_metadata_helpers_accept_wrapped_or_plain_state_dict():
    state_dict = {
        "module.query_embed.weight": torch.zeros(7, 256),
        "module.conv_temp.weight": torch.zeros(11, 1, 1),
    }

    assert extract_state_dict({"state_dict": state_dict}) is state_dict
    assert get_num_queries(state_dict=state_dict) == 7
    assert get_num_frames(state_dict=state_dict) == 11


def test_checkpoint_metadata_helpers_raise_for_missing_keys():
    with pytest.raises(ValueError, match="query_embed.weight"):
        get_num_queries(state_dict={"other": torch.zeros(1)})
    with pytest.raises(ValueError, match="conv_temp.weight"):
        get_num_frames(state_dict={"other": torch.zeros(1)})


def test_normalize_state_dict_keys_strips_dataparallel_prefix():
    model = torch.nn.Linear(2, 1)
    state_dict = {f"module.{key}": value.clone() for key, value in model.state_dict().items()}

    normalized = normalize_state_dict_keys(state_dict, model)

    assert sorted(normalized) == sorted(model.state_dict())


def test_load_checkpoint_strict_enough_catches_incompatible_state_dict():
    model = torch.nn.Linear(2, 1)

    with pytest.raises(RuntimeError, match="No model weights"):
        load_checkpoint_strict_enough(model, state_dict={"unrelated": torch.zeros(1)})


def test_normalize_video_batch_scales_each_sample_and_handles_constant_inputs():
    inputs = torch.tensor(
        [
            [[[[2.0, 4.0], [6.0, 8.0]]]],
            [[[[5.0, 5.0], [5.0, 5.0]]]],
        ]
    )

    normalized = normalize_video_batch(inputs)

    torch.testing.assert_close(normalized[0].amin(), torch.tensor(0.0))
    torch.testing.assert_close(normalized[0].amax(), torch.tensor(1.0))
    torch.testing.assert_close(normalized[1], torch.zeros_like(normalized[1]))
    assert torch.isfinite(normalized).all()


def test_run_batched_inference_returns_one_record_per_sample():
    class TinyModel(torch.nn.Module):
        def forward(self, inputs):
            batch_size = inputs.shape[0]
            return (
                torch.full((batch_size, 2, 3), 0.25),
                torch.zeros((batch_size, 2, 3, 2)),
                torch.full((batch_size, 2, 1), 0.5),
                torch.full((batch_size, 2, 1), 0.75),
            )

    dataloader = [
        {
            "video": torch.from_numpy(np.ones((2, 3, 4, 4), dtype=np.float32)),
            "file_path": ["first.mat", "second.mat"],
            "sample_idx": [0, 1],
        }
    ]

    results = run_batched_inference(TinyModel(), dataloader, torch.device("cpu"))

    assert [record["file_path"] for record in results] == ["first.mat", "second.mat"]
    assert [record["sample_idx"] for record in results] == [0, 1]
    assert results[0]["obj_estimation"].shape == (1, 2, 3)
    assert results[0]["estimation_xy"].shape == (1, 2, 3, 2)
    assert results[0]["estimation_H"].shape == (2, 1)
    assert results[0]["estimation_C"].shape == (2, 1)
