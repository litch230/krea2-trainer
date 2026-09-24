from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from safetensors import safe_open

from krea2_train_network import Krea2NetworkTrainer
from library import strategy_krea2


class DummyAccelerator:
    device = torch.device("cpu")


class DummyConditioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(1), requires_grad=False)


def _args(model_path):
    return SimpleNamespace(
        cache_text_encoder_outputs=True,
        qwen3_vl=str(model_path),
        text_encoder=None,
    )


def test_complete_krea2_text_cache_does_not_load_qwen(monkeypatch, tmp_path):
    model_path = tmp_path / "qwen.safetensors"
    model_path.touch()
    calls = []

    class CompleteCacheDataset:
        def new_cache_text_encoder_outputs(self, models, accelerator):
            calls.append(list(models))

    monkeypatch.setattr(
        "krea2_train_network.krea2_utils.load_krea2_text_encoder",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Qwen should not be loaded")),
    )

    Krea2NetworkTrainer().cache_text_encoder_outputs_if_needed(
        _args(model_path), DummyAccelerator(), None, None, [], CompleteCacheDataset(), torch.bfloat16
    )

    assert calls == [[]]


def test_missing_krea2_text_cache_loads_qwen_and_retries(monkeypatch, tmp_path):
    model_path = tmp_path / "qwen.safetensors"
    model_path.touch()
    conditioner = DummyConditioner()
    calls = []

    class MissingCacheDataset:
        def new_cache_text_encoder_outputs(self, models, accelerator):
            calls.append(list(models))
            if not models:
                raise strategy_krea2.Krea2TextEncoderRequiredError("cache miss")

    monkeypatch.setattr(
        "krea2_train_network.krea2_utils.load_krea2_text_encoder",
        lambda path, dtype, device: conditioner,
    )

    Krea2NetworkTrainer().cache_text_encoder_outputs_if_needed(
        _args(model_path), DummyAccelerator(), None, None, [], MissingCacheDataset(), torch.bfloat16
    )

    assert calls[0] == []
    assert calls[1] == [conditioner]


def test_krea2_text_encoding_reports_missing_encoder_clearly():
    strategy = strategy_krea2.Krea2TextEncodingStrategy()

    try:
        strategy.encode_tokens(None, [], (torch.ones(1, 2), torch.ones(1, 2)))
    except strategy_krea2.Krea2TextEncoderRequiredError as error:
        assert "cache is missing" in str(error)
    else:
        raise AssertionError("Expected missing encoder error")


def test_krea2_text_cache_batches_and_saves_scaled_fp8(tmp_path):
    caching = strategy_krea2.Krea2TextEncoderOutputsCachingStrategy(
        cache_to_disk=True, batch_size=4, max_token_length=256
    )
    tokenizer = object.__new__(strategy_krea2.Krea2TokenizeStrategy)
    tokenizer.tokenize = lambda captions: (
        torch.ones(len(captions), 5, dtype=torch.long),
        torch.ones(len(captions), 5, dtype=torch.long),
    )
    encoding = strategy_krea2.Krea2TextEncodingStrategy()
    calls = []
    expected = torch.linspace(-3, 3, 2 * 5 * 12 * 16).reshape(2, 5, 12, 16)

    def encode_batch(_tokenizer, _models, _tokens):
        calls.append(True)
        return expected, torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)

    encoding.encode_tokens = encode_batch
    infos = [
        SimpleNamespace(caption="first", text_encoder_outputs_npz=str(tmp_path / "first.safetensors")),
        SimpleNamespace(caption="second", text_encoder_outputs_npz=str(tmp_path / "second.safetensors")),
    ]
    conditioner = DummyConditioner()

    caching.cache_batch_outputs(tokenizer, [conditioner], encoding, infos)

    assert len(calls) == 1
    with safe_open(str(tmp_path / "second.safetensors"), framework="pt") as cache:
        assert cache.get_tensor("hidden_state").dtype == torch.float8_e4m3fn
        assert cache.metadata()["format"] == "scaled_fp8_v3"
        assert cache.metadata()["max_token_length"] == "256"
    restored, mask = caching.load_outputs_npz(str(tmp_path / "second.safetensors"))
    assert restored.shape == (5, 12, 16)
    assert np.max(np.abs(restored - expected[1].numpy())) < 0.08
    assert mask.tolist() == [1, 1, 1, 1, 1]


def test_krea2_scaled_fp8_preserves_hidden_states_with_large_scales(tmp_path):
    caching = strategy_krea2.Krea2TextEncoderOutputsCachingStrategy(
        cache_to_disk=True, batch_size=1, max_token_length=256
    )
    tokenizer = object.__new__(strategy_krea2.Krea2TokenizeStrategy)
    tokenizer.tokenize = lambda captions: (
        torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2, dtype=torch.long)
    )
    encoding = strategy_krea2.Krea2TextEncodingStrategy()
    hidden = torch.full((1, 2, 12, 16), 4.0e7, dtype=torch.float32)
    encoding.encode_tokens = lambda *_args: (hidden, torch.ones(1, 2, dtype=torch.bool))
    info = SimpleNamespace(caption="large", text_encoder_outputs_npz=str(tmp_path / "large.safetensors"))

    caching.cache_batch_outputs(tokenizer, [DummyConditioner()], encoding, [info])
    restored, _ = caching.load_outputs_npz(str(tmp_path / "large.safetensors"))

    assert np.isfinite(restored).all()
    assert np.max(np.abs(restored - hidden[0].numpy())) < 1.0e5


def test_krea2_text_cache_token_length_change_invalidates_old_cache(tmp_path):
    cache_path = tmp_path / "caption.safetensors"
    from safetensors.torch import save_file

    save_file(
        {"hidden_state": torch.zeros(2, 12, 16), "attention_mask": torch.ones(2, dtype=torch.int32)},
        str(cache_path),
        metadata={"max_token_length": "512"},
    )

    strategy_256 = strategy_krea2.Krea2TextEncoderOutputsCachingStrategy(
        cache_to_disk=True, skip_disk_cache_validity_check=True, max_token_length=256
    )
    assert strategy_256.is_disk_cached_outputs_expected(str(cache_path)) is False
