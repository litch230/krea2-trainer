from types import SimpleNamespace
from contextlib import nullcontext

import torch
from PIL import Image

from krea2_train_network import Krea2NetworkTrainer
from library.krea2 import krea2_sampling


class _Accelerator:
    device = torch.device("cpu")

    def unwrap_model(self, model):
        return model

    def autocast(self):
        return nullcontext()


class _FlowProbeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(patch=2)
        self.received_img = None

    def forward(self, img, context, t, pos, mask):
        self.received_img = img.detach().clone()
        return torch.zeros_like(img)


def test_krea2_trainer_uses_native_sampler_and_saves_preview(tmp_path, monkeypatch):
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("test character --w 512 --h 512 --s 1 --d 7 --l 1\n", encoding="utf-8")
    args = SimpleNamespace(
        sample_at_first=False,
        sample_every_n_epochs=1,
        sample_every_n_steps=None,
        sample_prompts=str(prompt_file),
        output_dir=str(tmp_path),
        output_name="krea-test",
    )
    trainer = Krea2NetworkTrainer()
    trainer.sample_prompts_te_outputs = {
        "test character": (
            torch.zeros(1, 4, 12, 16),
            torch.ones(1, 4, dtype=torch.bool),
        )
    }
    model = torch.nn.Linear(1, 1)
    vae = torch.nn.Linear(1, 1)
    calls = []

    def fake_sample(*sample_args, **sample_kwargs):
        calls.append((sample_args, sample_kwargs))
        return [Image.new("RGB", (8, 8), "white")]

    monkeypatch.setattr(krea2_sampling, "sample", fake_sample)
    trainer.sample_images(_Accelerator(), args, 1, 10, torch.device("cpu"), vae, [], [], model)

    assert len(calls) == 1
    assert calls[0][1]["steps"] == 1
    assert calls[0][1]["seed"] == 7
    assert (tmp_path / "sample" / "krea-test_e000001_00_00.png").is_file()


def test_krea2_training_uses_rectified_flow_velocity_target(monkeypatch):
    trainer = Krea2NetworkTrainer()
    accelerator = _Accelerator()
    model = _FlowProbeModel()
    latents = torch.full((1, 16, 1, 4, 4), 0.25)
    fixed_noise = torch.full_like(latents, -0.75)
    monkeypatch.setattr(torch, "randn_like", lambda value, **kwargs: fixed_noise.clone())
    args = SimpleNamespace(
        timestep_sample_method="uniform",
        sigmoid_scale=1.0,
        discrete_flow_shift=1.0,
        adaptive_sigmoid=False,
        sigmoid_bias=0.0,
        sigmoid_mix=1.0,
        ip_noise_gamma=0.0,
        gradient_checkpointing=False,
    )
    text_conds = (
        torch.zeros(1, 2, 1, 8),
        torch.ones(1, 2, dtype=torch.bool),
    )

    prediction, target, timesteps, weighting = trainer.get_noise_pred_and_target(
        args,
        accelerator,
        None,
        latents,
        {},
        text_conds,
        model,
        None,
        torch.float32,
        True,
    )

    torch.testing.assert_close(target, fixed_noise - latents)
    assert prediction.shape == latents.shape
    assert weighting is None
    assert 0.0 < timesteps.item() < 1000.0

    t = timesteps.item() / 1000.0
    expected_noisy = (1.0 - t) * latents.squeeze(2) + t * fixed_noise.squeeze(2)
    from einops import rearrange
    received_noisy = rearrange(
        model.received_img,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=2,
        ph=2,
        pw=2,
    )
    torch.testing.assert_close(received_noisy, expected_noisy)
