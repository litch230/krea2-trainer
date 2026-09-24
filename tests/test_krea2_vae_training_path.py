import torch
import types
from torch import nn
from safetensors import safe_open

from krea2_train_network import Krea2NetworkTrainer
from library.krea2.krea2_vae import AutoencoderKLQwenImage, Upsample3d, convert_comfyui_state_dict
from library.strategy_krea2 import Krea2LatentsCachingStrategy


class RecordingKrea2VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.received = None

    def encode_pixels_to_latents(self, pixels):
        self.received = pixels
        return torch.ones(
            pixels.shape[0], 16, pixels.shape[2], pixels.shape[3] // 16, pixels.shape[4] // 16,
            device=pixels.device,
            dtype=pixels.dtype,
        )


def test_krea2_uncached_images_gain_a_frame_axis_before_vae_encoding():
    trainer = Krea2NetworkTrainer()
    vae = RecordingKrea2VAE()
    images = torch.randn(2, 3, 64, 48)

    latents = trainer.encode_images_to_latents(None, vae, images)

    assert vae.received.shape == (2, 3, 1, 64, 48)
    assert latents.shape == (2, 16, 1, 4, 3)


def test_krea2_does_not_apply_sd_latent_scaling():
    trainer = Krea2NetworkTrainer()
    latents = torch.randn(2, 16, 1, 4, 3)

    assert trainer.shift_scale_latents(None, latents) is latents


def test_krea2_vae_encoding_uses_posterior_mean_not_a_random_sample(monkeypatch):
    vae = object.__new__(AutoencoderKLQwenImage)
    nn.Module.__init__(vae)
    vae.encoder = nn.Module()
    vae.encoder.conv_in = nn.Conv3d(1, 1, 1, bias=False)
    vae.latents_mean = None
    vae.latents_std = None
    distribution = types.SimpleNamespace(mean=torch.full((1, 1, 1, 1, 1), 3.0))
    distribution.sample = lambda: (_ for _ in ()).throw(AssertionError("posterior sample must not be used"))
    vae.encode = types.MethodType(lambda self, pixels: distribution, vae)

    result = vae.encode_pixels_to_latents(torch.zeros(1, 1, 1, 1, 1))

    torch.testing.assert_close(result, distribution.mean)


def test_krea2_decode_to_pixels_denormalizes_and_returns_unit_rgb():
    vae = object.__new__(AutoencoderKLQwenImage)
    nn.Module.__init__(vae)
    vae.decoder = nn.Module()
    vae.decoder.conv_in = nn.Conv3d(3, 3, 1, bias=False)
    vae.register_buffer("latents_mean", torch.zeros(1, 3, 1, 1, 1))
    vae.register_buffer("latents_std", torch.ones(1, 3, 1, 1, 1))
    vae.decode = types.MethodType(lambda self, z: z, vae)

    latents = torch.tensor([-1.0, 0.0, 1.0]).view(1, 3, 1, 1, 1)
    pixels = vae.decode_to_pixels(latents)

    assert pixels.shape == (1, 3, 1, 1)
    torch.testing.assert_close(pixels.flatten(), torch.tensor([0.0, 0.5, 1.0]))


def test_krea2_image_upsample_preserves_single_frame_axis():
    upsample = Upsample3d(dim=4, temperal_upsample=True)
    x = torch.randn(1, 4, 1, 8, 8)

    out = upsample(x)

    assert out.shape == (1, 2, 1, 16, 16)


def test_krea2_comfy_vae_block_names_are_converted_to_local_layout():
    marker = torch.ones(1)
    converted = convert_comfyui_state_dict(
        {
            "conv1.bias": marker,
            "encoder.downsamples.5.resample.1.weight": marker,
            "encoder.middle.1.proj.weight": marker,
            "decoder.upsamples.7.resample.1.bias": marker,
            "decoder.upsamples.7.time_conv.weight": marker,
        }
    )

    assert set(converted) == {
        "quant_conv.bias",
        "encoder.down_blocks.5.resample.weight",
        "encoder.mid_block.1.proj.weight",
        "decoder.up_blocks.7.resample.bias",
        "decoder.up_blocks.7.time_conv.weight",
    }


def test_krea2_latent_cache_is_written_and_reusable(tmp_path):
    strategy = Krea2LatentsCachingStrategy(
        cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=False
    )
    npz_path = tmp_path / "image_0064x0048_krea2.npz"
    latents = torch.randn(16, 1, 3, 4)

    strategy.save_latents_to_disk(
        str(npz_path), latents, [64, 48], [0, 0, 64, 48], key_reso_suffix="_3x4"
    )

    cache_path = npz_path.with_suffix(".safetensors")
    assert cache_path.exists()
    with safe_open(str(cache_path), framework="pt") as cache:
        assert cache.get_tensor("latents_3x4").shape == latents.shape
    assert strategy.is_disk_cached_latents_expected(
        (64, 48), str(npz_path), flip_aug=False, alpha_mask=False
    ) is True
