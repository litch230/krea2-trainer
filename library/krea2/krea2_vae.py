"""Krea 2 / Qwen-Image AutoencoderKL implementation.
Copied from Qwen-Image / Wan VAE.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from library.safetensors_utils import load_safetensors

logger = logging.getLogger(__name__)

CACHE_T = 2

ACT2CLS = {
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}


def get_activation(act_fn: str) -> nn.Module:
    act_fn = act_fn.lower()
    if act_fn in ACT2CLS:
        return ACT2CLS[act_fn]()
    else:
        raise ValueError(f"activation function {act_fn} not found in ACT2FN mapping {list(ACT2CLS.keys())}")


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if self.deterministic:
            return self.mean
        if generator is not None:
            sample = torch.randn(self.mean.shape, generator=generator, device=self.mean.device, dtype=self.mean.dtype)
        else:
            sample = torch.randn_like(self.mean)
        return self.mean + self.std * sample


class CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x: torch.Tensor, cache_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)


class RMS_norm(nn.Module):
    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class ResnetBlock3d(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, feat_cache: Optional[List[torch.Tensor]] = None, feat_idx: Optional[List[int]] = None) -> torch.Tensor:
        res = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d):
                if feat_cache is not None and feat_idx is not None:
                    idx = feat_idx[0]
                    cache_x = feat_cache[idx]
                    feat_cache[idx] = x[:, :, -CACHE_T:].clone()
                    feat_idx[0] += 1
                else:
                    cache_x = None
                x = layer(x, cache_x)
            else:
                x = layer(x)
        return x + res


class AttentionBlock3d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.norm = RMS_norm(dim, images=False)
        self.to_qkv = CausalConv3d(dim, dim * 3, 1)
        self.proj = CausalConv3d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm(x)
        b, c, t, h, w = x.shape
        q, k, v = self.to_qkv(x).reshape(b, 3, c, t, h, w).unbind(1)

        q = q.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)
        k = k.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)
        v = v.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)
        return res + self.proj(out)


class Downsample3d(nn.Module):
    def __init__(self, dim: int, temperal_downsample: bool = True):
        super().__init__()
        self.temperal_downsample = temperal_downsample
        # Qwen/Wan performs spatial resampling independently for every frame.
        # The checkpoint therefore stores a 2D convolution here, not Conv3d.
        self.resample = nn.Conv2d(dim, dim, 3, stride=2)

    def forward(self, x: torch.Tensor, feat_cache: Optional[List[torch.Tensor]] = None, feat_idx: Optional[List[int]] = None) -> torch.Tensor:
        b, c, t, h, w = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = F.pad(x, (0, 1, 0, 1))
        x = self.resample(x)
        return rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)


class Upsample3d(nn.Module):
    def __init__(self, dim: int, temperal_upsample: bool = True):
        super().__init__()
        self.temperal_upsample = temperal_upsample
        self.resample = nn.Conv2d(dim, dim // 2, 3, padding=1)
        self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0)) if temperal_upsample else None

    def forward(self, x: torch.Tensor, feat_cache: Optional[List[torch.Tensor]] = None, feat_idx: Optional[List[int]] = None) -> torch.Tensor:
        b, c, t, h, w = x.shape
        # Qwen-Image uses the first-frame path of the causal video VAE.  A
        # single latent frame represents one image and must not be expanded in
        # time.  ComfyUI's cached decoder also skips temporal upsampling for
        # that first frame; applying both temporal stages here produced four
        # RGB frames, 4x decoder activations and severe VRAM paging.
        if self.temperal_upsample and t > 1:
            if feat_cache is not None and feat_idx is not None:
                idx = feat_idx[0]
                cache_x = feat_cache[idx]
                feat_cache[idx] = x[:, :, -CACHE_T:].clone()
                feat_idx[0] += 1
            else:
                cache_x = None
            x_time = self.time_conv(x, cache_x)
            x_time = x_time.reshape(b, 2, c, t, h, w).permute(0, 2, 3, 1, 4, 5).reshape(b, c, t * 2, h, w)
            x = x_time

        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = F.interpolate(x, scale_factor=(2, 2), mode="nearest-exact")
        x = self.resample(x)
        return rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)


class Encoder3d(nn.Module):
    def __init__(self, dim: int = 128, z_dim: int = 32, dim_mult: List[int] = [1, 2, 4, 4], num_res_blocks: int = 2, attn_scales: List[float] = [], temperal_downsample: List[bool] = [True, True, False], dropout: float = 0.0, input_channels: int = 3):
        super().__init__()
        self.conv_in = CausalConv3d(input_channels, dim, 3, padding=1)

        down_blocks = []
        cur_dim = dim
        for i, mult in enumerate(dim_mult):
            out_dim = dim * mult
            for _ in range(num_res_blocks):
                down_blocks.append(ResnetBlock3d(cur_dim, out_dim, dropout=dropout))
                cur_dim = out_dim
                if (1.0 / (2**i)) in attn_scales:
                    down_blocks.append(AttentionBlock3d(cur_dim))
            if i != len(dim_mult) - 1:
                down_blocks.append(Downsample3d(cur_dim, temperal_downsample[i]))
        self.down_blocks = nn.ModuleList(down_blocks)

        mid_block = [
            ResnetBlock3d(cur_dim, cur_dim, dropout=dropout),
            AttentionBlock3d(cur_dim),
            ResnetBlock3d(cur_dim, cur_dim, dropout=dropout),
        ]
        self.mid_block = nn.ModuleList(mid_block)

        self.norm_out = RMS_norm(cur_dim, images=False)
        self.conv_out = CausalConv3d(cur_dim, z_dim, 3, padding=1)

    def forward(self, x: torch.Tensor, feat_cache: Optional[List[torch.Tensor]] = None, feat_idx: Optional[List[int]] = None) -> torch.Tensor:
        # Image training always uses one frame and no streaming cache. Keep an
        # explicit path identical to WanVAE's non-cached forward; this also
        # avoids accidentally carrying cache state through image batches.
        if feat_cache is None:
            x = self.conv_in(x)
            for layer in self.down_blocks:
                x = layer(x)
            for layer in self.mid_block:
                x = layer(x)
            return self.conv_out(F.silu(self.norm_out(x)))

        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = feat_cache[idx]
            feat_cache[idx] = x[:, :, -CACHE_T:].clone()
            feat_idx[0] += 1
        else:
            cache_x = None

        x = self.conv_in(x, cache_x)
        for layer in self.down_blocks:
            if isinstance(layer, (ResnetBlock3d, Downsample3d)):
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        for layer in self.mid_block:
            if isinstance(layer, ResnetBlock3d):
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        x = F.silu(self.norm_out(x))
        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = feat_cache[idx]
            feat_cache[idx] = x[:, :, -CACHE_T:].clone()
            feat_idx[0] += 1
        else:
            cache_x = None
        return self.conv_out(x, cache_x)


class Decoder3d(nn.Module):
    def __init__(self, dim: int = 128, z_dim: int = 16, dim_mult: List[int] = [1, 2, 4, 4], num_res_blocks: int = 2, attn_scales: List[float] = [], temperal_upsample: List[bool] = [False, True, True], dropout: float = 0.0, output_channels: int = 3):
        super().__init__()
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        cur_dim = dims[0]
        self.conv_in = CausalConv3d(z_dim, cur_dim, 3, padding=1)

        mid_block = [
            ResnetBlock3d(cur_dim, cur_dim, dropout=dropout),
            AttentionBlock3d(cur_dim),
            ResnetBlock3d(cur_dim, cur_dim, dropout=dropout),
        ]
        self.mid_block = nn.ModuleList(mid_block)

        up_blocks = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i in (1, 2, 3):
                in_dim //= 2
            for _ in range(num_res_blocks + 1):
                up_blocks.append(ResnetBlock3d(in_dim, out_dim, dropout=dropout))
                in_dim = out_dim
                if (1.0 / (2**i)) in attn_scales:
                    up_blocks.append(AttentionBlock3d(out_dim))
            if i != len(dim_mult) - 1:
                up_blocks.append(Upsample3d(out_dim, temperal_upsample[i]))
            cur_dim = out_dim
        self.up_blocks = nn.ModuleList(up_blocks)

        self.norm_out = RMS_norm(cur_dim, images=False)
        self.conv_out = CausalConv3d(cur_dim, output_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, feat_cache: Optional[List[torch.Tensor]] = None, feat_idx: Optional[List[int]] = None) -> torch.Tensor:
        if feat_cache is None:
            x = self.conv_in(x)
            for layer in self.mid_block:
                x = layer(x)
            for layer in self.up_blocks:
                x = layer(x)
            return self.conv_out(F.silu(self.norm_out(x)))

        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = feat_cache[idx]
            feat_cache[idx] = x[:, :, -CACHE_T:].clone()
            feat_idx[0] += 1
        else:
            cache_x = None

        x = self.conv_in(x, cache_x)
        for layer in self.mid_block:
            if isinstance(layer, ResnetBlock3d):
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        for layer in self.up_blocks:
            if isinstance(layer, (ResnetBlock3d, Upsample3d)):
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        x = F.silu(self.norm_out(x))
        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = feat_cache[idx]
            feat_cache[idx] = x[:, :, -CACHE_T:].clone()
            feat_idx[0] += 1
        else:
            cache_x = None
        return self.conv_out(x, cache_x)


class AutoencoderKLQwenImage(nn.Module):
    def __init__(self, base_dim: int = 96, z_dim: int = 16, dim_mult: List[int] = [1, 2, 4, 4], num_res_blocks: int = 2, attn_scales: List[float] = [], temperal_downsample: List[bool] = [False, True, True], dropout: float = 0.0, latents_mean: Optional[List[float]] = None, latents_std: Optional[List[float]] = None, input_channels: int = 3):
        super().__init__()
        self.base_dim = base_dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        self.encoder = Encoder3d(base_dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales, self.temperal_downsample, dropout, input_channels=input_channels)
        self.quant_conv = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(base_dim, z_dim, dim_mult, num_res_blocks, attn_scales, self.temperal_upsample, dropout, output_channels=input_channels)

        if latents_mean is not None:
            self.register_buffer("latents_mean", torch.tensor(latents_mean).view(1, z_dim, 1, 1, 1))
        else:
            self.latents_mean = None
        if latents_std is not None:
            self.register_buffer("latents_std", torch.tensor(latents_std).view(1, z_dim, 1, 1, 1))
        else:
            self.latents_std = None

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        moments = self.encoder(x)
        moments = self.quant_conv(moments)
        return DiagonalGaussianDistribution(moments)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def encode_pixels_to_latents(self, pixels: torch.Tensor) -> torch.Tensor:
        # pixels: (B, C, 1, H, W)
        device = self.encoder.conv_in.weight.device
        dtype = self.encoder.conv_in.weight.dtype
        pixels = pixels.to(device, dtype=dtype)
        # Wan/Qwen image inference uses the posterior mean. Sampling the
        # posterior here injects VAE noise into every disk cache and does not
        # match ComfyUI's WanVAE.encode implementation.
        dist = self.encode(pixels)
        latents = dist.mean
        if self.latents_mean is not None and self.latents_std is not None:
            latents = (latents - self.latents_mean.to(latents.device, latents.dtype)) / self.latents_std.to(latents.device, latents.dtype)
        return latents

    def decode_to_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode normalized Krea 2 latents to RGB pixels in the [0, 1] range."""
        device = self.decoder.conv_in.weight.device
        dtype = self.decoder.conv_in.weight.dtype
        latents = latents.to(device=device, dtype=dtype)
        if self.latents_mean is not None and self.latents_std is not None:
            latents = (
                latents * self.latents_std.to(device=device, dtype=dtype)
                + self.latents_mean.to(device=device, dtype=dtype)
            )
        pixels = self.decode(latents)
        if pixels.ndim == 5:
            if pixels.shape[2] != 1:
                raise RuntimeError(
                    f"Krea 2 image VAE returned {pixels.shape[2]} frames; expected exactly one"
                )
            pixels = pixels.squeeze(2)
        return ((pixels.float() + 1.0) / 2.0).clamp_(0.0, 1.0)


def convert_comfyui_state_dict(sd: dict) -> dict:
    if "conv1.bias" not in sd:
        return sd
    key_map = {
        "conv1": "quant_conv",
        "conv2": "post_quant_conv",
        "decoder.conv1": "decoder.conv_in",
        "decoder.head.0": "decoder.norm_out",
        "decoder.head.2": "decoder.conv_out",
        "decoder.middle.0.residual.0": "decoder.mid_block.0.residual.0",
        "decoder.middle.0.residual.2": "decoder.mid_block.0.residual.2",
        "decoder.middle.0.residual.3": "decoder.mid_block.0.residual.3",
        "decoder.middle.0.residual.6": "decoder.mid_block.0.residual.6",
        "decoder.middle.1.norm": "decoder.mid_block.1.norm",
        "decoder.middle.1.proj": "decoder.mid_block.1.proj",
        "decoder.middle.1.to_qkv": "decoder.mid_block.1.to_qkv",
        "decoder.middle.2.residual.0": "decoder.mid_block.2.residual.0",
        "decoder.middle.2.residual.2": "decoder.mid_block.2.residual.2",
        "decoder.middle.2.residual.3": "decoder.mid_block.2.residual.3",
        "decoder.middle.2.residual.6": "decoder.mid_block.2.residual.6",
        "decoder.upsamples": "decoder.up_blocks",
        "encoder.conv1": "encoder.conv_in",
        "encoder.head.0": "encoder.norm_out",
        "encoder.head.2": "encoder.conv_out",
        "encoder.middle": "encoder.mid_block",
        "encoder.downsamples": "encoder.down_blocks",
    }

    new_sd = {}
    for key, val in sd.items():
        new_key = key
        for old_pfx, new_pfx in key_map.items():
            if key == old_pfx or key.startswith(old_pfx + "."):
                new_key = new_pfx + key[len(old_pfx):]
                break
        # ComfyUI wraps the causal spatial convolution in Sequential at index
        # 1, while this implementation stores that convolution directly.
        new_key = new_key.replace(".resample.1.", ".resample.")
        new_sd[new_key] = val
    return new_sd


def load_krea2_vae(vae_path: str, device: Union[str, torch.device] = "cpu", dtype: torch.dtype = torch.bfloat16) -> AutoencoderKLQwenImage:
    VAE_CONFIG = {
        "base_dim": 96,
        "z_dim": 16,
        "dim_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_scales": [],
        "temperal_downsample": [False, True, True],
        "dropout": 0.0,
        "latents_mean": [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921],
        "latents_std": [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916],
    }

    vae = AutoencoderKLQwenImage(**VAE_CONFIG)
    logger.info(f"Loading Krea 2 VAE from {vae_path}")
    sd = load_safetensors(vae_path, device=str(device), disable_mmap=True)
    sd = convert_comfyui_state_dict(sd)
    target_sd = vae.state_dict()
    for k, v in list(sd.items()):
        if k in target_sd:
            t_shape = target_sd[k].shape
            if v.shape != t_shape and v.numel() == target_sd[k].numel():
                sd[k] = v.reshape(t_shape)
    info = vae.load_state_dict(sd, strict=False, assign=True)
    required_missing = [k for k in info.missing_keys if k not in {"latents_mean", "latents_std"}]
    # Encoder temporal convolutions are part of the video path in the ComfyUI
    # checkpoint. A single-frame image never uses them and Encoder3d does not
    # instantiate them.
    allowed_unexpected = {
        f"encoder.down_blocks.{index}.time_conv.{suffix}"
        for index in (5, 8)
        for suffix in ("weight", "bias")
    }
    required_unexpected = [k for k in info.unexpected_keys if k not in allowed_unexpected]
    if required_missing or required_unexpected:
        raise RuntimeError(
            "Krea 2 VAE checkpoint conversion is incomplete: "
            f"missing={required_missing}, unexpected={required_unexpected}"
        )
    logger.info(
        "Loaded all Krea 2 image-VAE weights (%d ignored video-only tensors).",
        len(info.unexpected_keys),
    )
    vae.to(device, dtype=dtype)
    return vae
