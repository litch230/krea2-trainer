"""Training utilities used by the Krea 2 trainer."""

from typing import Optional, Tuple

import torch

from library.sd3_train_utils import FlowMatchEulerDiscreteScheduler


def get_noisy_model_input_and_timesteps(
    args,
    latents: torch.Tensor,
    noise: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    global_step: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the rectified-flow noisy input and its normalized timestep."""
    batch_size = latents.shape[0]
    method = getattr(args, "timestep_sample_method", "logit_normal") or "logit_normal"
    scale = getattr(args, "sigmoid_scale", 1.0)
    scale = 1.0 if scale is None else scale
    shift = getattr(args, "discrete_flow_shift", 1.0)

    simple_logit = (
        method in {"logit_normal", "sigmoid", "shift"}
        and not getattr(args, "adaptive_sigmoid", False)
        and getattr(args, "sigmoid_bias", 0.0) == 0.0
        and getattr(args, "sigmoid_mix", 1.0) == 1.0
    )
    if simple_logit:
        t = torch.sigmoid(torch.randn(batch_size, device=device) * scale)
    elif method == "uniform" and not getattr(args, "adaptive_sigmoid", False):
        t = torch.rand(batch_size, device=device)
    elif method in {"uniform", "sigmoid", "shift", "sigmoid_monotonic", "cosine", "logit_normal"}:
        from library.timestep_samplers import global_sampler

        bias = getattr(args, "sigmoid_bias", 0.0)
        mix = getattr(args, "sigmoid_mix", 1.0)
        if getattr(args, "adaptive_sigmoid", False) and global_step is not None:
            max_steps = getattr(args, "max_train_steps", 1)
            if max_steps > 0:
                progress = min(1.0, max(0.0, global_step / max_steps))
                bias *= 1.0 - progress
                mix *= progress
        sampler_mode = "sigmoid" if method in {"logit_normal", "shift"} else method
        t = global_sampler.sample(
            batch_size,
            mode=sampler_mode,
            bias=bias,
            scale=scale,
            mix=mix,
            device=device,
        )
    else:
        raise NotImplementedError(f"Unknown timestep_sample_method: {method}")

    if method == "shift" and shift is not None and shift != 1.0:
        t = (t * shift) / (1 + (shift - 1) * t)
    t = t.clamp(1e-5, 1.0 - 1e-5)
    expanded_t = t.view(-1, *([1] * (latents.ndim - 1)))

    noise_gamma = getattr(args, "ip_noise_gamma", None)
    if noise_gamma:
        extra_noise = torch.randn_like(latents, device=latents.device, dtype=dtype)
        if getattr(args, "ip_noise_gamma_random_strength", False):
            noise_gamma = torch.rand(1, device=latents.device, dtype=dtype) * noise_gamma
        noisy_input = (1 - expanded_t) * latents + expanded_t * (noise + noise_gamma * extra_noise)
    else:
        noisy_input = (1 - expanded_t) * latents + expanded_t * noise

    return noisy_input.to(dtype), t.to(dtype), t.view(-1, 1).to(dtype)
