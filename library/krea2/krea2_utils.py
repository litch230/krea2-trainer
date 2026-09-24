"""Shared loaders / helpers for the Krea 2 (K2) integration."""

import json
import logging
from typing import Optional, Union

import torch

from library.krea2.krea2_encoder import (
    QWEN3_VL_4B_INSTRUCT_REPO_ID,
    Qwen3VLConditioner,
    TextEncoderConfig,
    load_qwen3_vl_conditioner,
)
from library.krea2.krea2_mmdit import SingleMMDiTConfig, SingleStreamDiT
from library.krea2.krea2_vae import load_krea2_vae
from library.fp8_optimization_utils import apply_fp8_monkey_patch
from library.lora_utils import load_safetensors_with_lora_and_fp8
from library.safetensors_utils import load_safetensors

logger = logging.getLogger(__name__)


KREA2_FP8_OPTIMIZATION_TARGET_KEYS = ["blocks."]
KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS = ["mod.", "norm", "txtfusion"]


def _read_comfy_fp8_full_precision_layers(path: str) -> set[str]:
    """Read Comfy's per-layer scaled-FP8 execution policy, if present."""
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            raw = (handle.metadata() or {}).get("_quantization_metadata")
        if not raw:
            return set()
        layers = json.loads(raw).get("layers", {})
        return {
            name
            for name, config in layers.items()
            if config.get("format") in ("float8_e4m3fn", "float8_e5m2")
            and config.get("full_precision_matrix_mult", False)
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("Could not read scaled-FP8 metadata from %s: %s", path, exc)
        return set()


single_mmdit_large_wide = SingleMMDiTConfig(
    features=6144,
    tdim=256,
    txtdim=2560,
    heads=48,
    kvheads=12,
    multiplier=4,
    layers=28,
    patch=2,
    channels=16,
    txtheads=20,
    txtkvheads=20,
    txtlayers=12,
)


def load_krea2_dit(
    dit_path: str,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    config: SingleMMDiTConfig = single_mmdit_large_wide,
    fp8_scaled: bool = False,
    loading_device: Optional[Union[str, torch.device]] = None,
    attn_mode: str = "torch",
    split_attn: bool = False,
    honor_fp8_full_precision: bool = True,
    lora_weights: Optional[list] = None,
    lora_multipliers: Optional[list] = None,
) -> SingleStreamDiT:
    device = torch.device(device)
    loading_device = device if loading_device is None else torch.device(loading_device)
    has_lora = lora_weights is not None and len(lora_weights) > 0

    logger.info(
        f"Loading Krea 2 DiT weights from {dit_path}"
        + (" (fp8 scaled)" if fp8_scaled else "")
        + (f" (+{len(lora_weights)} LoRA merged)" if has_lora else "")
    )
    with torch.device("meta"):
        dit = SingleStreamDiT(config, attn_mode=attn_mode, split_attn=split_attn)

    if fp8_scaled or has_lora:
        sd = load_safetensors_with_lora_and_fp8(
            model_files=dit_path,
            lora_weights_list=lora_weights,
            lora_multipliers=lora_multipliers,
            fp8_optimization=fp8_scaled,
            calc_device=device,
            move_to_device=(loading_device == device),
            dit_weight_dtype=None if fp8_scaled else dtype,
            target_keys=KREA2_FP8_OPTIMIZATION_TARGET_KEYS if fp8_scaled else None,
            exclude_keys=KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS if fp8_scaled else None,
        )
        if fp8_scaled:
            apply_fp8_monkey_patch(dit, sd, use_scaled_mm=True)
        extra_keys = [k for k in sd.keys() if k.endswith(".weight_scale") or k.endswith(".input_scale") or k.endswith(".weight_scale_2") or k.endswith(".comfy_quant")]
        for k in extra_keys:
            del sd[k]
        if loading_device.type != "cpu":
            for key in sd.keys():
                sd[key] = sd[key].to(loading_device)
        dit.load_state_dict(sd, strict=False, assign=True)
    else:
        from safetensors.torch import load_file
        sd = load_file(dit_path, device=str(loading_device))
        has_fp8 = any(v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) for v in sd.values())
        if has_fp8:
            logger.info("Detected native FP8 scaled weights in checkpoint. Applying FP8 monkey patch for ultra-low VRAM.")
            full_precision_layers = (
                _read_comfy_fp8_full_precision_layers(dit_path)
                if honor_fp8_full_precision
                else set()
            )
            if full_precision_layers:
                logger.info(
                    "Honoring Comfy scaled-FP8 policy: %d sensitive Linear layers use dequantized matmul.",
                    len(full_precision_layers),
                )
            elif not honor_fp8_full_precision:
                logger.info("Using tensor-core scaled FP8 for all DiT Linear layers.")
            apply_fp8_monkey_patch(
                dit,
                sd,
                use_scaled_mm=True,
                full_precision_module_paths=full_precision_layers,
            )
        else:
            if dtype is not None:
                for k in sd.keys():
                    if torch.is_floating_point(sd[k]):
                        sd[k] = sd[k].to(dtype=dtype)
            extra_keys = [k for k in sd.keys() if k.endswith(".weight_scale") or k.endswith(".input_scale") or k.endswith(".weight_scale_2") or k.endswith(".comfy_quant")]
            for k in extra_keys:
                if k in sd:
                    del sd[k]
        dit.load_state_dict(sd, strict=False, assign=True)

    return dit


def load_krea2_dit_state_dict(
    dit_path: str,
    fp8_scaled: bool = False,
    calc_device: Union[str, torch.device] = "cpu",
    result_device: Union[str, torch.device] = "cpu",
    config: SingleMMDiTConfig = single_mmdit_large_wide,
) -> dict:
    calc_dev = torch.device(calc_device)
    rd = torch.device(result_device)
    move_to_device = calc_dev == rd

    if fp8_scaled:
        sd = load_safetensors_with_lora_and_fp8(
            model_files=dit_path,
            lora_weights_list=None,
            lora_multipliers=None,
            fp8_optimization=True,
            calc_device=calc_dev,
            move_to_device=move_to_device,
            dit_weight_dtype=None,
            target_keys=KREA2_FP8_OPTIMIZATION_TARGET_KEYS,
            exclude_keys=KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS,
        )
    else:
        sd = load_safetensors(dit_path, device=result_device, disable_mmap=True, dtype=torch.bfloat16)

    sd = {k: v.to(rd) for k, v in sd.items()}
    return sd


def load_krea2_text_encoder(
    path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: Union[str, torch.device] = "cpu",
    max_length: int = TextEncoderConfig.max_length,
    select_layers: tuple = TextEncoderConfig.select_layers,
    tokenizer_repo: str = QWEN3_VL_4B_INSTRUCT_REPO_ID,
) -> Qwen3VLConditioner:
    return load_qwen3_vl_conditioner(
        path,
        dtype=dtype,
        device=device,
        max_length=max_length,
        select_layers=select_layers,
        tokenizer_repo=tokenizer_repo,
    )


@torch.no_grad()
def get_krea2_prompt_embeds(encoder: Qwen3VLConditioner, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    hiddens, mask = encoder(prompts)
    return hiddens, mask.to(dtype=torch.bool)
