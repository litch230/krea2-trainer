import os
import ast
import re
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

import networks.lora as lora


KREA2_TARGET_REPLACE_MODULES = None


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_block_indices(value, num_blocks):
    """Parse values such as ``18-27``, ``18,20,22`` or a Python list."""
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple, set)):
        raw_items = value
    else:
        text = str(value).strip()
        if text.startswith("["):
            raw_items = ast.literal_eval(text)
        else:
            raw_items = re.split(r"\s*,\s*|\s+", text)

    selected = set()
    for item in raw_items:
        token = str(item).strip()
        if not token:
            continue
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)
        if match:
            start, end = map(int, match.groups())
            if start > end:
                start, end = end, start
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))

    invalid = sorted(index for index in selected if index < 0 or index >= num_blocks)
    if invalid:
        raise ValueError(f"Krea 2 block indices out of range 0-{num_blocks - 1}: {invalid}")
    if not selected:
        raise ValueError("At least one Krea 2 block must be selected")
    return sorted(selected)


def _as_pattern_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        return list(parsed) if isinstance(parsed, (list, tuple)) else [str(parsed)]
    return list(value)


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    num_blocks = len(unet.blocks)
    block_indices = _parse_block_indices(kwargs.pop("train_block_indices", None), num_blocks)
    start_value = kwargs.pop("train_blocks_start", None)
    end_value = kwargs.pop("train_blocks_end", None)
    if block_indices is None and (start_value not in (None, "") or end_value not in (None, "")):
        start = 0 if start_value in (None, "") else int(start_value)
        end = num_blocks - 1 if end_value in (None, "") else int(end_value)
        block_indices = _parse_block_indices(f"{start}-{end}", num_blocks)

    train_text_fusion = _as_bool(kwargs.pop("train_text_fusion", None), True)
    train_final_layer = _as_bool(kwargs.pop("train_final_layer", None), True)
    frozen_prefix_no_grad = _as_bool(kwargs.pop("frozen_prefix_no_grad", None), True)

    exclude_patterns = _as_pattern_list(kwargs.get("exclude_patterns"))
    include_patterns = _as_pattern_list(kwargs.get("include_patterns"))
    if block_indices is not None:
        inactive = [index for index in range(num_blocks) if index not in block_indices]
        if inactive:
            inactive_group = "|".join(str(index) for index in inactive)
            exclude_patterns.append(rf"blocks\.(?:{inactive_group})\..*")

        # Prefix no-grad is safe only when no trainable adapter runs before the
        # first selected main block. These small conditioning projections are
        # deliberately frozen for partial-block training.
        exclude_patterns.extend([r"first(?:\..*)?", r"tmlp\..*", r"txtmlp\..*", r"tproj\..*"])

    if not train_text_fusion:
        exclude_patterns.append(r"txtfusion\..*")
    if not train_final_layer:
        exclude_patterns.append(r"last\..*")

    kwargs["exclude_patterns"] = exclude_patterns
    if include_patterns:
        kwargs["include_patterns"] = include_patterns

    optimize_prefix = (
        block_indices is not None
        and frozen_prefix_no_grad
        and not train_text_fusion
        and not include_patterns
    )
    if hasattr(unet, "configure_partial_lora_training"):
        unet.configure_partial_lora_training(block_indices, frozen_prefix_no_grad=optimize_prefix)
    if block_indices is not None:
        logger.info(
            "Krea 2 partial LoRA: blocks=%s, text_fusion=%s, final_layer=%s, frozen_prefix_no_grad=%s",
            block_indices, train_text_fusion, train_final_layer, optimize_prefix,
        )

    return lora.create_network(
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        target_replace_modules=KREA2_TARGET_REPLACE_MODULES,
        lora_prefix="lora_unet",
        **kwargs,
    )


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    **kwargs,
):
    neuron_dropout = kwargs.pop("neuron_dropout", None)
    return create_arch_network(
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )
