"""Krea 2 (K2) text encoder: Qwen3-VL-4B conditioner.

Returns the stacked selected hidden states (b, seq, num_select_layers, dim) plus the
attention mask.
"""

import logging
from dataclasses import dataclass

import torch
from accelerate import init_empty_weights
from torch import Tensor
from transformers import (
    AutoTokenizer,
    Qwen2TokenizerFast,
)
try:
    from transformers import (
        Qwen3VLConfig,
        Qwen3VLForConditionalGeneration,
    )
except ImportError as exc:
    raise ImportError(
        "Krea 2 requires transformers 4.57.1 or newer with native Qwen3-VL support. "
        "Run install.bat to update the environment."
    ) from exc

from library.safetensors_utils import load_split_weights
from library.fp8_optimization_utils import apply_fp8_monkey_patch

logger = logging.getLogger(__name__)


QWEN3_VL_4B_INSTRUCT_REPO_ID = "Qwen/Qwen3-VL-4B-Instruct"

QWEN3_VL_4B_INSTRUCT_CONFIG = {
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "image_token_id": 151655,
    "model_type": "qwen3_vl",
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "dtype": "bfloat16",
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2560,
        "initializer_range": 0.02,
        "intermediate_size": 9728,
        "max_position_embeddings": 262144,
        "model_type": "qwen3_vl_text",
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"},
        "rope_theta": 5000000,
        "tie_word_embeddings": True,
        "use_cache": True,
        "vocab_size": 151936,
    },
    "tie_word_embeddings": True,
    "transformers_version": "4.57.0.dev0",
    "video_token_id": 151656,
    "vision_config": {
        "deepstack_visual_indexes": [5, 11, 17],
        "depth": 24,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1024,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "model_type": "qwen3_vl",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2560,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "vision_end_token_id": 151653,
    "vision_start_token_id": 151652,
}


@dataclass
class TextEncoderConfig:
    max_length: int = 512
    select_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
    tokenizer_repo: str = QWEN3_VL_4B_INSTRUCT_REPO_ID


def _convert_comfyui_qwen3vl_state_dict(sd: dict[str, Tensor]) -> dict[str, Tensor]:
    converted: dict[str, Tensor] = {}
    for key, value in sd.items():
        if key.startswith("model.language_model.") or key.startswith("model.visual."):
            new_key = key
        elif key.startswith("visual."):
            new_key = "model.visual." + key[len("visual.") :]
        elif key.startswith("language_model."):
            new_key = "model." + key
        elif key.startswith("model."):
            new_key = "model.language_model." + key[len("model.") :]
        else:
            new_key = key
        converted[new_key] = value
    return converted


def _load_qwen3_vl_model(
    model_path: str,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    disable_mmap: bool = True,
) -> Qwen3VLForConditionalGeneration:
    config = Qwen3VLConfig.from_dict(QWEN3_VL_4B_INSTRUCT_CONFIG)
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration._from_config(config)

    logger.info(f"Loading Krea 2 text encoder (Qwen3-VL) weights from {model_path}")
    # Preserve native FP8 tensors. Casting the stored FP8 numbers to BF16 here
    # does *not* dequantize them; their sibling weight_scale must be applied by
    # each Linear layer during the forward pass.
    sd = load_split_weights(model_path, device=str(device), disable_mmap=disable_mmap, dtype=None)
    sd = _convert_comfyui_qwen3vl_state_dict(sd)

    has_scaled_fp8 = any(
        value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        for key, value in sd.items()
        if key.endswith(".weight")
    ) and any(key.endswith(".weight_scale") for key in sd)
    if has_scaled_fp8:
        logger.info("Detected scaled-FP8 Qwen3-VL weights; enabling scale-aware Linear layers.")
        apply_fp8_monkey_patch(model, sd, use_scaled_mm=True)

    info = model.load_state_dict(sd, strict=False, assign=True)
    model.tie_weights()

    unexpected = [
        k for k in info.unexpected_keys
        if not (k.endswith(".weight_scale") or k.endswith(".comfy_quant") or k.endswith(".scale") or "weight_scale" in k or "comfy_quant" in k)
    ]
    missing = [k for k in info.missing_keys if k != "lm_head.weight"]
    if unexpected or missing:
        raise RuntimeError(
            f"Qwen3-VL text encoder checkpoint did not match the model: missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    model.to(device)
    # Keep patched weights in native FP8. Non-quantized parameters were stored
    # as BF16 by the official Comfy checkpoint and already have compute dtype.
    if dtype is not None and not has_scaled_fp8:
        model.to(dtype)
    return model.eval().requires_grad_(False)


class Qwen3VLConditioner(torch.nn.Module):
    def __init__(
        self,
        qwen: Qwen3VLForConditionalGeneration,
        tokenizer: AutoTokenizer,
        processor: Qwen2TokenizerFast,
        max_length: int = TextEncoderConfig.max_length,
        select_layers: tuple[int, ...] = TextEncoderConfig.select_layers,
    ):
        super().__init__()
        self.qwen = qwen
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.select_layers = select_layers

    @property
    def device(self) -> torch.device:
        return next(self.qwen.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.qwen.parameters()).dtype

    def to(self, *args, **kwargs):
        self.qwen.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, text: str | list[str]) -> tuple[Tensor, Tensor]:
        if isinstance(text, str):
            text = [text]

        texts = []
        for t in text:
            messages = [{"role": "user", "content": [{"type": "text", "text": t}]}]
            formatted = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            texts.append(formatted)

        encodings = self.processor(
            text=texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(self.qwen.device)
        attention_mask = encodings["attention_mask"].to(self.qwen.device)

        outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states
        selected = [hidden_states[i] for i in self.select_layers]
        stacked = torch.stack(selected, dim=2)  # (b, seq, num_select, dim)

        return stacked, attention_mask


def load_qwen3_vl_conditioner(
    model_path: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
    max_length: int = TextEncoderConfig.max_length,
    select_layers: tuple[int, ...] = TextEncoderConfig.select_layers,
    tokenizer_repo: str = QWEN3_VL_4B_INSTRUCT_REPO_ID,
    disable_mmap: bool = True,
) -> Qwen3VLConditioner:
    qwen = _load_qwen3_vl_model(model_path, dtype=dtype, device=device, disable_mmap=disable_mmap)
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, max_length=max_length, local_files_only=True)
        processor = Qwen2TokenizerFast.from_pretrained(tokenizer_repo, max_length=max_length, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, max_length=max_length)
        processor = Qwen2TokenizerFast.from_pretrained(tokenizer_repo, max_length=max_length)
    conditioner = Qwen3VLConditioner(qwen, tokenizer, processor, max_length=max_length, select_layers=select_layers)
    return conditioner.eval().requires_grad_(False)
