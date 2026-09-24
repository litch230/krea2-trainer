import os
from typing import Any, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, Qwen2TokenizerFast
from library import train_util
from library.strategy_base import (
    LatentsCachingStrategy,
    TokenizeStrategy,
    TextEncodingStrategy,
    TextEncoderOutputsCachingStrategy,
)
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file as safetensors_save
from library.utils import setup_logging
from library.krea2.krea2_encoder import QWEN3_VL_4B_INSTRUCT_REPO_ID, Qwen3VLConditioner
from library.krea2.krea2_sampling import gather_valid_text

setup_logging()
import logging

logger = logging.getLogger(__name__)


class Krea2TextEncoderRequiredError(RuntimeError):
    """Raised when a Krea 2 text cache miss requires the Qwen3-VL encoder."""


class Krea2TokenizeStrategy(TokenizeStrategy):
    PROMPT_TEMPLATE_ENCODE_PREFIX = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
        "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
    )
    PROMPT_TEMPLATE_ENCODE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
    PROMPT_TEMPLATE_ENCODE_START_IDX = 34
    PROMPT_TEMPLATE_ENCODE_SUFFIX_START_IDX = 5

    def __init__(
        self, max_length: Optional[int] = 512, tokenizer_cache_dir: Optional[str] = None
    ) -> None:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                QWEN3_VL_4B_INSTRUCT_REPO_ID, cache_dir=tokenizer_cache_dir, local_files_only=True
            )
            self.processor = Qwen2TokenizerFast.from_pretrained(
                QWEN3_VL_4B_INSTRUCT_REPO_ID, cache_dir=tokenizer_cache_dir, local_files_only=True
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                QWEN3_VL_4B_INSTRUCT_REPO_ID, cache_dir=tokenizer_cache_dir
            )
            self.processor = Qwen2TokenizerFast.from_pretrained(
                QWEN3_VL_4B_INSTRUCT_REPO_ID, cache_dir=tokenizer_cache_dir
            )
        self.max_length = max_length if max_length is not None else 512

    def tokenize(
        self, text: Union[str, List[str]], is_negative: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text = [text] if isinstance(text, str) else text
        prefixed_text = [self.PROMPT_TEMPLATE_ENCODE_PREFIX + value for value in text]
        suffix = self.processor(
            text=[self.PROMPT_TEMPLATE_ENCODE_SUFFIX] * len(text),
            return_tensors="pt",
        )
        encodings = self.tokenizer(
            prefixed_text,
            padding="max_length",
            truncation=True,
            max_length=(
                self.max_length
                + self.PROMPT_TEMPLATE_ENCODE_START_IDX
                - self.PROMPT_TEMPLATE_ENCODE_SUFFIX_START_IDX
            ),
            return_tensors="pt",
        )
        input_ids = torch.cat((encodings["input_ids"], suffix["input_ids"]), dim=1)
        attention_mask = torch.cat((encodings["attention_mask"], suffix["attention_mask"]), dim=1)
        return input_ids, attention_mask


class Krea2TextEncodingStrategy(TextEncodingStrategy):
    def __init__(self, select_layers: Tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)) -> None:
        self.select_layers = select_layers

    def encode_tokens(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], tokens: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not models or models[0] is None:
            raise Krea2TextEncoderRequiredError(
                "A Krea 2 text embedding cache is missing, but the Qwen3-VL text encoder is not loaded."
            )
        conditioner: Qwen3VLConditioner = models[0]
        input_ids, attention_mask = tokens
        input_ids = input_ids.to(conditioner.qwen.device)
        attention_mask = attention_mask.to(conditioner.qwen.device)

        outputs = conditioner.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        selected = [hidden_states[i] for i in self.select_layers]
        stacked = torch.stack(selected, dim=2)
        prefix_tokens = getattr(
            tokenize_strategy,
            "PROMPT_TEMPLATE_ENCODE_START_IDX",
            Krea2TokenizeStrategy.PROMPT_TEMPLATE_ENCODE_START_IDX,
        )
        stacked = stacked[:, prefix_tokens:]
        attention_mask = attention_mask[:, prefix_tokens:]
        stacked, mask = gather_valid_text(stacked, attention_mask.to(dtype=torch.bool))
        return stacked, mask


class Krea2LatentsCachingStrategy(LatentsCachingStrategy):
    KREA2_LATENTS_NPZ_SUFFIX = "_krea2.npz"
    cache_metadata = {"krea2_latent_format": "qwen_image_mean_v2"}

    def __init__(
        self, cache_to_disk: bool = True, batch_size: int = 1, skip_disk_cache_validity_check: bool = False
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)

    @property
    def cache_suffix(self) -> str:
        return Krea2LatentsCachingStrategy.KREA2_LATENTS_NPZ_SUFFIX

    def get_latents_npz_path(
        self, absolute_path: str, image_size: Tuple[int, int]
    ) -> str:
        return self._get_latents_npz_path(absolute_path, image_size)

    def is_disk_cached_latents_expected(
        self,
        bucket_reso: Tuple[int, int],
        npz_path: str,
        flip_aug: bool,
        alpha_mask: bool,
    ) -> bool:
        if not self.skip_disk_cache_validity_check:
            safetensors_path = os.path.splitext(npz_path)[0] + ".safetensors"
            if os.path.exists(safetensors_path):
                with safe_open(safetensors_path, framework="pt") as f:
                    if (f.metadata() or {}).get("krea2_latent_format") != self.cache_metadata["krea2_latent_format"]:
                        return False
        return self._default_is_disk_cached_latents_expected(
            16, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True
        )

    def load_latents_from_disk(
        self, npz_path: str, bucket_reso: Tuple[int, int]
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[List[int]],
        Optional[List[int]],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        import json
        safetensors_path = os.path.splitext(npz_path)[0] + ".safetensors"
        if os.path.exists(safetensors_path):
            with safe_open(safetensors_path, framework="pt") as f:
                keys = set(f.keys())
                target_key = None
                expected_key = f"latents_{bucket_reso[1] // 16}x{bucket_reso[0] // 16}"
                if expected_key in keys:
                    target_key = expected_key
                else:
                    latents_keys = [k for k in keys if k.startswith("latents") and not k.startswith("latents_flipped")]
                    if latents_keys:
                        target_key = latents_keys[0]
                if target_key is not None:
                    suffix = target_key[len("latents"):]
                    latents = f.get_tensor(target_key).float().numpy()
                    flipped = f.get_tensor("latents_flipped" + suffix).float().numpy() if ("latents_flipped" + suffix) in keys else None
                    alpha = f.get_tensor("alpha_mask" + suffix).float().numpy() if ("alpha_mask" + suffix) in keys else None
                    cond = f.get_tensor("cond_latents" + suffix).float().numpy() if ("cond_latents" + suffix) in keys else None
                    md = f.metadata() or {}
                    orig_k = "original_size" + suffix if ("original_size" + suffix) in md else next((k for k in md if "original_size" in k), None)
                    crop_k = "crop_ltrb" + suffix if ("crop_ltrb" + suffix) in md else next((k for k in md if "crop_ltrb" in k), None)
                    orig_sz = json.loads(md[orig_k]) if orig_k and orig_k in md else [0, 0]
                    crop_l = json.loads(md[crop_k]) if crop_k and crop_k in md else [0, 0, 0, 0]
                    return latents, orig_sz, crop_l, flipped, alpha, cond

        return self._default_load_latents_from_disk(16, npz_path, bucket_reso)

    def cache_batch_latents(
        self,
        model,
        batch: List,
        flip_aug: bool,
        alpha_mask: bool,
        random_crop: bool,
    ):
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(target_device)

        def encode_by_vae(img_tensor: torch.Tensor) -> torch.Tensor:
            # img_tensor: (B, C, H, W) normalized to [-1, 1] by train_util
            # Qwen-Image / Krea 2 VAE expects 5D tensor (B, C, F=1, H, W)
            pixels = img_tensor.unsqueeze(2)
            latents = model.encode_pixels_to_latents(pixels)  # (B, Z_dim, 1, H_lat, W_lat)
            return latents

        vae_device = target_device
        vae_dtype = model.encoder.conv_in.weight.dtype

        self._default_cache_batch_latents(
            encode_by_vae,
            vae_device,
            vae_dtype,
            batch,
            flip_aug,
            alpha_mask,
            random_crop,
            multi_resolution=True,
        )
        # Keep the VAE on the GPU between cache batches. NetworkTrainer moves
        # it back to CPU and clears device memory once after the complete
        # dataset pass; doing that here would transfer the full VAE for every
        # batch and makes cache creation dramatically slower.


class Krea2TextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    KREA2_TEXT_ENCODER_OUTPUTS_SAFETENSORS_SUFFIX = "_krea2_qwen3_vl.safetensors"

    def __init__(
        self,
        cache_to_disk: bool = True,
        batch_size: Optional[int] = 1,
        skip_disk_cache_validity_check: bool = False,
        is_partial: bool = False,
        max_token_length: int = 512,
    ) -> None:
        super().__init__(
            cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial
        )
        self.max_token_length = int(max_token_length)

    @property
    def cache_suffix(self) -> str:
        return Krea2TextEncoderOutputsCachingStrategy.KREA2_TEXT_ENCODER_OUTPUTS_SAFETENSORS_SUFFIX

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        cache_dir = os.path.join(os.path.dirname(image_abs_path), "text_encoder_cache")
        os.makedirs(cache_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_abs_path))[0]
        return os.path.join(cache_dir, base_name + self.cache_suffix)

    def is_disk_cached_outputs_expected(self, npz_path: str) -> bool:
        if not self.cache_to_disk:
            return False

        safetensors_path = os.path.splitext(npz_path)[0] + ".safetensors"
        if os.path.exists(safetensors_path):
            try:
                with safe_open(safetensors_path, framework="pt") as f:
                    metadata = f.metadata() or {}
                    if metadata.get("format") in {"scaled_fp8_v1", "scaled_fp8_v2"}:
                        # v1 could overflow its float16 scales. v2 was encoded
                        # before scaled-FP8 Qwen weights were dequantized, so
                        # its hidden states are numerically invalid.
                        return False
                    cached_max_length = metadata.get("max_token_length")
                    # Legacy caches have no length metadata. They are valid only
                    # for the former Krea 2 default of 512 tokens.
                    if cached_max_length is None:
                        if self.max_token_length != 512:
                            return False
                    elif int(cached_max_length) != self.max_token_length:
                        return False
                    if self.skip_disk_cache_validity_check:
                        return True
                    keys = set(f.keys())
                    if "hidden_state" not in keys:
                        return False
                    if "attention_mask" not in keys:
                        return False
            except Exception as e:
                logger.error(f"Error loading file: {safetensors_path}")
                raise e
            return True
        return False

    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        safetensors_path = os.path.splitext(npz_path)[0] + ".safetensors"
        with safe_open(safetensors_path, framework="pt") as f:
            hidden_state = f.get_tensor("hidden_state").float()
            if "hidden_state_scale" in f.keys():
                hidden_state = hidden_state * f.get_tensor("hidden_state_scale").float()
            hidden_state = hidden_state.numpy()
            attention_mask = f.get_tensor("attention_mask").int().numpy()
        return [hidden_state, attention_mask]

    @torch.no_grad()
    def cache_batch_outputs(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        text_encoding_strategy: TextEncodingStrategy,
        batch: List[train_util.ImageInfo],
    ) -> None:
        assert isinstance(text_encoding_strategy, Krea2TextEncodingStrategy)
        assert isinstance(tokenize_strategy, Krea2TokenizeStrategy)
        if not models or models[0] is None:
            raise Krea2TextEncoderRequiredError(
                "At least one Krea 2 text embedding cache is missing. Qwen3-VL must be loaded to create it."
            )

        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        conditioner = models[0]
        conditioner.to(target_device)

        # Encode the whole cache batch in one Qwen pass. Each item is sliced
        # back to its valid token length before saving, so padding costs no disk.
        tokens = tokenize_strategy.tokenize([info.caption for info in batch])
        hidden_state, attention_masks = text_encoding_strategy.encode_tokens(
            tokenize_strategy, models, tokens
        )

        for i, info in enumerate(batch):
            valid_length = int(attention_masks[i].sum().item())
            hidden_state_i = hidden_state[i, :valid_length].float().cpu()
            attention_mask_i = attention_masks[i, :valid_length].cpu().to(torch.int32)

            # Per-token/per-layer scaling preserves substantially more range
            # than a raw FP8 cast while using roughly half the BF16 disk space.
            fp8_max = torch.finfo(torch.float8_e4m3fn).max
            hidden_scale = hidden_state_i.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / fp8_max
            hidden_state_fp8 = (hidden_state_i / hidden_scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
            # Qwen3-VL hidden states can require scales above float16's 65504
            # limit. Keep the tiny scale tensor in float32 to avoid Inf/NaN.
            hidden_scale = hidden_scale.to(torch.float32)

            if self.cache_to_disk:
                assert info.text_encoder_outputs_npz is not None
                safetensors_path = os.path.splitext(info.text_encoder_outputs_npz)[0] + ".safetensors"
                safetensors_save(
                    {
                        "hidden_state": hidden_state_fp8.contiguous(),
                        "hidden_state_scale": hidden_scale.contiguous(),
                        "attention_mask": attention_mask_i.contiguous(),
                    },
                    safetensors_path,
                    metadata={
                        "format": "scaled_fp8_v3",
                        "max_token_length": str(self.max_token_length),
                    },
                )
            else:
                info.text_encoder_outputs = [
                    hidden_state_i.float().numpy(),
                    attention_mask_i.numpy(),
                ]
        # Keep the conditioner on the GPU between cache batches. The trainer
        # releases it once after the complete dataset cache pass.
