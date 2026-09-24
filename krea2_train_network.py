"""LoRA training backend for Krea 2."""

import argparse
import copy
import sys
import io
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if sys.platform == "win32":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from typing import Any, Tuple, List

import torch
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
from torch import Tensor
from accelerate import Accelerator

import train_network
from library.train_util import clean_memory_on_device
from library import (
    krea2_train_utils,
    strategy_base,
    strategy_krea2,
    train_util,
)
from library.krea2 import krea2_utils, krea2_sampling
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class Krea2NetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None

    def cast_text_encoder(self, args):
        return not args.cache_text_encoder_outputs

    def cast_unet(self, args):
        return False

    def assert_extra_args(self, args, train_dataset_group, val_dataset_group):
        super().assert_extra_args(args, train_dataset_group, val_dataset_group)

        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            logger.warning("Enabling cache_text_encoder_outputs due to disk caching")
            args.cache_text_encoder_outputs = True

        train_dataset_group.verify_bucket_reso_steps(16)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(16)

    def load_target_model(self, args, weight_dtype, accelerator):
        ckpt_path = args.pretrained_model_name_or_path
        if ckpt_path is None or ckpt_path == "":
            raise ValueError("pretrained_model_name_or_path is required")

        dit = krea2_utils.load_krea2_dit(
            dit_path=ckpt_path,
            device="cpu",
            dtype=weight_dtype,
            fp8_scaled=getattr(args, "fp8_scaled", False),
            attn_mode=getattr(args, "attn_mode", "torch"),
            split_attn=getattr(args, "split_attn", False),
            # Dequantizing Comfy's 96 "full precision" projections on every
            # checkpoint recomputation makes LoRA training several times
            # slower. Tensor-core scaled FP8 keeps the frozen base weights in
            # place and still leaves LoRA arithmetic in BF16.
            honor_fp8_full_precision=False,
        )
        if getattr(args, "blocks_to_swap", 0) > 0:
            from library.krea2.krea2_mmdit import BlockSwapConfig
            config = BlockSwapConfig(device=accelerator.device, supports_backward=True)
            dit.enable_block_swap(args.blocks_to_swap, config)

        te_path = getattr(args, "qwen3_vl", None) or getattr(args, "text_encoder", None)
        is_te_cached = (
            (getattr(args, "cache_text_encoder_outputs", False) or getattr(args, "cache_text_encoder_outputs_to_disk", False))
            and getattr(args, "skip_text_encoder_cache_check", False)
        )
        if te_path is not None and os.path.exists(te_path) and not is_te_cached:
            qwen3_vl = krea2_utils.load_krea2_text_encoder(te_path, dtype=weight_dtype, device=torch.device("cpu"))
        else:
            qwen3_vl = None

        vae_path = getattr(args, "vae", None)
        if vae_path is not None and os.path.exists(vae_path):
            vae = krea2_utils.load_krea2_vae(vae_path, device="cpu", dtype=weight_dtype)
        else:
            vae = None

        text_encoder_models = [qwen3_vl] if qwen3_vl is not None else []
        return "krea2", text_encoder_models, vae, dit

    def get_tokenize_strategy(self, args):
        return strategy_krea2.Krea2TokenizeStrategy(
            getattr(args, "krea2_max_token_length", 512), args.tokenizer_cache_dir
        )

    def get_tokenizers(self, tokenize_strategy: strategy_krea2.Krea2TokenizeStrategy):
        return [tokenize_strategy.tokenizer]

    def get_latents_caching_strategy(self, args):
        return strategy_krea2.Krea2LatentsCachingStrategy(
            args.cache_latents_to_disk,
            args.vae_batch_size,
            args.skip_latent_cache_check,
        )

    @staticmethod
    def _datasets_in_group(dataset_group):
        return getattr(dataset_group, "datasets", [dataset_group])

    def attach_existing_latent_caches(self, args, dataset_group):
        strategy = self.get_latents_caching_strategy(args)
        missing = []
        attached = 0
        for dataset in self._datasets_in_group(dataset_group):
            for info in dataset.image_data.values():
                subset = dataset.image_to_subset[info.image_key]
                cache_path = strategy.get_latents_npz_path(info.absolute_path, info.image_size)
                if not strategy.is_disk_cached_latents_expected(
                    info.bucket_reso, cache_path, subset.flip_aug, subset.alpha_mask
                ):
                    missing.append(os.path.splitext(cache_path)[0] + ".safetensors")
                    continue
                info.latents_npz = cache_path
                attached += 1
        if missing:
            examples = "\n".join(missing[:5])
            raise FileNotFoundError(
                f"Skip cache creation requested, but {len(missing)} Krea latent cache(s) are missing. "
                f"First missing files:\n{examples}"
            )
        logger.info("Skip cache creation: attached %d existing Krea latent caches.", attached)

    def get_text_encoding_strategy(self, args):
        return strategy_krea2.Krea2TextEncodingStrategy()

    def get_text_encoders_train_flags(self, args, text_encoders):
        return [False]

    def encode_images_to_latents(self, args, vae, images):
        """Encode an uncached image batch with the Krea 2/Qwen-Image VAE."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                "Krea 2 VAE expects training images shaped (B, 3, H, W), "
                f"but received {tuple(images.shape)}"
            )

        # The disk-cache pass moves the VAE back to CPU. Missing cache entries
        # can still reach this method, so restore it before encoding them.
        vae_param = next(vae.parameters(), None)
        if vae_param is not None and (vae_param.device != images.device or vae_param.dtype != images.dtype):
            logger.warning(
                "An uncached Krea 2 image batch was found; moving the VAE "
                f"from {vae_param.device}/{vae_param.dtype} to {images.device}/{images.dtype}."
            )
            vae.to(device=images.device, dtype=images.dtype)

        # Qwen-Image uses a causal 3D VAE and requires (B, C, F, H, W).
        return vae.encode_pixels_to_latents(images.unsqueeze(2))

    def shift_scale_latents(self, args, latents):
        # encode_pixels_to_latents and the Krea 2 disk cache already apply the
        # model's per-channel latent mean/std normalization.
        return latents

    def get_noise_scheduler(self, args, device):
        """Krea 2 is a rectified-flow model, not a DDPM noise predictor."""
        return krea2_train_utils.FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=getattr(args, "discrete_flow_shift", 1.0),
        )

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet,
        network,
        weight_dtype,
        train_unet,
        is_train=True,
    ):
        """Build the shifted logit-normal rectified-flow training objective.

        The Krea 2 DiT predicts the velocity ``noise - clean_latent`` along the
        linear path ``(1-t) * clean_latent + t * noise``.  Falling back to the
        base SD trainer here teaches DDPM epsilon prediction and rapidly
        corrupts a LoRA, which presents as coloured noise during sampling.
        """
        noise = torch.randn_like(latents)
        noisy_latents, normalized_timesteps, _sigmas = krea2_train_utils.get_noisy_model_input_and_timesteps(
            args,
            latents,
            noise,
            accelerator.device,
            weight_dtype,
        )
        # call_unet follows the trainer-wide 0..1000 timestep convention and
        # normalizes it immediately before Krea 2's timestep embedding.
        timesteps = normalized_timesteps * 1000.0

        if args.gradient_checkpointing:
            noisy_latents.requires_grad_(True)
            if self.is_train_text_encoder(args):
                for tensor in text_encoder_conds:
                    if tensor is not None and (torch.is_floating_point(tensor) or torch.is_complex(tensor)):
                        tensor.requires_grad_(True)

        with torch.set_grad_enabled(is_train), accelerator.autocast():
            model_pred = self.call_unet(
                args,
                accelerator,
                unet,
                noisy_latents.requires_grad_(train_unet),
                timesteps,
                text_encoder_conds,
                batch,
                weight_dtype,
            )

        target = noise - latents
        return model_pred, target, timesteps, None

    def call_unet(
        self,
        args,
        accelerator,
        unet,
        noisy_latents,
        timesteps,
        text_conds,
        batch,
        weight_dtype,
    ):
        from einops import rearrange, repeat
        model = unet  # SingleStreamDiT
        if getattr(args, "gradient_checkpointing", False) and not model.training:
            model.train()
        device = accelerator.device
        patch = getattr(model.config, "patch", 2)

        if noisy_latents.dim() == 5:
            nmi = noisy_latents.squeeze(2)
        else:
            nmi = noisy_latents

        bsize, _, lat_h, lat_w = nmi.shape
        h_, w_ = lat_h // patch, lat_w // patch

        img_tokens = rearrange(nmi, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

        imgids = torch.zeros((h_, w_, 3), device=device)
        imgids[..., 1] = torch.arange(h_, device=device)[:, None]
        imgids[..., 2] = torch.arange(w_, device=device)[None, :]
        imgpos = repeat(imgids, "h w three -> b (h w) three", b=bsize, three=3)
        imgmask = torch.ones(bsize, h_ * w_, device=device, dtype=torch.bool)

        if isinstance(text_conds, (list, tuple)):
            context, txtmask = text_conds[0], text_conds[1]
        else:
            context, txtmask = text_conds, None

        if isinstance(context, (list, tuple)):
            context = context[0]

        context = context.to(device=device, dtype=weight_dtype)
        if context.dim() == 3:
            context = context.unsqueeze(2)

        max_len = context.shape[1]
        txtpos = torch.zeros(bsize, max_len, 3, device=device)

        if txtmask is None:
            txtmask = torch.ones(bsize, max_len, device=device, dtype=torch.bool)
        else:
            txtmask = txtmask.to(device=device, dtype=torch.bool)

        mask = torch.cat((imgmask, txtmask), dim=1)
        pos = torch.cat((imgpos, txtpos), dim=1)

        img_tokens = img_tokens.to(device=device, dtype=weight_dtype)
        t = (timesteps / 1000.0).to(device=device, dtype=weight_dtype)

        model_pred = model(img=img_tokens, context=context, t=t, pos=pos, mask=mask)

        model_pred = rearrange(model_pred, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=patch, pw=patch, h=h_, w=w_)
        model_pred = model_pred.unsqueeze(2)  # (B, C, 1, H, W)
        return model_pred

    def prepare_unet_with_accelerator(self, args, accelerator, unet):
        if getattr(args, "blocks_to_swap", 0) > 0:
            from library.custom_offloading_utils import BlockSwapConfig
            config = BlockSwapConfig(device=accelerator.device, supports_backward=True, h2d_only=True, use_pinned_memory=True)
            unet.enable_block_swap(args.blocks_to_swap, config)
            unet.move_to_device_except_swap_blocks(accelerator.device)
            return unet
        return accelerator.prepare(unet)

    def cache_text_encoder_outputs_if_needed(self, args, accelerator, unet, vae, text_encoders, dataset, weight_dtype):
        if not args.cache_text_encoder_outputs:
            return

        if getattr(args, "skip_cache_generation", False):
            strategy = self.get_text_encoder_outputs_caching_strategy(args)
            missing = []
            attached = 0
            for current_dataset in self._datasets_in_group(dataset):
                for info in current_dataset.image_data.values():
                    cache_path = strategy.get_outputs_npz_path(info.absolute_path)
                    if not strategy.is_disk_cached_outputs_expected(cache_path):
                        missing.append(cache_path)
                        continue
                    info.text_encoder_outputs_npz = cache_path
                    attached += 1
            if missing:
                examples = "\n".join(missing[:5])
                raise FileNotFoundError(
                    f"Skip cache creation requested, but {len(missing)} Krea text cache(s) are missing. "
                    f"First missing files:\n{examples}"
                )
            logger.info("Skip cache creation: attached %d existing Krea text caches.", attached)
            return

        loaded_lazily = False
        cache_models = text_encoders
        try:
            try:
                # With skip_text_encoder_cache_check, load_target_model intentionally
                # omits Qwen3-VL. The cache scan can still complete without a model
                # when every expected file exists.
                dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)
            except strategy_krea2.Krea2TextEncoderRequiredError:
                te_path = getattr(args, "qwen3_vl", None) or getattr(args, "text_encoder", None)
                if not te_path or not os.path.exists(te_path):
                    raise FileNotFoundError(
                        "A Krea 2 text cache is missing and the configured Qwen3-VL model could not be found: "
                        f"{te_path!r}"
                    )
                logger.warning("Missing Krea 2 text cache detected; loading Qwen3-VL to create only missing outputs.")
                conditioner = krea2_utils.load_krea2_text_encoder(
                    te_path, dtype=weight_dtype, device=torch.device("cpu")
                )
                cache_models = [conditioner]
                loaded_lazily = True
                dataset.new_cache_text_encoder_outputs(cache_models, accelerator)

            # Krea 2 cannot use the generic Stable Diffusion sampler. Pre-encode
            # sample prompts while Qwen is available, then release Qwen before the
            # DiT is placed on the GPU for training.
            samples_enabled = (
                getattr(args, "sample_at_first", False)
                or getattr(args, "sample_every_n_steps", None) is not None
                or getattr(args, "sample_every_n_epochs", None) is not None
            )
            if samples_enabled and os.path.isfile(getattr(args, "sample_prompts", "")):
                if not cache_models:
                    te_path = getattr(args, "qwen3_vl", None) or getattr(args, "text_encoder", None)
                    if not te_path or not os.path.exists(te_path):
                        logger.warning("Krea 2 sample prompts cannot be encoded because Qwen3-VL was not found.")
                    else:
                        cache_models = [krea2_utils.load_krea2_text_encoder(te_path, dtype=weight_dtype, device="cpu")]
                        loaded_lazily = True

                if cache_models:
                    tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
                    text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()
                    prompts = train_util.load_prompts(args.sample_prompts)
                    sample_texts = []
                    for prompt in prompts:
                        for text in (prompt.get("prompt", ""), prompt.get("negative_prompt", "")):
                            if text and text not in sample_texts:
                                sample_texts.append(text)

                    self.sample_prompts_te_outputs = {}
                    cache_models[0].to(accelerator.device)
                    with torch.no_grad():
                        for text in sample_texts:
                            tokens = tokenize_strategy.tokenize(text)
                            encoded = text_encoding_strategy.encode_tokens(tokenize_strategy, cache_models, tokens)
                            tup = tuple(t.detach().cpu() for t in encoded)
                            self.sample_prompts_te_outputs[text] = tup
                            self.sample_prompts_te_outputs[text.strip()] = tup
                            self.sample_prompts_te_outputs[" ".join(text.strip().split())] = tup
                    logger.info("Cached Krea 2 embeddings for %d unique sample prompt texts.", len(sample_texts))
        finally:
            import gc
            del cache_models[:]
            del text_encoders[:]
            gc.collect()
            clean_memory_on_device(accelerator.device)
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizers, text_encoder, unet):
        if global_step == 0 or epoch == 0:
            if global_step != 0 or not getattr(args, "sample_at_first", False):
                return
        elif getattr(args, "sample_every_n_epochs", None) is not None:
            if epoch is None or epoch <= 0 or epoch % args.sample_every_n_epochs != 0:
                return
        elif getattr(args, "sample_every_n_steps", None) is not None:
            if epoch is not None or global_step % args.sample_every_n_steps != 0:
                return
        else:
            return

        if not self.sample_prompts_te_outputs:
            logger.warning("Skipping Krea 2 samples: cached sample prompt embeddings are unavailable.")
            return
        if vae is None:
            logger.warning("Skipping Krea 2 samples: VAE is unavailable.")
            return

        model = accelerator.unwrap_model(unet)
        prompts = train_util.load_prompts(args.sample_prompts)
        sample_dir = os.path.join(args.output_dir, "sample")
        os.makedirs(sample_dir, exist_ok=True)
        was_training = model.training
        model.eval()

        is_latents_cached = bool(getattr(args, "cache_latents", False) or getattr(args, "cache_latents_to_disk", False))
        is_vae_needed_in_training = (not is_latents_cached) or (getattr(args, "clipping_loss_weight", 0.0) > 0.0)
        target_vae_device = device if is_vae_needed_in_training else torch.device("cpu")

        try:
            for prompt in prompts:
                prompt_text = prompt.get("prompt", "")
                negative_text = prompt.get("negative_prompt", "")
                norm_prompt = " ".join(prompt_text.strip().split())
                encoded = (
                    self.sample_prompts_te_outputs.get(prompt_text)
                    or self.sample_prompts_te_outputs.get(prompt_text.strip())
                    or self.sample_prompts_te_outputs.get(norm_prompt)
                )
                if encoded is None:
                    logger.warning("Skipping uncached Krea 2 sample prompt: %s", prompt_text[:80])
                    continue
                txt, txtmask = encoded
                unencoded = None
                if negative_text:
                    norm_neg = " ".join(negative_text.strip().split())
                    unencoded = (
                        self.sample_prompts_te_outputs.get(negative_text)
                        or self.sample_prompts_te_outputs.get(negative_text.strip())
                        or self.sample_prompts_te_outputs.get(norm_neg)
                    )
                untxt, untxtmask = unencoded if unencoded is not None else (None, None)

                images = krea2_sampling.sample(
                    model,
                    vae,
                    txt,
                    txtmask,
                    untxt=untxt,
                    untxtmask=untxtmask,
                    device=device,
                    dtype=torch.bfloat16,
                    width=prompt.get("width", 512),
                    height=prompt.get("height", 512),
                    steps=prompt.get("sample_steps", 28),
                    cfg_scale=prompt.get("scale", 5.5),
                    seed=prompt.get("seed", 0),
                )
                enum = prompt.get("enum", 0)
                suffix = f"e{epoch:06d}" if epoch is not None else f"{global_step:06d}"
                for image_index, image in enumerate(images):
                    path = os.path.join(sample_dir, f"{args.output_name}_{suffix}_{enum:02d}_{image_index:02d}.png")
                    image.save(path)
                    logger.info("Saved Krea 2 sample: %s", path)
        finally:
            if vae is not None:
                vae.to(target_vae_device)
            if was_training or getattr(args, "gradient_checkpointing", False):
                model.train()
            import gc
            gc.collect()
            clean_memory_on_device(device)

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_krea2.Krea2TextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk,
                args.text_encoder_batch_size or 4,
                args.skip_text_encoder_cache_check,
                max_token_length=getattr(args, "krea2_max_token_length", 512),
            )
        return None


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    train_util.add_dit_training_arguments(parser)
    parser.add_argument("--qwen3_vl", type=str, default=None, help="path to Qwen3-VL text encoder safetensors")
    parser.add_argument("--fp8_scaled", action="store_true", help="enable scaled fp8 quantization for DiT")
    parser.add_argument("--attn_mode", type=str, default="torch", help="attention implementation (torch, flash, sageattn, xformers)")
    parser.add_argument("--split_attn", action="store_true", help="use split attention for variable sequence lengths")
    parser.add_argument("--stop_file", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--skip_cache_generation", action="store_true", help="reuse complete Krea caches without a cache pass")
    parser.add_argument(
        "--krea2_max_token_length", type=int, default=512, choices=[128, 256, 384, 512, 768, 1024, 2048],
        help="per-caption Krea 2 token ceiling; shorter captions retain only their actual token count",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)
    trainer = Krea2NetworkTrainer()
    trainer.train(args)
