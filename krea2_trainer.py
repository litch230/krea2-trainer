#!/usr/bin/env python3
"""Krea 2 LoRA trainer and local configuration interface.

The launcher keeps machine-specific paths on the command line, builds the
training configuration, validates the dataset, and delegates training and
cache creation to ``Krea2NetworkTrainer``.

Example:
    python krea2_trainer.py \
      --model models/krea2_raw_fp8_scaled.safetensors \
      --qwen3-vl models/qwen3vl_4b_fp8_scaled.safetensors \
      --vae models/qwen_image_vae.safetensors \
      --dataset-config configs/my_krea2_dataset.toml \
      --output-dir output/krea2_lora

Model weights are not distributed with this file. Krea 2 model weights are
subject to the Krea 2 Community License; this source file follows the license
of the repository in which it is distributed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import tomllib
from pathlib import Path
from typing import Any


# These must be set before importing torch in the training process.
os.environ.setdefault(
    "PYTORCH_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.8",
)
os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")


ROOT = Path(__file__).resolve().parent
TOKEN_LENGTHS = (128, 256, 384, 512, 768, 1024, 2048)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
SETTINGS_FILE = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Krea2Trainer" / "ui_settings.json"


def read_dataset_entries(path: Path) -> list[dict[str, Any]]:
    """Read dataset rows from a trainer TOML and resolve image directories."""
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    datasets = document.get("datasets", [])
    if isinstance(datasets, dict):
        datasets = [datasets]
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("The file does not contain any [[datasets]] entries.")

    entries: list[dict[str, Any]] = []
    for index, dataset in enumerate(datasets, start=1):
        if not isinstance(dataset, dict):
            raise ValueError(f"Dataset {index} must be a TOML table.")
        subsets = dataset.get("subsets", [])
        if isinstance(subsets, dict):
            subsets = [subsets]
        if not isinstance(subsets, list):
            raise ValueError(f"Dataset {index} subsets must be TOML tables.")

        raw_dirs = [subset.get("image_dir") for subset in subsets if isinstance(subset, dict)]
        if not raw_dirs and dataset.get("image_dir"):
            raw_dirs = [dataset["image_dir"]]
        image_dirs: list[Path] = []
        for raw_dir in raw_dirs:
            if not raw_dir:
                continue
            image_dir = Path(str(raw_dir)).expanduser()
            if not image_dir.is_absolute():
                image_dir = path.parent / image_dir
            image_dirs.append(image_dir.resolve())

        entries.append(
            {
                "dataset": index,
                "resolution": dataset.get("resolution", "—"),
                "batch": int(dataset.get("batch_size", 1)),
                "subsets": len(subsets) or len(image_dirs),
                "image_dirs": image_dirs,
                "recursive": any(bool(subset.get("recursive_image_scan", False)) for subset in subsets if isinstance(subset, dict)),
            }
        )
    return entries


def validate_dataset_entries(entries: list[dict[str, Any]]) -> tuple[int, int]:
    """Validate dataset directories and return (directory_count, image_count)."""
    if not entries:
        raise ValueError("Add at least one dataset or image directory.")
    directory_count = 0
    image_count = 0
    for entry in entries:
        if int(entry.get("batch", 0)) < 1:
            raise ValueError(f"Dataset {entry.get('dataset', '?')} has an invalid batch size.")
        image_dirs = entry.get("image_dirs", [])
        if not image_dirs:
            raise ValueError(f"Dataset {entry.get('dataset', '?')} has no image directory.")
        for directory in image_dirs:
            directory = Path(directory)
            if not directory.is_dir():
                raise ValueError(f"Image directory not found: {directory}")
            iterator = directory.rglob("*") if entry.get("recursive", False) else directory.glob("*")
            count = sum(1 for item in iterator if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)
            if count == 0:
                raise ValueError(f"No supported images found in: {directory}")
            directory_count += 1
            image_count += count
    return directory_count, image_count


def build_dataset_document(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a trainer-compatible dataset document from UI entries."""
    datasets: list[dict[str, Any]] = []
    for entry in entries:
        resolution = entry.get("resolution", [512, 512])
        if resolution == "—":
            resolution = [512, 512]
        if isinstance(resolution, tuple):
            resolution = list(resolution)
        subsets = [
            {
                "image_dir": str(Path(directory).resolve()),
                "num_repeats": 1,
                "recursive_image_scan": bool(entry.get("recursive", True)),
            }
            for directory in entry.get("image_dirs", [])
        ]
        datasets.append(
            {
                "resolution": resolution,
                "batch_size": int(entry.get("batch", 1)),
                "caption_extension": ".txt",
                "subsets": subsets,
            }
        )
    return {"general": {"enable_bucket": False}, "datasets": datasets}


def validate_existing_krea_caches(entries: list[dict[str, Any]], max_token_length: int) -> tuple[int, int, int]:
    """Verify cache pairs for every image and report ignored legacy caches."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is required to validate Krea caches") from exc

    latent_count = 0
    text_count = 0
    legacy_count = 0
    missing_latents: list[str] = []
    missing_text: list[str] = []
    invalid_text: list[str] = []
    scanned_cache_dirs: set[Path] = set()
    for entry in entries:
        for directory in entry.get("image_dirs", []):
            directory = Path(directory)
            iterator = directory.rglob("*") if entry.get("recursive", False) else directory.glob("*")
            images = [item for item in iterator if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS]
            for image_path in images:
                # Caches are stored beside each image directory, not necessarily
                # beside the dataset root. This matters for recursive datasets.
                image_directory = image_path.parent
                latent_directory = image_directory / "latent_cache"
                text_directory = image_directory / "text_encoder_cache"

                if image_directory not in scanned_cache_dirs:
                    scanned_cache_dirs.add(image_directory)
                    legacy_count += len(list(latent_directory.glob("*_anima*.safetensors")))
                    legacy_count += len(list(text_directory.glob("*_anima*.safetensors")))
                    legacy_count += len(list((image_directory / "cache_text_encoder").glob("*_anima*.safetensors")))

                latent_matches = list(latent_directory.glob(f"{image_path.stem}_*_krea2.safetensors"))
                if not latent_matches:
                    missing_latents.append(str(image_path))
                else:
                    latent_count += 1
                text_path = text_directory / f"{image_path.stem}_krea2_qwen3_vl.safetensors"
                if not text_path.is_file():
                    missing_text.append(str(image_path))
                    continue
                with safe_open(str(text_path), framework="pt", device="cpu") as handle:
                    cached_length = int((handle.metadata() or {}).get("max_token_length", 512))
                if cached_length != max_token_length:
                    invalid_text.append(
                        f"{image_path} (cached {cached_length}, selected {max_token_length})"
                    )
                else:
                    text_count += 1
    if missing_latents or missing_text or invalid_text:
        problems: list[str] = []
        if missing_latents:
            problems.append(f"Missing Krea latent caches: {len(missing_latents)}")
        if missing_text:
            problems.append(f"Missing Krea text caches: {len(missing_text)}")
        if invalid_text:
            problems.append(f"Text caches with a different max-token value: {len(invalid_text)}")
        examples = [*missing_latents, *missing_text, *invalid_text][:5]
        details = "\n".join(f"  {item}" for item in examples)
        raise ValueError(
            "Skip cache creation requires a complete Krea cache pair for every image.\n"
            + "\n".join(problems)
            + (f"\nExamples:\n{details}" if details else "")
            + "\nUncheck 'Skip cache creation' to reuse valid files and create only the missing caches."
        )
    return latent_count, text_count, legacy_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Krea 2 partial-LoRA trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    required = parser.add_argument_group("required paths")
    required.add_argument("--model", type=Path, required=True, help="Krea 2 Raw DiT safetensors")
    required.add_argument("--qwen3-vl", type=Path, required=True, help="Qwen3-VL encoder safetensors")
    required.add_argument("--vae", type=Path, required=True, help="Qwen-Image VAE safetensors")
    required.add_argument("--dataset-config", type=Path, required=True, help="dataset TOML")
    required.add_argument("--output-dir", type=Path, required=True, help="checkpoint output directory")

    training = parser.add_argument_group("training")
    training.add_argument("--output-name", default="krea2_lora")
    training.add_argument("--steps", type=int, default=1000)
    training.add_argument("--learning-rate", type=float, default=1e-5)
    training.add_argument("--network-dim", type=int, default=32)
    training.add_argument("--network-alpha", type=float, default=16.0)
    training.add_argument("--train-blocks", default="12-27", help="LoRA block range or comma-separated list")
    training.add_argument("--network-dropout", type=float, default=0.05)
    training.add_argument("--max-token-length", type=int, choices=TOKEN_LENGTHS, default=2048)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--save-every-n-steps", type=int)
    training.add_argument("--save-every-n-epochs", type=int)
    training.add_argument("--save-state", action="store_true")
    training.add_argument("--save-state-on-train-end", action="store_true")
    training.add_argument(
        "--keep-only-latest-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="replace older optimizer-state directories when a new state is saved",
    )
    training.add_argument("--sample-prompts", type=Path)
    training.add_argument("--sample-every-n-steps", type=int)
    training.add_argument("--sample-every-n-epochs", type=int)
    training.add_argument("--sample-at-first", action="store_true")
    training.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="trade training speed for lower activation memory",
    )
    training.add_argument("--vae-cache-batch-size", type=int, default=4)
    training.add_argument("--text-cache-batch-size", type=int, default=4)
    training.add_argument(
        "--skip-cache-creation",
        action="store_true",
        help="require and directly reuse complete Krea latent and text caches",
    )

    memory = parser.add_argument_group("memory")
    memory.add_argument(
        "--blocks-to-swap",
        type=int,
        default=0,
        help="DiT blocks offloaded to CPU; increase only if the default profile still OOMs",
    )
    memory.add_argument(
        "--force-fp8-conversion",
        action="store_true",
        help="convert a non-FP8 DiT checkpoint while loading; native scaled-FP8 is preferred",
    )
    memory.add_argument("--skip-vram-check", action="store_true")
    memory.add_argument(
        "--allow-larger-resolution",
        action="store_true",
        help="allow dataset resolutions above 512x512",
    )

    execution = parser.add_argument_group("execution")
    execution.add_argument("--ui", action="store_true", help="open the read-only visual configuration interface")
    execution.add_argument("--dry-run", action="store_true", help="validate and print the generated TOML")
    execution.add_argument("--write-config", type=Path, help="write the generated TOML and exit")
    execution.add_argument("--logging-dir", type=Path)
    execution.add_argument("--stop-file", type=Path, help=argparse.SUPPRESS)
    return parser


def positive(value: int | float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def validate_paths(args: argparse.Namespace) -> None:
    for name in ("model", "qwen3_vl", "vae", "dataset_config"):
        path = getattr(args, name)
        if not path.is_file():
            raise FileNotFoundError(f"--{name.replace('_', '-')} not found: {path}")
    if args.sample_prompts is not None and not args.sample_prompts.is_file():
        raise FileNotFoundError(f"--sample-prompts not found: {args.sample_prompts}")


def validate_dataset(path: Path, allow_larger_resolution: bool) -> None:
    with path.open("rb") as handle:
        config = tomllib.load(handle)

    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("dataset TOML must contain at least one [[datasets]] entry")

    for index, dataset in enumerate(datasets):
        batch_size = int(dataset.get("batch_size", 1))
        if batch_size != 1:
            raise ValueError(f"datasets[{index}].batch_size must be 1")
        resolution = dataset.get("resolution")
        if isinstance(resolution, int):
            width = height = resolution
        elif isinstance(resolution, list) and len(resolution) == 2:
            width, height = map(int, resolution)
        else:
            raise ValueError(f"datasets[{index}].resolution must be an integer or [width, height]")
        if width % 16 or height % 16:
            raise ValueError(f"datasets[{index}].resolution must be divisible by 16")
        if width * height > 512 * 512 and not allow_larger_resolution:
            raise ValueError(
                f"datasets[{index}].resolution is {width}x{height}; use 512x512 or pass "
                "--allow-larger-resolution and accept the additional VRAM risk"
            )


def validate_profile(args: argparse.Namespace) -> None:
    positive(args.steps, "--steps")
    positive(args.learning_rate, "--learning-rate")
    positive(args.network_dim, "--network-dim")
    positive(args.network_alpha, "--network-alpha")
    if not 0 <= args.network_dropout < 1:
        raise ValueError("--network-dropout must be in the range [0, 1)")
    if not 0 <= args.blocks_to_swap <= 27:
        raise ValueError("--blocks-to-swap must be between 0 and 27")
    if args.save_every_n_steps is not None:
        positive(args.save_every_n_steps, "--save-every-n-steps")
    if args.save_every_n_epochs is not None:
        positive(args.save_every_n_epochs, "--save-every-n-epochs")
    if args.sample_every_n_steps is not None:
        positive(args.sample_every_n_steps, "--sample-every-n-steps")
    if args.sample_every_n_epochs is not None:
        positive(args.sample_every_n_epochs, "--sample-every-n-epochs")
    if args.sample_every_n_steps is not None and args.sample_every_n_epochs is not None:
        raise ValueError("Choose sampling by steps or epochs, not both")
    if (args.sample_every_n_steps or args.sample_every_n_epochs or args.sample_at_first) and args.sample_prompts is None:
        raise ValueError("--sample-prompts is required when sampling is enabled")
    positive(args.vae_cache_batch_size, "--vae-cache-batch-size")
    positive(args.text_cache_batch_size, "--text-cache-batch-size")
    selected_blocks: set[int] = set()
    for token in re.split(r"[\s,]+", args.train_blocks.strip()):
        if not token:
            continue
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", token)
        if match is None:
            raise ValueError("--train-blocks must contain ranges or indices such as 10-27 or 10,12,14")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start > end:
            start, end = end, start
        selected_blocks.update(range(start, end + 1))
    if not selected_blocks or min(selected_blocks) < 0 or max(selected_blocks) > 27:
        raise ValueError("--train-blocks indices must be between 0 and 27")


def validate_gpu(skip: bool) -> tuple[str, float] | None:
    if skip:
        return None
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed in the active environment") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA-compatible GPU was detected. On Windows, use an NVIDIA GPU with a current driver. "
            "On Linux, CUDA and ROCm builds of PyTorch expose the accelerator through this same interface."
        )
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    gib = props.total_memory / 1024**3
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            f"{props.name} does not report BF16 support. Krea 2 requires a GPU with BF16 tensor support."
        )
    if importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError("bitsandbytes is required by the AdamW8bit optimizer")
    return props.name, gib


def validate_model_precision(path: Path, force_conversion: bool) -> bool:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is not installed in the active environment") from exc
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            uses_fp8 = any("F8" in str(handle.get_slice(key).get_dtype()).upper() for key in handle.keys())
    except Exception as exc:
        raise RuntimeError(f"could not inspect DiT checkpoint precision: {exc}") from exc
    if not uses_fp8 and not force_conversion:
        raise ValueError(
            "the DiT checkpoint is not FP8; use a native scaled-FP8 checkpoint or pass "
            "--force-fp8-conversion"
        )
    return uses_fp8


def make_config(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    logging_dir = args.logging_dir or args.output_dir / "logs"
    training: dict[str, Any] = {
        "output_name": args.output_name,
        "output_dir": str(args.output_dir.resolve()),
        "logging_dir": str(logging_dir.resolve()),
        "max_train_steps": args.steps,
        "learning_rate": args.learning_rate,
        "optimizer_type": "AdamW8bit",
        "optimizer_args": ["weight_decay=0.05"],
        "lr_scheduler": "constant",
        "mixed_precision": "bf16",
        "save_precision": "bf16",
        "max_data_loader_n_workers": 0,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "gradient_checkpointing": args.gradient_checkpointing,
        "attn_mode": "torch",
        "blocks_to_swap": args.blocks_to_swap,
        "persistent_data_loader_workers": False,
        "seed": args.seed,
        "vae_batch_size": args.vae_cache_batch_size,
        "text_encoder_batch_size": args.text_cache_batch_size,
        "krea2_max_token_length": args.max_token_length,
        "skip_latent_cache_check": True,
        "skip_text_encoder_cache_check": True,
        "skip_cache_generation": args.skip_cache_creation,
        "network_module": "networks.lora_krea2",
        "network_dim": args.network_dim,
        "network_alpha": args.network_alpha,
        "network_train_unet_only": True,
        "network_dropout": args.network_dropout,
        "network_args": [
            f"train_block_indices={args.train_blocks}",
            "train_text_fusion=False",
            "train_final_layer=True",
            "frozen_prefix_no_grad=True",
            "train_llm_adapter=False",
        ],
        "train_transformer": True,
        "train_embedders": False,
        "train_final_layer": True,
        "train_llm_adapter": False,
        "timestep_sample_method": "logit_normal",
        "discrete_flow_shift": 1.0,
        "sigmoid_scale": 1.0,
        "weighting_scheme": "logit_normal",
    }
    if args.save_every_n_steps is not None:
        training["save_every_n_steps"] = args.save_every_n_steps
    if args.save_every_n_epochs is not None:
        training["save_every_n_epochs"] = args.save_every_n_epochs
    if args.save_state:
        training["save_state"] = True
    if args.save_state_on_train_end:
        training["save_state_on_train_end"] = True
    if args.save_state or args.save_state_on_train_end:
        training["keep_only_latest_state"] = args.keep_only_latest_state
    if args.sample_prompts is not None:
        training["sample_prompts"] = str(args.sample_prompts.resolve())
    if args.sample_every_n_steps is not None:
        training["sample_every_n_steps"] = args.sample_every_n_steps
    if args.sample_every_n_epochs is not None:
        training["sample_every_n_epochs"] = args.sample_every_n_epochs
    if args.sample_at_first:
        training["sample_at_first"] = True
    if args.stop_file is not None:
        training["stop_file"] = str(args.stop_file.resolve())

    return {
        "model_arguments": {
            "pretrained_model_name_or_path": str(args.model.resolve()),
            "qwen3_vl": str(args.qwen3_vl.resolve()),
            "vae": str(args.vae.resolve()),
            "fp8_scaled": args.force_fp8_conversion,
        },
        "dataset_arguments": {
            "dataset_config": str(args.dataset_config.resolve()),
            "cache_latents_to_disk": True,
            "cache_text_encoder_outputs_to_disk": True,
        },
        "training_arguments": training,
    }


def dump_toml(config: dict[str, dict[str, Any]]) -> str:
    try:
        import toml
    except ImportError as exc:
        raise RuntimeError("The repository dependency 'toml' is not installed") from exc
    return toml.dumps(config)


def run_training(config_path: Path) -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import torch
    from krea2_train_network import Krea2NetworkTrainer, setup_parser
    from library import train_util

    parser = setup_parser()
    # read_config_from_file performs a second parse using sys.argv. Hide the
    # launcher's own options during that parse so they are not mistaken for
    # low-level trainer arguments.
    original_argv = sys.argv
    sys.argv = [str(ROOT / "krea2_train_network.py"), "--config_file", str(config_path)]
    try:
        train_args = parser.parse_args()
        train_util.verify_command_line_training_args(train_args)
        train_args = train_util.read_config_from_file(train_args, parser)
    finally:
        sys.argv = original_argv

    torch.cuda.reset_peak_memory_stats()
    Krea2NetworkTrainer().train(train_args)
    allocated = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    print(f"Peak CUDA memory: {allocated:.2f} GiB allocated, {reserved:.2f} GiB reserved")


def launch_ui() -> None:
    """Open the local configuration UI."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError as exc:
        raise RuntimeError("Tkinter is required for --ui") from exc

    root = tk.Tk()
    root.title("Krea 2 Trainer")
    root.geometry("1000x780")
    root.minsize(820, 620)

    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
    style.configure("Subtitle.TLabel", font=("Segoe UI", 10), foreground="#555555")
    style.configure("Section.TLabelframe.Label", font=("Segoe UI", 10, "bold"))

    outer = ttk.Frame(root)
    outer.pack(fill="both", expand=True)
    canvas = tk.Canvas(outer, highlightthickness=0)
    scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    content = ttk.Frame(canvas, padding=18)
    window_id = canvas.create_window((0, 0), window=content, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    content.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window_id, width=event.width))

    ttk.Label(content, text="Krea 2 Trainer", style="Title.TLabel").pack(anchor="w")
    ttk.Label(
        content,
        text="Training configuration",
        style="Subtitle.TLabel",
    ).pack(anchor="w", pady=(2, 16))

    paths: dict[str, tk.StringVar] = {
        "model": tk.StringVar(),
        "qwen": tk.StringVar(),
        "vae": tk.StringVar(),
        "dataset": tk.StringVar(),
        "output": tk.StringVar(value=str((ROOT / "output" / "krea2_lora").resolve())),
    }

    path_frame = ttk.LabelFrame(content, text="Paths", style="Section.TLabelframe")
    path_frame.pack(fill="x", pady=(0, 12))
    path_frame.columnconfigure(1, weight=1)

    def choose_file(variable: tk.StringVar, filetypes: list[tuple[str, str]]) -> None:
        selected = filedialog.askopenfilename(parent=root, filetypes=filetypes)
        if selected:
            variable.set(selected)

    def choose_directory(variable: tk.StringVar) -> None:
        selected = filedialog.askdirectory(parent=root)
        if selected:
            variable.set(selected)

    def choose_dataset_config() -> None:
        selected = filedialog.askopenfilename(
            parent=root,
            title="Select dataset config",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
        )
        if selected:
            paths["dataset"].set(selected)
            root.after_idle(load_dataset)

    path_rows = (
        ("DiT Krea 2", "model", [("Safetensors", "*.safetensors"), ("All files", "*.*")], False),
        ("Qwen3-VL", "qwen", [("Safetensors", "*.safetensors"), ("All files", "*.*")], False),
        ("VAE", "vae", [("Safetensors", "*.safetensors"), ("All files", "*.*")], False),
        ("Dataset config", "dataset", [("TOML", "*.toml"), ("All files", "*.*")], False),
        ("Output directory", "output", [], True),
    )
    for row, (label, key, filetypes, directory) in enumerate(path_rows):
        ttk.Label(path_frame, text=label, width=16).grid(row=row, column=0, sticky="w", padx=8, pady=5)
        ttk.Entry(path_frame, textvariable=paths[key]).grid(row=row, column=1, sticky="ew", padx=4, pady=5)
        if key == "dataset":
            command = choose_dataset_config
        elif directory:
            command = lambda variable=paths[key]: choose_directory(variable)
        else:
            command = lambda variable=paths[key], kinds=filetypes: choose_file(variable, kinds)
        ttk.Button(path_frame, text="Browse…", command=command).grid(row=row, column=2, padx=8, pady=5)

    parameters: dict[str, tk.StringVar] = {
        "steps": tk.StringVar(value="1000"),
        "learning_rate": tk.StringVar(value="0.00001"),
        "network_dim": tk.StringVar(value="32"),
        "network_alpha": tk.StringVar(value="16"),
        "train_blocks": tk.StringVar(value="12-27"),
        "dropout": tk.StringVar(value="0.05"),
        "tokens": tk.StringVar(value="2048"),
        "block_swap": tk.StringVar(value="0"),
    }
    parameter_frame = ttk.LabelFrame(content, text="Training", style="Section.TLabelframe")
    parameter_frame.pack(fill="x", pady=(0, 12))
    parameter_specs = (
        ("Steps", "steps"),
        ("Learning rate", "learning_rate"),
        ("LoRA rank", "network_dim"),
        ("LoRA alpha", "network_alpha"),
        ("LoRA blocks", "train_blocks"),
        ("Dropout", "dropout"),
        ("Max tokens", "tokens"),
        ("Block swap", "block_swap"),
    )
    for column in (1, 3):
        parameter_frame.columnconfigure(column, weight=1)
    for index, (label, key) in enumerate(parameter_specs):
        row, pair = divmod(index, 2)
        column = pair * 2
        ttk.Label(parameter_frame, text=label).grid(row=row, column=column, sticky="w", padx=(8, 4), pady=6)
        ttk.Entry(parameter_frame, textvariable=parameters[key], width=20).grid(
            row=row, column=column + 1, sticky="ew", padx=(4, 12), pady=6
        )

    flags: dict[str, tk.BooleanVar] = {
        "bf16": tk.BooleanVar(value=True),
        "adamw8": tk.BooleanVar(value=True),
        "gradient_checkpointing": tk.BooleanVar(value=True),
        "cache_latents": tk.BooleanVar(value=True),
        "cache_text": tk.BooleanVar(value=True),
        "partial_lora": tk.BooleanVar(value=True),
        "frozen_prefix": tk.BooleanVar(value=True),
        "train_final": tk.BooleanVar(value=True),
        "train_text_fusion": tk.BooleanVar(value=False),
        "force_fp8": tk.BooleanVar(value=False),
    }
    flag_frame = ttk.LabelFrame(content, text="Options", style="Section.TLabelframe")
    flag_frame.pack(fill="x", pady=(0, 12))
    flag_specs = (
        ("BF16 precision", "bf16"),
        ("AdamW 8-bit", "adamw8"),
        ("Gradient checkpointing (lower VRAM, slower)", "gradient_checkpointing"),
        ("Cache latents", "cache_latents"),
        ("Cache text encoder", "cache_text"),
        ("Partial LoRA", "partial_lora"),
        ("Frozen prefix without gradients", "frozen_prefix"),
        ("Train final layer", "train_final"),
        ("Train text fusion", "train_text_fusion"),
        ("Convert BF16 DiT to FP8", "force_fp8"),
    )
    for column in range(2):
        flag_frame.columnconfigure(column, weight=1)
    for index, (label, key) in enumerate(flag_specs):
        checkbox = ttk.Checkbutton(flag_frame, text=label, variable=flags[key])
        if key not in {"gradient_checkpointing", "force_fp8"}:
            checkbox.configure(state="disabled")
        checkbox.grid(row=index // 2, column=index % 2, sticky="w", padx=10, pady=5)

    skip_cache_creation = tk.BooleanVar(value=False)
    vae_cache_batch = tk.StringVar(value="4")
    text_cache_batch = tk.StringVar(value="4")
    cache_frame = ttk.LabelFrame(content, text="Cache", style="Section.TLabelframe")
    cache_frame.pack(fill="x", pady=(0, 12))
    cache_frame.columnconfigure(3, weight=1)
    ttk.Checkbutton(
        cache_frame,
        text="Skip cache creation (require complete Krea caches)",
        variable=skip_cache_creation,
    ).grid(row=0, column=0, columnspan=4, sticky="w", padx=8, pady=6)
    ttk.Label(cache_frame, text="VAE batch").grid(row=1, column=0, sticky="w", padx=(8, 4), pady=6)
    ttk.Entry(cache_frame, textvariable=vae_cache_batch, width=10).grid(row=1, column=1, sticky="w", padx=(4, 18), pady=6)
    ttk.Label(cache_frame, text="Text encoder batch").grid(row=1, column=2, sticky="w", padx=(8, 4), pady=6)
    ttk.Entry(cache_frame, textvariable=text_cache_batch, width=10).grid(row=1, column=3, sticky="w", padx=4, pady=6)
    ttk.Label(
        cache_frame,
        text=(
            "Normal mode reuses valid caches and creates only missing files. "
            "Krea suffixes: _krea2.safetensors and _krea2_qwen3_vl.safetensors; legacy _anima files are ignored."
        ),
        style="Subtitle.TLabel",
        wraplength=1050,
    ).grid(row=2, column=0, columnspan=4, sticky="w", padx=8, pady=(2, 7))

    save_every_steps = tk.StringVar(value="100")
    save_every_epochs = tk.StringVar(value="")
    save_training_state = tk.BooleanVar(value=True)
    save_on_stop = tk.BooleanVar(value=True)
    sampling_mode = tk.StringVar(value="Epochs")
    sampling_interval = tk.StringVar(value="1")
    sample_prompts = tk.StringVar()

    save_frame = ttk.LabelFrame(content, text="Saving and sampling", style="Section.TLabelframe")
    save_frame.pack(fill="x", pady=(0, 12))
    for column in (1, 3):
        save_frame.columnconfigure(column, weight=1)
    ttk.Label(save_frame, text="Save every N steps").grid(row=0, column=0, sticky="w", padx=(8, 4), pady=6)
    ttk.Entry(save_frame, textvariable=save_every_steps).grid(row=0, column=1, sticky="ew", padx=(4, 12), pady=6)
    ttk.Label(save_frame, text="Save every N epochs").grid(row=0, column=2, sticky="w", padx=(8, 4), pady=6)
    ttk.Entry(save_frame, textvariable=save_every_epochs).grid(row=0, column=3, sticky="ew", padx=(4, 12), pady=6)
    ttk.Checkbutton(save_frame, text="Save optimizer state", variable=save_training_state).grid(
        row=1, column=0, columnspan=2, sticky="w", padx=8, pady=6
    )
    ttk.Checkbutton(
        save_frame,
        text="Save final checkpoint on Stop",
        variable=save_on_stop,
        state="disabled",
    ).grid(
        row=1, column=2, columnspan=2, sticky="w", padx=8, pady=6
    )
    ttk.Label(save_frame, text="Sampling schedule").grid(row=2, column=0, sticky="w", padx=(8, 4), pady=6)
    ttk.Combobox(
        save_frame,
        textvariable=sampling_mode,
        values=("Disabled", "Steps", "Epochs"),
        state="readonly",
    ).grid(row=2, column=1, sticky="ew", padx=(4, 12), pady=6)
    ttk.Label(save_frame, text="Every N").grid(row=2, column=2, sticky="w", padx=(8, 4), pady=6)
    ttk.Entry(save_frame, textvariable=sampling_interval).grid(row=2, column=3, sticky="ew", padx=(4, 12), pady=6)
    ttk.Label(save_frame, text="Sample prompts").grid(row=3, column=0, sticky="w", padx=(8, 4), pady=6)
    ttk.Entry(save_frame, textvariable=sample_prompts).grid(row=3, column=1, columnspan=2, sticky="ew", padx=4, pady=6)

    def choose_sample_prompts() -> None:
        selected = filedialog.askopenfilename(
            parent=root,
            title="Select sample prompts",
            filetypes=[("Text files", "*.txt"), ("TOML files", "*.toml"), ("All files", "*.*")],
        )
        if selected:
            sample_prompts.set(selected)

    ttk.Button(save_frame, text="Browse…", command=choose_sample_prompts).grid(row=3, column=3, padx=8, pady=6)

    dataset_frame = ttk.LabelFrame(content, text="Datasets", style="Section.TLabelframe")
    dataset_frame.pack(fill="both", expand=True, pady=(0, 12))
    columns = ("dataset", "resolution", "batch", "subsets", "images")
    dataset_table = ttk.Treeview(dataset_frame, columns=columns, show="headings", height=6)
    headings = {
        "dataset": "Dataset",
        "resolution": "Resolution",
        "batch": "Batch",
        "subsets": "Subsets",
        "images": "Image directories",
    }
    widths = {"dataset": 80, "resolution": 100, "batch": 70, "subsets": 75, "images": 520}
    for column in columns:
        dataset_table.heading(column, text=headings[column])
        dataset_table.column(column, width=widths[column], anchor="w")
    dataset_table.pack(fill="both", expand=True, padx=8, pady=8)

    status = tk.StringVar(value="Select a dataset config to view its entries.")
    dataset_records: dict[str, dict[str, Any]] = {}

    def save_ui_settings() -> None:
        """Persist the current form without storing model data or credentials."""
        datasets: list[dict[str, Any]] = []
        for entry in dataset_records.values():
            stored_entry = dict(entry)
            stored_entry["image_dirs"] = [str(path) for path in entry.get("image_dirs", [])]
            resolution = stored_entry.get("resolution")
            if isinstance(resolution, tuple):
                stored_entry["resolution"] = list(resolution)
            datasets.append(stored_entry)

        document = {
            "version": 1,
            "paths": {key: variable.get() for key, variable in paths.items()},
            "parameters": {key: variable.get() for key, variable in parameters.items()},
            "flags": {key: bool(variable.get()) for key, variable in flags.items()},
            "cache": {
                "skip_creation": bool(skip_cache_creation.get()),
                "vae_batch": vae_cache_batch.get(),
                "text_batch": text_cache_batch.get(),
            },
            "saving": {
                "every_steps": save_every_steps.get(),
                "every_epochs": save_every_epochs.get(),
                "optimizer_state": bool(save_training_state.get()),
                "sampling_mode": sampling_mode.get(),
                "sampling_interval": sampling_interval.get(),
                "sample_prompts": sample_prompts.get(),
            },
            "datasets": datasets,
        }
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = SETTINGS_FILE.with_suffix(".tmp")
        temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(SETTINGS_FILE)

    def restore_ui_settings() -> None:
        if not SETTINGS_FILE.is_file():
            return
        try:
            document = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise ValueError("settings root must be an object")

            for key, value in document.get("paths", {}).items():
                if key in paths and isinstance(value, str):
                    paths[key].set(value)
            for key, value in document.get("parameters", {}).items():
                if key in parameters and isinstance(value, (str, int, float)):
                    parameters[key].set(str(value))
            for key, value in document.get("flags", {}).items():
                if key in flags and isinstance(value, bool):
                    flags[key].set(value)

            cache = document.get("cache", {})
            if isinstance(cache, dict):
                skip_cache_creation.set(bool(cache.get("skip_creation", skip_cache_creation.get())))
                vae_cache_batch.set(str(cache.get("vae_batch", vae_cache_batch.get())))
                text_cache_batch.set(str(cache.get("text_batch", text_cache_batch.get())))

            saving = document.get("saving", {})
            if isinstance(saving, dict):
                save_every_steps.set(str(saving.get("every_steps", save_every_steps.get())))
                save_every_epochs.set(str(saving.get("every_epochs", save_every_epochs.get())))
                save_training_state.set(bool(saving.get("optimizer_state", save_training_state.get())))
                mode = str(saving.get("sampling_mode", sampling_mode.get()))
                sampling_mode.set(mode if mode in {"Disabled", "Steps", "Epochs"} else "Epochs")
                sampling_interval.set(str(saving.get("sampling_interval", sampling_interval.get())))
                sample_prompts.set(str(saving.get("sample_prompts", sample_prompts.get())))

            stored_datasets = document.get("datasets", [])
            if isinstance(stored_datasets, list):
                for raw_entry in stored_datasets:
                    if not isinstance(raw_entry, dict):
                        continue
                    entry = dict(raw_entry)
                    entry["image_dirs"] = [Path(value) for value in entry.get("image_dirs", []) if isinstance(value, str)]
                    insert_dataset_entry(entry)

            if dataset_records:
                status.set(f"Restored {len(dataset_records)} dataset(s) from the previous session.")
            elif paths["dataset"].get().strip() and Path(paths["dataset"].get().strip()).is_file():
                load_dataset()
            else:
                status.set("Previous settings restored.")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            status.set(f"Could not restore previous settings: {exc}")

    def insert_dataset_entry(entry: dict[str, Any]) -> str:
        resolution = entry.get("resolution", "—")
        if isinstance(resolution, (list, tuple)):
            resolution = " × ".join(map(str, resolution))
        image_dirs = "; ".join(str(path) for path in entry.get("image_dirs", []))
        item_id = dataset_table.insert(
            "",
            "end",
            values=(
                entry.get("dataset", len(dataset_records) + 1),
                resolution,
                entry.get("batch", 1),
                entry.get("subsets", 1),
                image_dirs,
            ),
        )
        dataset_records[item_id] = entry
        dataset_table.selection_set(item_id)
        dataset_table.focus(item_id)
        dataset_table.see(item_id)
        return item_id

    def load_dataset() -> None:
        dataset_table.delete(*dataset_table.get_children())
        dataset_records.clear()
        raw_path = paths["dataset"].get().strip()
        if not raw_path:
            status.set("No dataset selected.")
            return
        try:
            entries = read_dataset_entries(Path(raw_path))
            for entry in entries:
                insert_dataset_entry(entry)
            status.set(f"Loaded {len(entries)} dataset(s) from {Path(raw_path).name}.")
            update_preview()
        except (OSError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
            status.set("Could not load the dataset config.")
            messagebox.showerror("Invalid dataset", str(exc), parent=root)

    def add_image_folder() -> None:
        root.lift()
        selected = filedialog.askdirectory(parent=root, title="Select image directory", mustexist=True)
        if not selected:
            status.set("Dataset selection cancelled.")
            return
        entry = {
            "dataset": len(dataset_records) + 1,
            "resolution": [512, 512],
            "batch": 1,
            "subsets": 1,
            "image_dirs": [Path(selected).resolve()],
            "recursive": True,
        }
        try:
            _directories, image_count = validate_dataset_entries([entry])
        except ValueError as exc:
            status.set("The selected folder is not a valid image dataset.")
            messagebox.showerror("Invalid image directory", str(exc), parent=root)
            return
        insert_dataset_entry(entry)
        status.set(f"Added {Path(selected).name}: {image_count} image(s).")
        update_preview()

    def remove_selected_dataset() -> None:
        selected = dataset_table.selection()
        if not selected:
            status.set("Select a dataset row to remove.")
            return
        for item_id in selected:
            dataset_records.pop(item_id, None)
        dataset_table.delete(*selected)
        status.set("Removed the selected dataset entry.")
        update_preview()

    def validate_selected_datasets() -> None:
        try:
            directory_count, image_count = validate_dataset_entries(list(dataset_records.values()))
            cache_message = ""
            if skip_cache_creation.get():
                latents, text_outputs, legacy = validate_existing_krea_caches(
                    list(dataset_records.values()), int(parameters["tokens"].get())
                )
                cache_message = f" Krea caches: {latents} latent, {text_outputs} text. Legacy caches ignored: {legacy}."
        except (RuntimeError, ValueError) as exc:
            status.set("Dataset validation failed.")
            messagebox.showerror("Dataset validation", str(exc), parent=root)
            return
        status.set(
            f"Validation passed: {image_count} image(s) in {directory_count} "
            f"director{'y' if directory_count == 1 else 'ies'}.{cache_message}"
        )
        messagebox.showinfo("Dataset validation", status.get(), parent=root)

    def save_dataset_config() -> None:
        try:
            validate_dataset_entries(list(dataset_records.values()))
            rendered = dump_toml(build_dataset_document(list(dataset_records.values())))
        except (RuntimeError, ValueError) as exc:
            status.set("Dataset validation failed.")
            messagebox.showerror("Save dataset config", str(exc), parent=root)
            return
        selected = filedialog.asksaveasfilename(
            parent=root,
            title="Save dataset config",
            defaultextension=".toml",
            initialfile="krea2_dataset.toml",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
        )
        if not selected:
            status.set("Save cancelled.")
            return
        try:
            Path(selected).write_text(rendered, encoding="utf-8")
        except OSError as exc:
            status.set("Could not save the dataset config.")
            messagebox.showerror("Save dataset config", str(exc), parent=root)
            return
        paths["dataset"].set(selected)
        status.set(f"Saved dataset config: {Path(selected).name}")
        update_preview()

    dataset_actions = ttk.Frame(dataset_frame)
    dataset_actions.pack(fill="x", padx=8, pady=(0, 8))
    ttk.Button(dataset_actions, text="Reload config", command=load_dataset).pack(side="left")
    ttk.Button(dataset_actions, text="Add image folder…", command=add_image_folder).pack(side="left", padx=8)
    ttk.Button(dataset_actions, text="Remove selected", command=remove_selected_dataset).pack(side="left")
    ttk.Button(dataset_actions, text="Save config…", command=save_dataset_config).pack(side="left", padx=8)
    ttk.Button(dataset_actions, text="Validate", command=validate_selected_datasets).pack(side="right")

    preview_frame = ttk.LabelFrame(content, text="Configuration", style="Section.TLabelframe")
    preview_frame.pack(fill="both", expand=True, pady=(0, 12))
    preview = tk.Text(preview_frame, height=12, wrap="none", font=("Consolas", 9), relief="flat")
    preview.pack(fill="both", expand=True, padx=8, pady=8)

    def update_preview() -> None:
        enabled = [key for key, variable in flags.items() if variable.get()]
        disabled = [key for key, variable in flags.items() if not variable.get()]
        dataset_lines: list[str] = []
        selected_config = paths["dataset"].get().strip()
        if selected_config:
            dataset_lines.append(f"config = {selected_config}")
        elif dataset_records:
            dataset_lines.append("config = <unsaved>")
        else:
            dataset_lines.append("config = <not selected>")
        for index, entry in enumerate(dataset_records.values(), start=1):
            resolution = entry.get("resolution", [512, 512])
            if isinstance(resolution, (list, tuple)):
                resolution = "x".join(map(str, resolution))
            dataset_lines.extend(
                [
                    f"dataset_{index}.resolution = {resolution}",
                    f"dataset_{index}.batch = {entry.get('batch', 1)}",
                    f"dataset_{index}.images = {'; '.join(str(path) for path in entry.get('image_dirs', []))}",
                ]
            )
        lines = [
            "[models]",
            f"dit = {paths['model'].get() or '<not selected>'}",
            f"qwen3_vl = {paths['qwen'].get() or '<not selected>'}",
            f"vae = {paths['vae'].get() or '<not selected>'}",
            "",
            "[dataset]",
            *dataset_lines,
            "",
            "[training]",
            *(f"{key} = {variable.get()}" for key, variable in parameters.items()),
            f"save_every_n_steps = {save_every_steps.get() or '<disabled>'}",
            f"save_every_n_epochs = {save_every_epochs.get() or '<disabled>'}",
            f"save_optimizer_state = {save_training_state.get()}",
            f"sampling = {sampling_mode.get()} every {sampling_interval.get()}",
            f"gradient_checkpointing = {flags['gradient_checkpointing'].get()}",
            f"skip_cache_creation = {skip_cache_creation.get()}",
            f"vae_cache_batch = {vae_cache_batch.get()}",
            f"text_cache_batch = {text_cache_batch.get()}",
            "",
            "[enabled]",
            *enabled,
            "",
            "[disabled]",
            *disabled,
        ]
        preview.configure(state="normal")
        preview.delete("1.0", "end")
        preview.insert("1.0", "\n".join(lines))
        preview.configure(state="disabled")

    log_frame = ttk.LabelFrame(content, text="Training log", style="Section.TLabelframe")
    log_frame.pack(fill="both", expand=True, pady=(0, 12))
    training_log = tk.Text(log_frame, height=10, wrap="word", font=("Consolas", 9), state="disabled")
    log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=training_log.yview)
    training_log.configure(yscrollcommand=log_scrollbar.set)
    training_log.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
    log_scrollbar.pack(side="right", fill="y", padx=(0, 8), pady=8)

    process_state: dict[str, Any] = {"process": None, "stop_file": None, "close_after": False}

    def append_log(message: str) -> None:
        training_log.configure(state="normal")
        training_log.insert("end", message)
        training_log.see("end")
        training_log.configure(state="disabled")

    def optional_positive_int(variable: tk.StringVar, label: str) -> int | None:
        value = variable.get().strip()
        if not value:
            return None
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{label} must be an integer") from exc
        if parsed <= 0:
            raise ValueError(f"{label} must be greater than zero")
        return parsed

    def build_training_command() -> tuple[list[str], Path]:
        model_paths = {name: Path(paths[name].get().strip()) for name in ("model", "qwen", "vae")}
        for name, path in model_paths.items():
            if not path.is_file():
                raise ValueError(f"Select a valid {name} file")

        entries = list(dataset_records.values())
        validate_dataset_entries(entries)
        if skip_cache_creation.get():
            validate_existing_krea_caches(entries, int(parameters["tokens"].get()))
        output_value = paths["output"].get().strip()
        if not output_value:
            raise ValueError("Select an output directory")
        output_dir = Path(output_value).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset_path = output_dir / "krea2_dataset.toml"
        dataset_path.write_text(dump_toml(build_dataset_document(entries)), encoding="utf-8")
        paths["dataset"].set(str(dataset_path))

        stop_file = output_dir / ".krea2_stop_requested"
        stop_file.unlink(missing_ok=True)
        vae_batch = optional_positive_int(vae_cache_batch, "VAE cache batch")
        text_batch = optional_positive_int(text_cache_batch, "Text encoder cache batch")
        if vae_batch is None or text_batch is None:
            raise ValueError("Cache batch sizes are required")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--model",
            str(model_paths["model"].resolve()),
            "--qwen3-vl",
            str(model_paths["qwen"].resolve()),
            "--vae",
            str(model_paths["vae"].resolve()),
            "--dataset-config",
            str(dataset_path.resolve()),
            "--output-dir",
            str(output_dir.resolve()),
            "--steps",
            parameters["steps"].get().strip(),
            "--learning-rate",
            parameters["learning_rate"].get().strip(),
            "--network-dim",
            parameters["network_dim"].get().strip(),
            "--network-alpha",
            parameters["network_alpha"].get().strip(),
            "--train-blocks",
            parameters["train_blocks"].get().strip(),
            "--network-dropout",
            parameters["dropout"].get().strip(),
            "--max-token-length",
            parameters["tokens"].get().strip(),
            "--blocks-to-swap",
            parameters["block_swap"].get().strip(),
            "--vae-cache-batch-size",
            str(vae_batch),
            "--text-cache-batch-size",
            str(text_batch),
            "--stop-file",
            str(stop_file.resolve()),
        ]
        if not flags["gradient_checkpointing"].get():
            command.append("--no-gradient-checkpointing")
        if skip_cache_creation.get():
            command.append("--skip-cache-creation")
        save_steps = optional_positive_int(save_every_steps, "Save every N steps")
        save_epochs = optional_positive_int(save_every_epochs, "Save every N epochs")
        if save_steps is not None:
            command.extend(["--save-every-n-steps", str(save_steps)])
        if save_epochs is not None:
            command.extend(["--save-every-n-epochs", str(save_epochs)])
        if save_training_state.get():
            command.extend(["--save-state", "--save-state-on-train-end"])
        if flags["force_fp8"].get():
            command.append("--force-fp8-conversion")

        mode = sampling_mode.get()
        if mode != "Disabled":
            interval = optional_positive_int(sampling_interval, "Sampling interval")
            prompt_path = Path(sample_prompts.get().strip())
            if not prompt_path.is_file():
                raise ValueError("Select a valid sample prompts file")
            command.extend(["--sample-prompts", str(prompt_path.resolve())])
            command.extend(
                ["--sample-every-n-steps" if mode == "Steps" else "--sample-every-n-epochs", str(interval)]
            )
        return command, stop_file

    def finish_training(return_code: int) -> None:
        process_state["process"] = None
        stop_path = process_state.get("stop_file")
        if isinstance(stop_path, Path):
            stop_path.unlink(missing_ok=True)
        start_button.configure(state="normal")
        stop_button.configure(state="disabled")
        if return_code == 0:
            status.set("Training finished. Final checkpoint saved.")
        else:
            status.set(f"Training exited with code {return_code}. Check the log.")
        append_log(f"\n[process exited with code {return_code}]\n")
        try:
            save_ui_settings()
        except OSError as exc:
            append_log(f"[could not save interface settings: {exc}]\n")
        if process_state.get("close_after"):
            root.destroy()

    def monitor_training(process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            root.after(0, append_log, line)
        return_code = process.wait()
        root.after(0, finish_training, return_code)

    def start_training() -> None:
        if process_state["process"] is not None:
            return
        try:
            command, stop_file = build_training_command()
            save_ui_settings()
        except (OSError, RuntimeError, ValueError) as exc:
            messagebox.showerror("Cannot start training", str(exc), parent=root)
            return
        update_preview()
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creation_flags,
            )
        except OSError as exc:
            messagebox.showerror("Cannot start training", str(exc), parent=root)
            return
        process_state.update({"process": process, "stop_file": stop_file, "close_after": False})
        training_log.configure(state="normal")
        training_log.delete("1.0", "end")
        training_log.configure(state="disabled")
        append_log("Starting training…\n")
        status.set("Training is running.")
        start_button.configure(state="disabled")
        stop_button.configure(state="normal")
        threading.Thread(target=monitor_training, args=(process,), daemon=True).start()

    def stop_training() -> None:
        process = process_state.get("process")
        stop_file = process_state.get("stop_file")
        if process is None or not isinstance(stop_file, Path):
            return
        try:
            stop_file.write_text("stop\n", encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Cannot stop training", str(exc), parent=root)
            return
        status.set("Stop requested. Waiting for the current step, then saving…")
        append_log("\n[stop requested; the final checkpoint will be saved after the current step]\n")
        stop_button.configure(state="disabled")

    def close_window() -> None:
        try:
            save_ui_settings()
        except OSError as exc:
            messagebox.showerror("Cannot save settings", str(exc), parent=root)
            return
        if process_state.get("process") is not None:
            if not messagebox.askyesno(
                "Training is running",
                "Request a graceful stop, save the final checkpoint, and close afterward?",
                parent=root,
            ):
                return
            process_state["close_after"] = True
            stop_training()
            return
        root.destroy()

    button_row = ttk.Frame(content)
    button_row.pack(fill="x")
    start_button = ttk.Button(button_row, text="Start training", command=start_training)
    start_button.pack(side="left")
    stop_button = ttk.Button(button_row, text="Stop and save", command=stop_training, state="disabled")
    stop_button.pack(side="left", padx=8)
    ttk.Button(button_row, text="Refresh", command=update_preview).pack(side="left")
    ttk.Button(button_row, text="Close", command=close_window).pack(side="right")
    ttk.Label(content, textvariable=status, style="Subtitle.TLabel").pack(anchor="w", pady=(10, 0))

    restore_ui_settings()
    update_preview()
    root.protocol("WM_DELETE_WINDOW", close_window)
    root.mainloop()


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--ui" in raw_argv:
        try:
            launch_ui()
            return 0
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    args = build_parser().parse_args(raw_argv)
    try:
        validate_profile(args)
        validate_paths(args)
        validate_dataset(args.dataset_config, args.allow_larger_resolution)
        configuration_only = bool(args.dry_run or args.write_config)
        gpu = None if configuration_only else validate_gpu(args.skip_vram_check)
        if not configuration_only:
            model_uses_fp8 = validate_model_precision(args.model, args.force_fp8_conversion)
            if model_uses_fp8 and args.force_fp8_conversion:
                print("DiT checkpoint is already FP8; BF16-to-FP8 conversion was disabled.")
                args.force_fp8_conversion = False
        config = make_config(args)
        rendered = dump_toml(config)

        if gpu is not None:
            print(f"GPU: {gpu[0]} ({gpu[1]:.1f} GiB)")
        if args.dry_run:
            print(rendered, end="")
            return 0
        if args.write_config:
            args.write_config.parent.mkdir(parents=True, exist_ok=True)
            args.write_config.write_text(rendered, encoding="utf-8")
            print(f"Wrote configuration: {args.write_config}")
            return 0

        args.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".toml",
            prefix="krea2_trainer_",
            encoding="utf-8",
            delete=False,
        ) as handle:
            handle.write(rendered)
            temporary_config = Path(handle.name)
        try:
            run_training(temporary_config)
        finally:
            temporary_config.unlink(missing_ok=True)
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
