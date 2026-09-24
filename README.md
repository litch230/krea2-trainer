# Krea 2 Trainer

Small Windows GUI for training Krea 2 LoRAs.

## Install

Requirements:

- Windows 10 or 11
- 64-bit Python 3.10, 3.11, or 3.12
- NVIDIA GPU with BF16 support
- Recent NVIDIA driver

Run:

```text
install.bat
```

The script creates `.venv` and installs PyTorch and the remaining packages. It uses the CUDA 12.8 PyTorch wheels when `nvidia-smi` is available.

To use another PyTorch index:

```bat
set KREA2_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
install.bat
```

## Start

Run:

```text
start_ui.bat
```

Or start it from PowerShell:

```powershell
Set-Location "D:\Krea2 Trainer"
.\.venv\Scripts\python.exe .\krea2_trainer.py --ui
```

## Model files

The weights are not included. Select these files in the GUI:

- Krea 2 Raw DiT checkpoint (`.safetensors`)
- compatible Qwen3-VL text encoder checkpoint
- compatible Qwen-Image VAE checkpoint

A native scaled-FP8 DiT is recommended. Use **Convert DiT to FP8** only when the selected DiT is not already FP8.

## Dataset

Each image should have a caption with the same base name:

```text
dataset/
  image_001.png
  image_001.txt
  image_002.png
  image_002.txt
```

Click **Add image folder**, select the directory, then click **Validate**. The GUI writes `krea2_dataset.toml` to the output directory when training starts.

Supported images: PNG, JPEG, WebP, BMP, and TIFF.

## Cache files

The trainer reuses valid caches and creates only the missing ones.

- Latents: `*_krea2.safetensors`
- Text encoder: `*_krea2_qwen3_vl.safetensors`

**Skip cache creation** requires both cache files for every image. Leave it disabled on the first run.

## Memory settings

If caching runs out of memory, reduce **VAE batch** and **Text encoder batch** to `1` or `2`.

If training runs out of memory:

1. Keep **Gradient checkpointing** enabled.
2. Increase **Block swap** a few blocks at a time.
3. Reduce max tokens if the captions do not need `2048` tokens.
4. Reduce the training resolution if necessary.

When the training already fits, disabling gradient checkpointing can improve step speed.

The Windows installer targets NVIDIA CUDA. AMD ROCm builds of PyTorch are available on Linux and must be installed manually.

## Saving and sampling

Default settings:

- save every 100 steps
- save optimizer state
- save a final checkpoint after **Stop and save**
- sample every epoch

Sampling needs a prompts file. Select one in **Sample prompts**, or set **Sampling schedule** to **Disabled**.

## Command line

```powershell
.\.venv\Scripts\python.exe .\krea2_trainer.py --help
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q -s
```

## License

Code: Apache License 2.0. See [LICENSE.md](LICENSE.md).

Model weights have their own licenses and are not part of this repository.
