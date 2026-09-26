# Krea 2 Trainer

A simple Windows interface for training Krea 2 LoRAs.

## Installation

Run:

```text
install.bat
```

The installer can install Python if it is missing.

## Start

Run:

```text
start_ui.bat
```

## Update

Run `update.bat` from a copy cloned with Git.

## Usage

1. Select the DiT, Qwen3-VL, and VAE files.
2. Choose an output directory.
3. Add the folder containing your images and captions.
4. Click **Validate**.
5. Adjust the training settings and click **Start training**.

Captions use the same filename as their image:

```text
dataset/
  image_001.png
  image_001.txt
```

Use **Stop and save** to stop training and save the current result. The interface remembers the settings from the previous session.

Model files are not included.

## License

Apache License 2.0. See [LICENSE.md](LICENSE.md).
