from pathlib import Path

from library.train_util import remove_other_training_state_directories


def test_keep_only_latest_state_directory(tmp_path: Path):
    names = (
        "krea2_lora-step00000100-state",
        "krea2_lora-step00000200-state",
        "krea2_lora-000001-state",
        "krea2_lora-state",
        "another_model-step00000100-state",
    )
    for name in names:
        (tmp_path / name).mkdir()

    checkpoint = tmp_path / "krea2_lora-step00000200.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    keep = tmp_path / "krea2_lora-state"

    remove_other_training_state_directories(str(tmp_path), "krea2_lora", str(keep))

    assert keep.is_dir()
    assert (tmp_path / "another_model-step00000100-state").is_dir()
    assert checkpoint.is_file()
    assert not (tmp_path / "krea2_lora-step00000100-state").exists()
    assert not (tmp_path / "krea2_lora-step00000200-state").exists()
    assert not (tmp_path / "krea2_lora-000001-state").exists()
