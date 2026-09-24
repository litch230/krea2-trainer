import pytest
import torch

from library.krea2.krea2_mmdit import SingleMMDiTConfig, SingleStreamDiT
from networks import lora_krea2


def make_tiny_krea2(layers=4):
    config = SingleMMDiTConfig(
        features=64,
        tdim=16,
        txtdim=32,
        heads=4,
        kvheads=2,
        multiplier=1,
        layers=layers,
        patch=2,
        channels=4,
        txtheads=2,
        txtkvheads=2,
        txtlayers=2,
    )
    return SingleStreamDiT(config)


def test_parse_krea2_block_ranges():
    assert lora_krea2._parse_block_indices("2-4, 7", 8) == [2, 3, 4, 7]
    assert lora_krea2._parse_block_indices("[1, 3]", 4) == [1, 3]
    assert lora_krea2._parse_block_indices(None, 4) is None
    with pytest.raises(ValueError):
        lora_krea2._parse_block_indices("4", 4)


def test_partial_krea2_lora_creates_only_selected_blocks_and_final_layer():
    model = make_tiny_krea2()
    network = lora_krea2.create_network(
        1.0,
        4,
        4,
        None,
        [],
        model,
        train_blocks_start="2",
        train_blocks_end="3",
        train_text_fusion="False",
        train_final_layer="True",
        frozen_prefix_no_grad="True",
    )

    names = [adapter.lora_name for adapter in network.unet_loras]
    assert names
    assert all("blocks_0" not in name and "blocks_1" not in name for name in names)
    assert any("blocks_2" in name for name in names)
    assert any("blocks_3" in name for name in names)
    assert any("last_linear" in name for name in names)
    assert all("txtfusion" not in name for name in names)
    assert model.lora_trainable_block_indices == {2, 3}
    assert model.frozen_prefix_no_grad is True


def test_krea2_standard_lora_accepts_neuron_dropout_once():
    model = make_tiny_krea2()
    network = lora_krea2.create_network(
        1.0,
        4,
        4,
        None,
        [],
        model,
        neuron_dropout=0.1,
        train_block_indices="2-3",
        train_text_fusion="False",
    )

    assert network.unet_loras
    assert network.dropout == pytest.approx(0.1)
    assert all(adapter.dropout == pytest.approx(0.1) for adapter in network.unet_loras)


def test_frozen_prefix_checkpointing_preserves_lora_gradients():
    model = make_tiny_krea2()
    network = lora_krea2.create_network(
        1.0,
        4,
        4,
        None,
        [],
        model,
        train_block_indices="2-3",
        train_text_fusion="False",
        frozen_prefix_no_grad="True",
    )
    network.apply_to([], model, False, True)
    model.requires_grad_(False)
    network.requires_grad_(True)
    model.enable_gradient_checkpointing()
    model.train()
    network.train()

    prefix_requires_grad = []
    handle = model.blocks[1].register_forward_hook(
        lambda _module, _inputs, output: prefix_requires_grad.append(output.requires_grad)
    )
    batch, image_len, text_len = 1, 4, 3
    output = model(
        img=torch.randn(batch, image_len, 16),
        context=torch.randn(batch, text_len, 2, 32),
        t=torch.rand(batch),
        pos=torch.zeros(batch, image_len + text_len, 3),
        mask=torch.ones(batch, image_len + text_len, dtype=torch.bool),
    )
    output.square().mean().backward()
    handle.remove()

    assert prefix_requires_grad == [False]
    assert all(parameter.grad is not None for parameter in network.parameters())
