import torch

from library.fp8_optimization_utils import calculate_fp8_maxval, quantize_weight


def test_native_fp8_weight_is_not_left_in_block_layout():
    weight = torch.zeros((128, 128), dtype=torch.float8_e4m3fn)
    max_value = calculate_fp8_maxval(4, 3)

    quantized, scale = quantize_weight(
        "blocks.0.attn.wq.weight",
        weight,
        torch.float8_e4m3fn,
        max_value,
        -max_value,
        quantization_mode="block",
        block_size=64,
    )

    assert quantized.shape == (128, 128)
    assert quantized.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32
