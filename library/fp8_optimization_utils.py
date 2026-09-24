import os
from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

from tqdm import tqdm

from library.device_utils import clean_memory_on_device
from library.safetensors_utils import MemoryEfficientSafeOpen
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def calculate_fp8_maxval(exp_bits=4, mantissa_bits=3, sign_bits=1):
    """
    Calculate the maximum representable value in FP8 format.
    Default is E4M3 format (4-bit exponent, 3-bit mantissa, 1-bit sign). Only supports E4M3 and E5M2 with sign bit.

    Args:
        exp_bits (int): Number of exponent bits
        mantissa_bits (int): Number of mantissa bits
        sign_bits (int): Number of sign bits (0 or 1)

    Returns:
        float: Maximum value representable in FP8 format
    """
    assert exp_bits + mantissa_bits + sign_bits == 8, "Total bits must be 8"
    if exp_bits == 4 and mantissa_bits == 3 and sign_bits == 1:
        return torch.finfo(torch.float8_e4m3fn).max
    elif exp_bits == 5 and mantissa_bits == 2 and sign_bits == 1:
        return torch.finfo(torch.float8_e5m2).max
    else:
        raise ValueError(f"Unsupported FP8 format: E{exp_bits}M{mantissa_bits} with sign_bits={sign_bits}")


# The following is a manual calculation method (wrong implementation for E5M2), kept for reference.
"""
# Calculate exponent bias
bias = 2 ** (exp_bits - 1) - 1

# Calculate maximum mantissa value
mantissa_max = 1.0
for i in range(mantissa_bits - 1):
    mantissa_max += 2 ** -(i + 1)

# Calculate maximum value
max_value = mantissa_max * (2 ** (2**exp_bits - 1 - bias))

return max_value
"""


def quantize_fp8(tensor, scale, fp8_dtype, max_value, min_value):
    """
    Quantize a tensor to FP8 format using PyTorch's native FP8 dtype support.

    Args:
        tensor (torch.Tensor): Tensor to quantize
        scale (float or torch.Tensor): Scale factor
        fp8_dtype (torch.dtype): Target FP8 dtype (torch.float8_e4m3fn or torch.float8_e5m2)
        max_value (float): Maximum representable value in FP8
        min_value (float): Minimum representable value in FP8

    Returns:
        torch.Tensor: Quantized tensor in FP8 format
    """
    tensor = tensor.to(torch.float32)  # ensure tensor is in float32 for division

    # Create scaled tensor
    tensor = torch.div(tensor, scale).nan_to_num_(0.0)  # handle NaN values, equivalent to nonzero_mask in previous function

    # Clamp tensor to range
    tensor = tensor.clamp_(min=min_value, max=max_value)

    # Convert to FP8 dtype
    tensor = tensor.to(fp8_dtype)

    return tensor


def optimize_state_dict_with_fp8(
    state_dict: dict,
    calc_device: Union[str, torch.device],
    target_layer_keys: Optional[list[str]] = None,
    exclude_layer_keys: Optional[list[str]] = None,
    exp_bits: int = 4,
    mantissa_bits: int = 3,
    move_to_device: bool = False,
    quantization_mode: str = "block",
    block_size: Optional[int] = 64,
):
    """
    Optimize Linear layer weights in a model's state dict to FP8 format. The state dict is modified in-place.
    This function is a static version of load_safetensors_with_fp8_optimization without loading from files.

    Args:
        state_dict (dict): State dict to optimize, replaced in-place
        calc_device (str): Device to quantize tensors on
        target_layer_keys (list, optional): Layer key patterns to target (None for all Linear layers)
        exclude_layer_keys (list, optional): Layer key patterns to exclude
        exp_bits (int): Number of exponent bits
        mantissa_bits (int): Number of mantissa bits
        move_to_device (bool): Move optimized tensors to the calculating device

    Returns:
        dict: FP8 optimized state dict
    """
    if exp_bits == 4 and mantissa_bits == 3:
        fp8_dtype = torch.float8_e4m3fn
    elif exp_bits == 5 and mantissa_bits == 2:
        fp8_dtype = torch.float8_e5m2
    else:
        raise ValueError(f"Unsupported FP8 format: E{exp_bits}M{mantissa_bits}")

    # Calculate FP8 max value
    max_value = calculate_fp8_maxval(exp_bits, mantissa_bits)
    min_value = -max_value  # this function supports only signed FP8

    # Create optimized state dict
    optimized_count = 0

    # Enumerate tarket keys
    target_state_dict_keys = []
    for key in state_dict.keys():
        # Check if it's a weight key and matches target patterns
        is_target = (target_layer_keys is None or any(pattern in key for pattern in target_layer_keys)) and key.endswith(".weight")
        is_excluded = exclude_layer_keys is not None and any(pattern in key for pattern in exclude_layer_keys)
        is_target = is_target and not is_excluded

        if is_target and isinstance(state_dict[key], torch.Tensor):
            target_state_dict_keys.append(key)

    # Process each key
    for key in tqdm(target_state_dict_keys):
        value = state_dict[key]

        # Save original device and dtype
        original_device = value.device
        original_dtype = value.dtype

        # Move to calculation device
        if calc_device is not None:
            value = value.to(calc_device)

        quantized_weight, scale_tensor = quantize_weight(key, value, fp8_dtype, max_value, min_value, quantization_mode, block_size)

        # Add to state dict using original key for weight and new key for scale
        fp8_key = key  # Maintain original key
        scale_key = key.replace(".weight", ".scale_weight")

        if not move_to_device:
            quantized_weight = quantized_weight.to(original_device)

        # keep scale shape: [1] or [out,1] or [out, num_blocks, 1]. We can determine the quantization mode from the shape of scale_weight in the patched model.
        scale_tensor = scale_tensor.to(dtype=original_dtype, device=quantized_weight.device)

        state_dict[fp8_key] = quantized_weight
        state_dict[scale_key] = scale_tensor

        optimized_count += 1

        if calc_device is not None:  # optimized_count % 10 == 0 and
            # free memory on calculation device
            clean_memory_on_device(calc_device)

    logger.info(f"Number of optimized Linear layers: {optimized_count}")
    return state_dict


def quantize_weight(
    key: str,
    tensor: torch.Tensor,
    fp8_dtype: torch.dtype,
    max_value: float,
    min_value: float,
    quantization_mode: str = "block",
    block_size: int = 64,
):
    original_shape = tensor.shape

    # Determine quantization mode
    if quantization_mode == "block":
        if tensor.ndim != 2:
            quantization_mode = "tensor"  # fallback to per-tensor
        else:
            out_features, in_features = tensor.shape
            if in_features % block_size != 0:
                quantization_mode = "channel"  # fallback to per-channel
                logger.warning(
                    f"Layer {key} with shape {tensor.shape} is not divisible by block_size {block_size}, fallback to per-channel quantization."
                )
            else:
                num_blocks = in_features // block_size
                tensor = tensor.contiguous().view(out_features, num_blocks, block_size)  # [out, num_blocks, block_size]
    elif quantization_mode == "channel":
        if tensor.ndim != 2:
            quantization_mode = "tensor"  # fallback to per-tensor

    if tensor.dtype == torch.float8_e4m3fn or tensor.dtype == torch.float8_e5m2:
        return tensor, torch.tensor(1.0)

    # Calculate scale factor (per-tensor or per-output-channel with percentile or max)
    # value shape is expected to be [out_features, in_features] for Linear weights
    if quantization_mode == "channel" or quantization_mode == "block":
        # row-wise percentile to avoid being dominated by outliers
        # result shape: [out_features, 1] or [out_features, num_blocks, 1]
        scale_dim = 1 if quantization_mode == "channel" else 2
        abs_w = torch.abs(tensor)

        # shape: [out_features, 1] or [out_features, num_blocks, 1]
        row_max = torch.max(abs_w, dim=scale_dim, keepdim=True).values
        scale = row_max / max_value

    else:
        # per-tensor
        tensor_max = torch.max(torch.abs(tensor).view(-1))
        scale = tensor_max / max_value

    # numerical safety
    scale = torch.clamp(scale, min=1e-8)
    scale = scale.to(torch.float32)  # ensure scale is in float32 for division

    # Quantize weight to FP8 (scale can be scalar or [out,1], broadcasting works)
    quantized_weight = quantize_fp8(tensor, scale, fp8_dtype, max_value, min_value)

    # If block-wise, restore original shape
    if quantization_mode == "block":
        quantized_weight = quantized_weight.view(original_shape)  # restore to original shape [out, in]

    return quantized_weight, scale


def load_safetensors_with_fp8_optimization(
    model_files: List[str],
    calc_device: Union[str, torch.device],
    target_layer_keys=None,
    exclude_layer_keys=None,
    exp_bits=4,
    mantissa_bits=3,
    move_to_device=False,
    weight_hook=None,
    quantization_mode: str = "block",
    block_size: Optional[int] = 64,
) -> dict:
    """
    Load weight tensors from safetensors files and merge LoRA weights into the state dict with explicit FP8 optimization.

    Args:
        model_files (list[str]): List of model files to load
        calc_device (str or torch.device): Device to quantize tensors on
        target_layer_keys (list, optional): Layer key patterns to target for optimization (None for all Linear layers)
        exclude_layer_keys (list, optional): Layer key patterns to exclude from optimization
        exp_bits (int): Number of exponent bits
        mantissa_bits (int): Number of mantissa bits
        move_to_device (bool): Move optimized tensors to the calculating device
        weight_hook (callable, optional): Function to apply to each weight tensor before optimization
        quantization_mode (str): Quantization mode, "tensor", "channel", or "block"
        block_size (int, optional): Block size for block-wise quantization (used if quantization_mode is "block")

    Returns:
        dict: FP8 optimized state dict
    """
    if exp_bits == 4 and mantissa_bits == 3:
        fp8_dtype = torch.float8_e4m3fn
    elif exp_bits == 5 and mantissa_bits == 2:
        fp8_dtype = torch.float8_e5m2
    else:
        raise ValueError(f"Unsupported FP8 format: E{exp_bits}M{mantissa_bits}")

    # Calculate FP8 max value
    max_value = calculate_fp8_maxval(exp_bits, mantissa_bits)
    min_value = -max_value  # this function supports only signed FP8

    # Define function to determine if a key is a target key. target means fp8 optimization, not for weight hook.
    def is_target_key(key):
        # Check if weight key matches target patterns and does not match exclude patterns
        is_target = (target_layer_keys is None or any(pattern in key for pattern in target_layer_keys)) and key.endswith(".weight")
        is_excluded = exclude_layer_keys is not None and any(pattern in key for pattern in exclude_layer_keys)
        return is_target and not is_excluded

    # Create optimized state dict
    optimized_count = 0

    # Process each file
    state_dict = {}
    for model_file in model_files:
        with MemoryEfficientSafeOpen(model_file) as f:
            keys = f.keys()
            for key in tqdm(keys, desc=f"Loading {os.path.basename(model_file)}", unit="key"):
                value = f.get_tensor(key)

                # Save original device
                original_device = value.device  # usually cpu

                if weight_hook is not None:
                    # Apply weight hook if provided
                    value = weight_hook(key, value, keep_on_calc_device=(calc_device is not None))

                if not is_target_key(key):
                    target_device = calc_device if (calc_device is not None and move_to_device) else original_device
                    value = value.to(target_device)
                    state_dict[key] = value
                    continue

                # Move to calculation device
                if calc_device is not None:
                    value = value.to(calc_device)

                original_dtype = value.dtype
                quantized_weight, scale_tensor = quantize_weight(
                    key, value, fp8_dtype, max_value, min_value, quantization_mode, block_size
                )

                # Add to state dict using original key for weight and new key for scale
                fp8_key = key  # Maintain original key
                scale_key = key.replace(".weight", ".scale_weight")
                assert fp8_key != scale_key, "FP8 key and scale key must be different"

                if not move_to_device:
                    quantized_weight = quantized_weight.to(original_device)

                # keep scale shape: [1] or [out,1] or [out, num_blocks, 1]. We can determine the quantization mode from the shape of scale_weight in the patched model.
                scale_tensor = scale_tensor.to(dtype=original_dtype, device=quantized_weight.device)

                state_dict[fp8_key] = quantized_weight
                state_dict[scale_key] = scale_tensor

                optimized_count += 1

                if calc_device is not None and optimized_count % 10 == 0:
                    # free memory on calculation device
                    clean_memory_on_device(calc_device)

    logger.info(f"Number of optimized Linear layers: {optimized_count}")
    return state_dict


class FP8DequantizeLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, scale_weight, bias):
        ctx.save_for_backward(x, weight, scale_weight, bias)
        target_dtype = x.dtype
        target_device = x.device
        w = weight.to(device=target_device, dtype=target_dtype)
        s = scale_weight.to(device=target_device, dtype=target_dtype)
        if s.ndim < 3:
            dequant = w * s
        else:
            out_features, num_blocks, _ = s.shape
            dequant = w.contiguous().view(out_features, num_blocks, -1) * s
            dequant = dequant.view(w.shape)
        b = bias.to(device=target_device, dtype=target_dtype) if bias is not None else None
        return F.linear(x, dequant, b)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, scale_weight, bias = ctx.saved_tensors
        target_dtype = x.dtype
        target_device = x.device
        w = weight.to(device=target_device, dtype=target_dtype)
        s = scale_weight.to(device=target_device, dtype=target_dtype)
        if s.ndim < 3:
            dequant = w * s
        else:
            out_features, num_blocks, _ = s.shape
            dequant = w.contiguous().view(out_features, num_blocks, -1) * s
            dequant = dequant.view(w.shape)

        grad_x = grad_output.matmul(dequant)
        return grad_x, None, None, None


class FP8ScaledFrozenLinearFunction(torch.autograd.Function):
    """FP8 GEMM for a frozen Linear weight (the LoRA path owns trainable weights)."""

    @staticmethod
    def _quantize_per_tensor(value, fp8_dtype):
        fp8_max = torch.finfo(fp8_dtype).max
        scale = value.detach().abs().amax().float().clamp_min(1e-12) / fp8_max
        quantized = (value / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
        return quantized, scale

    @staticmethod
    def forward(ctx, x, weight, scale_weight, bias):
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1])
        x_fp8, scale_x = FP8ScaledFrozenLinearFunction._quantize_per_tensor(x_2d, weight.dtype)
        output = torch._scaled_mm(
            x_fp8,
            weight.t(),
            scale_x,
            scale_weight.float(),
            out_dtype=x.dtype,
            use_fast_accum=True,
        )
        if bias is not None:
            output = output + bias.to(device=output.device, dtype=output.dtype)
        ctx.save_for_backward(weight, scale_weight)
        ctx.input_shape = original_shape
        return output.reshape(*original_shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        weight, scale_weight = ctx.saved_tensors
        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
        row_count = grad_2d.shape[0]
        padded_rows = (row_count + 15) // 16 * 16
        if padded_rows != row_count:
            grad_2d = F.pad(grad_2d, (0, 0, 0, padded_rows - row_count))

        # Use the same E4M3 type as the frozen checkpoint weight. Mixed
        # E5M2/E4M3 GEMMs are prohibitively slow on consumer Blackwell in the
        # current CUDA/PyTorch stack. Numerical tests (including a sparse
        # 10,000x gradient outlier) keep relative grad_x error below 1% here.
        grad_fp8, scale_grad = FP8ScaledFrozenLinearFunction._quantize_per_tensor(
            grad_2d, weight.dtype
        )

        # grad_x = grad_output @ weight. Build a column-major view of the
        # stored row-major weight for cuBLASLt's FP8 mat2 layout requirement.
        weight_col_major = weight.t().contiguous().t()
        grad_x = torch._scaled_mm(
            grad_fp8,
            weight_col_major,
            scale_grad,
            scale_weight.float(),
            out_dtype=grad_output.dtype,
            use_fast_accum=True,
        )
        grad_x = grad_x[:row_count].reshape(ctx.input_shape)
        return grad_x, None, None, None


def fp8_linear_forward_patch(self, x, use_scaled_mm=False, max_value=None):
    del max_value
    scalar_scale = self.scale_weight.numel() == 1
    if use_scaled_mm and hasattr(torch, "_scaled_mm") and scalar_scale and x.is_cuda:
        return FP8ScaledFrozenLinearFunction.apply(x, self.weight, self.scale_weight, self.bias)
    return FP8DequantizeLinearFunction.apply(x, self.weight, self.scale_weight, self.bias)


def apply_fp8_monkey_patch(
    model,
    optimized_state_dict,
    use_scaled_mm=False,
    full_precision_module_paths=None,
):
    """
    Apply monkey patching to a model using FP8 optimized state dict.

    Args:
        model (nn.Module): Model instance to patch
        optimized_state_dict (dict): FP8 optimized state dict
        use_scaled_mm (bool): Use scaled_mm for FP8 Linear layers, requires SM 8.9+ (RTX 40 series)

    Returns:
        nn.Module: The patched model (same instance, modified in-place)
    """
    # # Calculate FP8 float8_e5m2 max value
    # max_value = calculate_fp8_maxval(5, 2)
    max_value = None  # do not quantize input tensor
    full_precision_module_paths = set(full_precision_module_paths or ())

    # Find all scale keys (supporting both .scale_weight and .weight_scale)
    for k in list(optimized_state_dict.keys()):
        if k.endswith(".weight_scale"):
            new_key = k.rsplit(".weight_scale", 1)[0] + ".scale_weight"
            optimized_state_dict[new_key] = optimized_state_dict.pop(k)

    scale_keys = [k for k in optimized_state_dict.keys() if k.endswith(".scale_weight")]

    # Enumerate patched layers
    patched_module_paths = set()
    scale_shape_info = {}
    for scale_key in scale_keys:
        # Extract module path from scale key (remove .scale_weight)
        module_path = scale_key.rsplit(".scale_weight", 1)[0]
        patched_module_paths.add(module_path)

        # Store scale shape information
        scale_shape_info[module_path] = optimized_state_dict[scale_key].shape

    patched_count = 0

    # Apply monkey patch to each layer with FP8 weights
    for name, module in model.named_modules():
        # Check if this module has a corresponding scale_weight
        has_scale = name in patched_module_paths

        # Apply patch if it's a Linear layer with FP8 scale
        if isinstance(module, nn.Linear) and has_scale:
            # register the scale_weight as a buffer to load the state_dict
            # module.register_buffer("scale_weight", torch.tensor(1.0, dtype=module.weight.dtype))
            scale_shape = scale_shape_info[name]
            module.register_buffer("scale_weight", torch.ones(scale_shape, dtype=module.weight.dtype))

            # Create a new forward method with the patched version.
            # Comfy's scaled-FP8 metadata marks numerically sensitive output
            # projections for a BF16/FP32-style matrix multiplication.  The
            # weights remain stored as FP8, but must be dequantized before the
            # GEMM instead of quantizing the activation too.
            module_use_scaled_mm = use_scaled_mm and name not in full_precision_module_paths

            def new_forward(self, x, _use_scaled_mm=module_use_scaled_mm):
                return fp8_linear_forward_patch(self, x, _use_scaled_mm, max_value)

            # Bind method to module
            module.forward = new_forward.__get__(module, type(module))

            patched_count += 1

    logger.info(f"Number of monkey-patched Linear layers: {patched_count}")
    return model
