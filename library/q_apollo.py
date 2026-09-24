from __future__ import annotations

import logging
import re
import weakref
import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from apollo_torch import APOLLOAdamW

logger = logging.getLogger(__name__)

# Cache aliases for optimizer type matching
QAPOLLO_OPTIMIZER_ALIASES = {
    "qapollo",
    "q_apollo",
    "qapolloadamw",
    "q_apollo_adamw",
    "q-apollo-adamw",
}

def is_qapollo_optimizer_type(optimizer_type: str | None) -> bool:
    if not optimizer_type:
        return False
    return optimizer_type.lower() in QAPOLLO_OPTIMIZER_ALIASES

def is_qapollo_parameter(parameter: Any) -> bool:
    return bool(getattr(parameter, "_qapollo_weight", False))

def _quantize_affine_uint(
    weight: torch.Tensor,
    *,
    group_size: int = -1,
    n_bit: int = 8,
    stochastic_round: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    original_shape = weight.shape
    if group_size is not None and group_size > 0:
        if weight.numel() % group_size != 0:
            raise ValueError(f"Tensor with {weight.numel()} elements is not divisible by group_size={group_size}")
        weight = weight.reshape(-1, group_size)
    else:
        weight = weight.reshape(1, -1) if weight.dim() == 1 else weight

    if weight.dim() != 2:
        raise ValueError(f"Expected a 2-D tensor for affine quantization, got {weight.dim()}-D")

    max_int = 2**n_bit - 1
    min_int = 0
    compute = weight.float()
    max_val = compute.amax(dim=1, keepdim=True)
    min_val = compute.amin(dim=1, keepdim=True)
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    q = compute / scales
    if stochastic_round:
        down = torch.floor(q)
        prob = q - down
        q = torch.where(torch.rand_like(prob) < prob, down + 1.0, down)
    else:
        q = torch.round(q)
    q = torch.clamp(q + zeros, min_int, max_int).reshape(original_shape).to(torch.uint8)
    return q, scales.to(torch.float32), zeros.to(torch.float32)

def _dequantize_affine_uint(
    weight: torch.Tensor,
    *,
    dtype: torch.dtype,
    group_size: int,
    scales: torch.Tensor,
    zeros: torch.Tensor,
) -> torch.Tensor:
    original_shape = weight.shape
    if group_size is not None and group_size > 0:
        dequant = weight.to(scales.dtype).reshape(-1, group_size)
    else:
        dequant = weight.to(scales.dtype).reshape(scales.shape[0], -1)
    dequant = (dequant - zeros) * scales
    return dequant.reshape(original_shape).to(dtype)

class QApolloLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Parameter, bias: Parameter | None) -> Tensor:
        float_weight = _dequantize_affine_uint(
            weight,
            dtype=x.dtype,
            group_size=int(weight.group_size),
            scales=weight.scales,
            zeros=weight.zeros,
        )
        ctx.has_bias = bias is not None
        if bias is None:
            ctx.save_for_backward(x, weight)
        else:
            ctx.save_for_backward(x, weight, bias)
        return F.linear(x, float_weight, bias)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        if ctx.has_bias:
            x, weight, bias = ctx.saved_tensors
        else:
            x, weight = ctx.saved_tensors
            bias = None
        float_weight = _dequantize_affine_uint(
            weight,
            dtype=grad_output.dtype,
            group_size=int(weight.group_size),
            scales=weight.scales,
            zeros=weight.zeros,
        )
        grad_input = grad_output @ float_weight

        grad_bias = None
        if bias is not None:
            grad_bias = grad_output.reshape(-1, bias.shape[0]).sum(0)

        out_features, in_features = weight.shape
        # Ensure grad_output and x have matching dtypes for matmul
        grad_weight = grad_output.reshape(-1, out_features).t() @ x.reshape(-1, in_features).to(grad_output.dtype)
        # Convert gradient to float32 for high-precision accumulation
        grad_weight = grad_weight.float()
        
        if getattr(weight, "float_grad", None) is None:
            weight.float_grad = grad_weight
        else:
            weight.float_grad = weight.float_grad.to(grad_weight.device, dtype=torch.float32) + grad_weight

        backward_hook = getattr(weight, "backward_hook", None)
        if callable(backward_hook):
            backward_hook(weight)

        return grad_input, None, grad_bias

class QApolloLinear(nn.Module):
    def __init__(
        self,
        source: nn.Linear,
        *,
        weight_bits: int = 8,
        group_size: int = 256,
        stochastic_round: bool = True,
    ) -> None:
        super().__init__()
        if weight_bits != 8:
            raise NotImplementedError("Q-APOLLO weight quantization currently supports only 8 bits")
        if source.weight.dim() != 2:
            raise ValueError(f"Q-APOLLO can only wrap 2-D Linear weights, got shape {tuple(source.weight.shape)}")
        if group_size > 0 and source.weight.numel() % group_size != 0:
            raise ValueError(
                f"Q-APOLLO Linear weight with {source.weight.numel()} elements is not divisible by group_size={group_size}"
            )

        q_weight, scales, zeros = _quantize_affine_uint(
            source.weight.detach(),
            group_size=group_size,
            n_bit=weight_bits,
            stochastic_round=False,
        )
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.compute_dtype = source.weight.dtype
        self.group_size = int(group_size)
        self.weight_bits = int(weight_bits)
        self.stochastic_round = bool(stochastic_round)

        self.weight = Parameter(q_weight.to(device=source.weight.device), requires_grad=False)
        self.register_buffer("scales", scales.to(device=source.weight.device))
        self.register_buffer("zeros", zeros.to(device=source.weight.device))
        if source.bias is None:
            self.bias = None
        else:
            self.bias = Parameter(source.bias.detach().clone(), requires_grad=source.bias.requires_grad)
        self._refresh_weight_attrs()

    def _refresh_weight_attrs(self) -> None:
        self.weight._qapollo_weight = True
        self.weight._qapollo_owner = weakref.ref(self)
        self.weight.scales = self.scales
        self.weight.zeros = self.zeros
        self.weight.group_size = self.group_size
        self.weight.stochastic_round = self.stochastic_round
        self.weight.float_grad = getattr(self.weight, "float_grad", None)

    def _apply(self, fn):
        super()._apply(fn)
        self._refresh_weight_attrs()
        return self

    def forward(self, input: Tensor) -> Tensor:
        self._refresh_weight_attrs()
        return QApolloLinearFunction.apply(input, self.weight, self.bias)

    @torch.no_grad()
    def dequantized_weight(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        self._refresh_weight_attrs()
        return _dequantize_affine_uint(
            self.weight,
            dtype=dtype or self.compute_dtype,
            group_size=self.group_size,
            scales=self.scales,
            zeros=self.zeros,
        )

@dataclass
class QApolloReplacementSummary:
    replaced: int = 0
    skipped: int = 0
    skipped_not_target: int = 0
    skipped_small: int = 0
    skipped_group_size: int = 0
    replaced_numel: int = 0
    replaced_names: list[str] | None = None

def _parse_target_tokens(targets: str | Iterable[str]) -> set[str]:
    if isinstance(targets, str):
        raw_tokens = re.split(r"[,+]", targets)
    else:
        raw_tokens = list(targets)
    tokens = {str(token).strip().lower() for token in raw_tokens if str(token).strip()}
    return tokens or {"all"}

def replace_linear_with_qapollo(
    model: nn.Module,
    *,
    targets: str | Iterable[str] = "all",
    weight_bits: int = 8,
    weight_group_size: int = 256,
    stochastic_round: bool = True,
    min_weight_numel: int = 16384,
    max_modules: int | None = None,
) -> QApolloReplacementSummary:
    tokens = _parse_target_tokens(targets)
    summary = QApolloReplacementSummary(replaced_names=[])

    modules = list(model.named_modules())
    module_by_name = dict(modules)
    for module_name, module in modules:
        if not isinstance(module, nn.Linear):
            continue
        
        # Simple match: if targets is 'all', match everything. Otherwise, match name tokens.
        is_target = "all" in tokens or any(token in module_name.lower() for token in tokens)
        if not is_target:
            summary.skipped_not_target += 1
            continue
            
        if module.weight.numel() < int(min_weight_numel):
            summary.skipped_small += 1
            continue
        if weight_group_size > 0 and module.weight.numel() % int(weight_group_size) != 0:
            summary.skipped_group_size += 1
            continue
        if max_modules is not None and summary.replaced >= max_modules:
            summary.skipped += 1
            continue

        parent_name, child_name = module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
        parent = module_by_name[parent_name] if parent_name else model
        parent._modules[child_name] = QApolloLinear(
            module,
            weight_bits=weight_bits,
            group_size=int(weight_group_size),
            stochastic_round=stochastic_round,
        )
        summary.replaced += 1
        summary.replaced_numel += int(module.weight.numel())
        summary.replaced_names.append(module_name)

    logger.info(f"Q-APOLLO: Replaced {summary.replaced} Linear layers with QApolloLinear modules.")
    return summary

def dequantize_qapollo_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    for module_name, module in model.named_modules():
        if not isinstance(module, QApolloLinear):
            continue
        weight_key = f"{module_name}.weight"
        if weight_key in state_dict:
            state_dict[weight_key] = module.dequantized_weight(dtype=dtype)
        state_dict.pop(f"{module_name}.scales", None)
        state_dict.pop(f"{module_name}.zeros", None)
    return state_dict

def quantize_qapollo_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    for module_name, module in model.named_modules():
        if not isinstance(module, QApolloLinear):
            continue
        
        weight_key = f"{module_name}.weight"
        scales_key = f"{module_name}.scales"
        zeros_key = f"{module_name}.zeros"
        
        if weight_key in state_dict:
            float_weight = state_dict[weight_key]
            
            # If the weights are already quantized (e.g. loading a state dict that has them), skip
            if float_weight.dtype == torch.uint8:
                continue
                
            q_weight, scales, zeros = _quantize_affine_uint(
                float_weight,
                group_size=module.group_size,
                n_bit=module.weight_bits,
                stochastic_round=False,
            )
            
            state_dict[weight_key] = q_weight
            state_dict[scales_key] = scales
            state_dict[zeros_key] = zeros
            
    return state_dict

class QApolloAdamW(APOLLOAdamW):
    """
    APOLLO Optimizer with INT8 Quantized Weight Support.
    It dequantizes the parameters before running the step, 
    and requantizes them back to INT8 afterwards.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, **kwargs):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, **kwargs)
        
        # Initialize seeds for quantized parameters since they have requires_grad=False
        # and were skipped in APOLLO's constructor.
        params_idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                params_idx += 1
                if getattr(p, "_qapollo_weight", False):
                    self.state[p]["seed"] = params_idx

    @torch.no_grad()
    def step(self, closure=None):
        quantized_params = []
        
        # 1. Pre-step: De-quantize weights and swap gradients
        for group in self.param_groups:
            for p in group["params"]:
                uses_float_grad = getattr(p, "float_grad", None) is not None
                if uses_float_grad:
                    self._average_float_grad_if_distributed(p)
                    
                    # Store original uint8 weight tensor
                    p.saved_data = p.data
                    
                    # De-quantize uint8 data to floating point tensor
                    float_weight = _dequantize_affine_uint(
                        p.data,
                        dtype=p.float_grad.dtype,
                        group_size=int(p.group_size),
                        scales=p.scales,
                        zeros=p.zeros,
                    )
                    p.data = float_weight.clone().to(p.device)
                    p.grad = p.float_grad
                    
                    quantized_params.append(p)

        # 2. Run the main optimizer step
        loss = super().step(closure)

        # 3. Post-step: Re-quantize weights and clean float gradients
        for p in quantized_params:
            q_weight, scales, zeros = _quantize_affine_uint(
                p.data,
                group_size=int(p.group_size),
                n_bit=8,
                stochastic_round=bool(getattr(p, "stochastic_round", True)),
            )
            # Write quantized weight back to p.data
            p.data = q_weight.to(p.device)
            p.scales = scales.to(p.device)
            p.zeros = zeros.to(p.device)
            
            p.grad = None
            p.float_grad = None
            
            # Sync scales/zeros back to owner QApolloLinear module
            owner_ref = getattr(p, "_qapollo_owner", None)
            owner = owner_ref() if callable(owner_ref) else None
            if owner is not None:
                owner._buffers["scales"] = p.scales
                owner._buffers["zeros"] = p.zeros
                owner._refresh_weight_attrs()
                
            if hasattr(p, "saved_data"):
                del p.saved_data

        return loss

    @staticmethod
    def _average_float_grad_if_distributed(p: torch.nn.Parameter) -> None:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        world_size = torch.distributed.get_world_size()
        if world_size <= 1:
            return
        grad_list = [torch.zeros_like(p.float_grad) for _ in range(world_size)]
        torch.distributed.all_gather(grad_list, p.float_grad)
        p.float_grad.copy_(sum(grad_list) / float(world_size))

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        for group in self.param_groups:
            for p in group["params"]:
                if getattr(p, "float_grad", None) is not None:
                    if set_to_none:
                        p.float_grad = None
                    else:
                        p.float_grad.detach_()
                        p.float_grad.zero_()
