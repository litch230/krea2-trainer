import math
import torch

class ModularTimestepSampler:
    """
    A modular timestep sampler that caches probability distributions and supports multiple modes:
    - uniform
    - sigmoid (bell curve, derivative of sigmoid, peaks at bias)
    - sigmoid_monotonic (standard cumulative sigmoid)
    - cosine (bell curve peaking at bias)
    """
    def __init__(self, num_timesteps: int = 1000):
        self.num_timesteps = num_timesteps
        self._cache = {}

    def get_weights(self, mode: str, bias: float, scale: float, mix: float, device: torch.device) -> torch.Tensor:
        # Check cache
        key = (mode, bias, scale, mix, device)
        if key in self._cache:
            return self._cache[key]

        # Base timesteps and normalized coordinates in [-1, 1]
        timesteps = torch.arange(self.num_timesteps, device=device, dtype=torch.float32)
        normalized = (timesteps / (self.num_timesteps - 1)) * 2 - 1

        # Base uniform distribution
        uniform = torch.ones(self.num_timesteps, device=device, dtype=torch.float32) / self.num_timesteps

        if mode == 'uniform':
            weights = uniform
        elif mode == 'sigmoid':
            # Bell curve: derivative of sigmoid, peaks at bias
            s = torch.sigmoid((normalized - bias) * scale)
            sigmoid_dist = s * (1.0 - s)
            
            # Normalize the sigmoid component
            sum_val = sigmoid_dist.sum()
            if sum_val > 0:
                sigmoid_dist = sigmoid_dist / sum_val
            else:
                sigmoid_dist = uniform

            # Mix with uniform
            weights = (1.0 - mix) * uniform + mix * sigmoid_dist
        elif mode == 'sigmoid_monotonic':
            # Monotonic sigmoid
            sigmoid_dist = torch.sigmoid((normalized - bias) * scale)
            
            sum_val = sigmoid_dist.sum()
            if sum_val > 0:
                sigmoid_dist = sigmoid_dist / sum_val
            else:
                sigmoid_dist = uniform

            weights = (1.0 - mix) * uniform + mix * sigmoid_dist
        elif mode == 'cosine':
            # Cosine-based centering
            # Peak at bias, normalized to [-1, 1]
            cosine_dist = torch.cos(torch.clamp(normalized - bias, -1.0, 1.0) * (math.pi / 2.0))
            cosine_dist = torch.clamp(cosine_dist, min=0.0)
            
            sum_val = cosine_dist.sum()
            if sum_val > 0:
                cosine_dist = cosine_dist / sum_val
            else:
                cosine_dist = uniform

            weights = (1.0 - mix) * uniform + mix * cosine_dist
        else:
            raise ValueError(f"Unknown sampling mode: {mode}")

        # Final normalization check
        weights = weights / weights.sum()
        self._cache[key] = weights
        return weights

    def sample(self, num_samples: int, mode: str, bias: float, scale: float, mix: float, device: torch.device) -> torch.Tensor:
        weights = self.get_weights(mode, bias, scale, mix, device)
        indices = torch.multinomial(weights, num_samples, replacement=True)
        # Map back to continuous float in [0, 1]
        t = indices.to(torch.float32) / (self.num_timesteps - 1)
        return t

# Instantiate a global sampler
global_sampler = ModularTimestepSampler(num_timesteps=1000)
