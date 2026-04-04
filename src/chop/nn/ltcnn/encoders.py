import torch
import torch.nn as nn


class QuantizationEncoder(nn.Module):
    """Converts continuous [0,1] pixel values into binary vectors via uniform quantization."""

    def __init__(self, bit_depth: int = 2):
        super().__init__()
        self.bit_depth = bit_depth
        self.n_levels = 2 ** bit_depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_quant = (x * (self.n_levels - 1)).round().long().clamp(0, self.n_levels - 1)
        bits = [((x_quant >> b) & 1).float() for b in range(self.bit_depth)]
        return torch.cat(bits, dim=1)


class ThermometerEncoder(nn.Module):
    """Converts continuous [0,1] pixel values into binary vectors via threshold comparison."""

    def __init__(self, bit_depth: int = 2):
        super().__init__()
        self.bit_depth = bit_depth
        thresholds = torch.tensor([(i + 1) / (bit_depth + 1) for i in range(bit_depth)])
        self.register_buffer("thresholds", thresholds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bits = [(x > t).float() for t in self.thresholds]
        return torch.cat(bits, dim=1)
