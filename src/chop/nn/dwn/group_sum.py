# Copied from torch_dwn v1.1.1 (https://github.com/alanbacellar/DWN)
# Author: Alan T. L. Bacellar <alanbacellar@gmail.com>
# Licensed under the MIT License.
# Reproduced here for research use within MASE.
# Source file: utils.py (GroupSum class)

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupSum(nn.Module):
    """
    Group-Sum aggregation layer.

    Partitions the binary input into k equal groups, sums each group,
    and divides by tau to produce class logits.

    Ported from the reference (Bacellar et al., ICML 2024).

    Args:
        k:   Number of output classes (groups).
        tau: Temperature divisor (use tau=1/0.3 ≈ 3.33 to match paper).
    """

    def __init__(self, k: int, tau: float = 1.0):
        super().__init__()
        self.k = k
        self.tau = tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Binary input (B, W). Padded to multiple of k if needed.
        Returns:
            Logits (B, k).
        """
        # Pad to multiple of k (matches reference pad_if_needed)
        remainder = x.shape[-1] % self.k
        if remainder != 0:
            x = F.pad(x, (0, self.k - remainder))

        x = x.view(*x.shape[:-1], self.k, x.shape[-1] // self.k)
        return x.sum(dim=-1) / self.tau
