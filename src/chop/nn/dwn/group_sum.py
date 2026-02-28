# Copied from torch_dwn v1.1.1 (https://github.com/alanbacellar/DWN)
# Author: Alan T. L. Bacellar <alanbacellar@gmail.com>
# Licensed under the MIT License.
# Reproduced here for research use within MASE.
# Source file: utils.py (GroupSum class)

import torch
import torch.nn as nn


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
            x: Binary input (B, W). W must be a multiple of k by model construction.
        Returns:
            Logits (B, k).
        """
        # Reshape (B, W) → (B, k, W//k) and sum the last dim.
        # Uses -1 for the last dim to avoid Proxy arithmetic in view args.
        return x.view(x.shape[0], self.k, -1).sum(dim=-1) / self.tau
