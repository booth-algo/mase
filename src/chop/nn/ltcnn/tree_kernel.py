import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lut_nodes import BatchedLUTNodes


class LUTTreeKernel(nn.Module):
    """
    N-ary tree of LUT nodes that reduces a flat input vector to a single scalar.

    Replaces a CNN's dot-product kernel. Tree depth is O(log_n(N)) where N is
    the number of leaf inputs and n is the LUT arity.
    """

    def __init__(self, n_leaf_inputs: int, n: int = 4):
        super().__init__()
        self.n = n
        self.n_leaf_inputs = n_leaf_inputs
        self.layers = nn.ModuleList()
        self.layer_sizes = []

        current_size = n_leaf_inputs
        while current_size > 1:
            num_nodes = math.ceil(current_size / n)
            self.layers.append(BatchedLUTNodes(num_nodes, n))
            self.layer_sizes.append((current_size, num_nodes))
            current_size = num_nodes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_leaf_inputs) in [0, 1]
        Returns:
            (batch,)
        """
        current = x
        for lut_layer, (_, num_nodes) in zip(self.layers, self.layer_sizes):
            if current.shape[-1] % self.n != 0:
                pad_size = self.n - (current.shape[-1] % self.n)
                current = F.pad(current, (0, pad_size), value=0.0)
            current = current.view(current.shape[0], -1, self.n)
            current = lut_layer(current)
        return current.squeeze(-1)
