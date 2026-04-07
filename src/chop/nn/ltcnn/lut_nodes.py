import torch
import torch.nn as nn


class BatchedLUTNodes(nn.Module):
    """Batch of independent differentiable LUT nodes with multilinear interpolation."""

    def __init__(self, num_nodes: int, n_inputs: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.n_inputs = n_inputs
        self.n_entries = 2 ** n_inputs

        self.tables = nn.Parameter(torch.rand(num_nodes, self.n_entries))

        # Precompute binary decomposition of each entry index
        entry_indices = torch.arange(self.n_entries).unsqueeze(1)
        bit_positions = torch.arange(n_inputs).unsqueeze(0)
        bits = ((entry_indices >> bit_positions) & 1).float()
        self.register_buffer("bit_patterns", bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_nodes, n_inputs) in [0, 1]
        Returns:
            (batch, num_nodes)
        """
        x = x.clamp(0.0, 1.0)
        # Multilinear interpolation of LUT entries
        x_exp = x.unsqueeze(-2)
        weights = x_exp * self.bit_patterns + (1.0 - x_exp) * (1.0 - self.bit_patterns)
        weights = weights.prod(dim=-1)
        return (weights * self.tables.unsqueeze(0)).sum(dim=-1)
