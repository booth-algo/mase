import torch
import torch.nn as nn
import torch.nn.functional as F

from .lut_nodes import BatchedLUTNodes


class LTNFeedForwardLayer(nn.Module):
    """Feed-forward layer using batched LUT nodes with fixed or learnable input mappings."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n: int = 4,
        learnable_mapping: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n = n
        self.learnable_mapping = learnable_mapping

        self.lut_layer = BatchedLUTNodes(out_features, n)

        if learnable_mapping:
            self.mapping_logits = nn.Parameter(
                torch.randn(out_features, n, in_features)
            )
        else:
            indices = torch.stack([
                torch.randint(0, in_features, (n,)) for _ in range(out_features)
            ])
            self.register_buffer("mapping_indices", indices)

    def _select_inputs_learnable(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.mapping_logits, dim=-1)
        return torch.einsum("bi,oni->bon", x, weights)

    def _select_inputs_fixed(self, x: torch.Tensor) -> torch.Tensor:
        idx = self.mapping_indices
        return x[:, idx.view(-1)].view(x.shape[0], self.out_features, self.n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.learnable_mapping:
            selected = self._select_inputs_learnable(x)
        else:
            selected = self._select_inputs_fixed(x)
        return self.lut_layer(selected)
