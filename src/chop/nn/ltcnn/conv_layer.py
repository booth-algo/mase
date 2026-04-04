import torch
import torch.nn as nn
import torch.nn.functional as F

from .tree_kernel import LUTTreeKernel


class LTConvLayer(nn.Module):
    """
    Convolutional layer using LUT tree kernels instead of dot products.

    Each output channel has its own LUTTreeKernel that slides over spatial
    input patches. Optional channel subsampling (Q) reduces computation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        n: int = 4,
        stride: int = 1,
        padding: int = 1,
        Q: int = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n = n
        self.stride = stride
        self.padding = padding

        effective_channels = Q if Q is not None else in_channels
        self.Q = Q
        self.n_leaf_inputs = kernel_size * kernel_size * effective_channels

        if Q is not None and Q < in_channels:
            indices = torch.stack([
                torch.randperm(in_channels)[:Q] for _ in range(out_channels)
            ])
        else:
            indices = None
        self.register_buffer("channel_indices", indices)

        self.kernels = nn.ModuleList([
            LUTTreeKernel(self.n_leaf_inputs, n) for _ in range(out_channels)
        ])

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extract sliding window patches. Returns (B, C, H_out, W_out, K, K)."""
        k, s, p = self.kernel_size, self.stride, self.padding
        if p > 0:
            x = F.pad(x, (p, p, p, p), value=0.0)
        return x.unfold(2, k, s).unfold(3, k, s)

    def _flatten_patches(
        self, patches: torch.Tensor, oc: int
    ) -> torch.Tensor:
        """Select channels for output `oc` and flatten patches to (B*H*W, leaf_inputs)."""
        if self.channel_indices is not None:
            patches = patches[:, self.channel_indices[oc]]

        batch, _, h_out, w_out, kh, kw = patches.shape
        # Rearrange to (B, H_out, W_out, C*K*K), then flatten spatial into batch dim
        flat = patches.permute(0, 2, 3, 1, 4, 5).reshape(batch * h_out * w_out, -1)

        if flat.shape[-1] < self.n_leaf_inputs:
            flat = F.pad(flat, (0, self.n_leaf_inputs - flat.shape[-1]), value=0.0)
        return flat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        patches = self._extract_patches(x)
        _, _, h_out, w_out = patches.shape[:4]

        outputs = []
        for oc in range(self.out_channels):
            flat = self._flatten_patches(patches, oc)
            out = self.kernels[oc](flat).view(batch, h_out, w_out)
            outputs.append(out)

        return torch.stack(outputs, dim=1)
