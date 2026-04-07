import torch
import torch.nn as nn
import torch.nn.functional as F

from .tree_kernel import LUTTreeKernel


class LTConvLayer(nn.Module):
    """Conv layer using LUT tree kernels. All output channels run in parallel."""

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

        effective_channels = min(Q, in_channels) if Q is not None else in_channels
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
        """(B, C, H, W) -> (B, C, H_out, W_out, K, K)"""
        k, s, p = self.kernel_size, self.stride, self.padding
        if p > 0:
            x = F.pad(x, (p, p, p, p), value=0.0)
        return x.unfold(2, k, s).unfold(3, k, s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        patches = self._extract_patches(x)
        _, _, h_out, w_out = patches.shape[:4]

        if self.channel_indices is None:
            # No Q subsampling: all channels share the same patch flatten
            flat = patches.permute(0, 2, 3, 1, 4, 5).reshape(batch * h_out * w_out, -1)
            if flat.shape[-1] < self.n_leaf_inputs:
                flat = F.pad(flat, (0, self.n_leaf_inputs - flat.shape[-1]), value=0.0)

            # Run all kernels on the same flat input
            outputs = []
            for kernel in self.kernels:
                outputs.append(kernel(flat))
            out = torch.stack(outputs, dim=1)  # (B*H*W, out_channels)
            return out.view(batch, h_out, w_out, self.out_channels).permute(0, 3, 1, 2).contiguous()
        else:
            # Q subsampling: each channel selects different input channels
            outputs = []
            for oc, kernel in enumerate(self.kernels):
                sel = patches[:, self.channel_indices[oc]]
                flat = sel.permute(0, 2, 3, 1, 4, 5).reshape(batch * h_out * w_out, -1)
                if flat.shape[-1] < self.n_leaf_inputs:
                    flat = F.pad(flat, (0, self.n_leaf_inputs - flat.shape[-1]), value=0.0)
                outputs.append(kernel(flat))
            out = torch.stack(outputs, dim=1)
            return out.view(batch, h_out, w_out, self.out_channels).permute(0, 3, 1, 2).contiguous()
