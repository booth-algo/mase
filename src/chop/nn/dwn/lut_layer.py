# Copied from torch_dwn v1.1.1 (https://github.com/alanbacellar/DWN)
# Author: Alan T. L. Bacellar <alanbacellar@gmail.com>
# Licensed under the MIT License.
# Reproduced here for research use within MASE.
#
# Adaptation for MASE: CUDA extension loaded via JIT (cuda_ext.py) instead of
# a pre-compiled wheel, enabling compatibility with the installed PyTorch ABI.

import torch
import torch.nn as nn

from .mapping import LearnableMapping, layer_mapping
from .utils import STEFunction


#
# JIT CUDA extension loader (MASE adaptation — replaces `import efd_cuda`)
#

_efd_cuda_ext = None
_efd_cuda_checked = False


def _get_efd_cuda():
    """Return the JIT-compiled CUDA extension, or None if unavailable."""
    global _efd_cuda_ext, _efd_cuda_checked
    if not _efd_cuda_checked:
        _efd_cuda_checked = True
        try:
            from .cuda_ext import get_cuda_ext
            _efd_cuda_ext = get_cuda_ext()
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load EFD CUDA extension: {e}")
            _efd_cuda_ext = None
    return _efd_cuda_ext


#
# EFD autograd function (paper's exact formulation)
#

class EFDFunction(torch.autograd.Function):
    """
    Extended Finite Difference (EFD) gradient estimator for LUT lookup.

    Forward:  standard LUT lookup — binarise input internally via mapping indices.
    Backward: weighted sum over ALL 2^n LUT entries with beta^(Hamming distance)
              weights (paper's formula, not simple one-bit XOR flip).

    Reference: Bacellar et al., ICML 2024
    """

    @staticmethod
    def forward(ctx, x, mapping, luts, alpha, beta):
        ext = _get_efd_cuda()
        if x.is_cuda and ext is not None:
            output = ext.forward(x, mapping, luts)
        else:
            # CPU fallback
            batch_size = x.shape[0]
            output_size, two_to_n = luts.shape
            n = mapping.shape[1]
            # Gather inputs: x[:, mapping] -> (batch, output_size, n)
            selected = x[:, mapping]  # (batch, output_size, n)
            # Convert to address: binary bits to integer
            powers = (2 ** torch.arange(n, device=x.device)).unsqueeze(0).unsqueeze(0)
            addresses = (selected * powers).sum(dim=-1).long()  # (batch, output_size)
            # Look up LUT contents
            output = luts[torch.arange(output_size, device=x.device).unsqueeze(0), addresses]
        ctx.save_for_backward(x, mapping, luts, alpha, beta)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        ext = _get_efd_cuda()
        if output_grad.is_cuda and ext is not None:
            input_grad, luts_grad = ext.backward(*ctx.saved_tensors, output_grad.contiguous())
        else:
            raise NotImplementedError(
                "EFDFunction.backward requires CUDA (forward has a CPU fallback, "
                "but backward does not). Move tensors to CUDA for training."
            )
        return input_grad, None, luts_grad, None, None


#
# Spectral regularisation (MASE addition, not in torch_dwn)
#

def spectral_reg_loss(layer: "LUTLayer", lambda_reg: float) -> torch.Tensor:
    """Spectral regularisation on LUT weights (Appendix of Bacellar et al.)."""
    w = layer.luts  # (O, 2^n)
    # Hadamard transform via recursive doubling
    n = layer.n
    h = w.clone()
    step = 1
    for _ in range(n):
        h = torch.cat([h[:, ::2] + h[:, 1::2], h[:, ::2] - h[:, 1::2]], dim=1)
        step *= 2
    return lambda_reg * (h ** 2).sum()


#
# LUTLayer
#

class LUTLayer(torch.nn.Module):
    """
    Lookup Table (LUT) layer — core trainable component of DWN.

    Copied from torch_dwn v1.1.1. See module docstring for attribution.

    Args:
        input_size:   Width of binary input vector.
        output_size:  Number of LUTs (output width).
        n:            LUT fan-in (2, 4, or 6).
        mapping:      'random' | 'learnable' | 'arange', or a (output_size, n) int32 Tensor.
        alpha:        EFD step size. Default: 0.5 * 0.75^(n-1).
        beta:         EFD Hamming-distance decay. Default: 0.25/0.75.
        ste:          Use STE binarization on LUT output.
        clamp_luts:   Clamp lut weights to [-1, 1] each forward.
        lm_tau:       Softmax temperature for learnable mapping backward.
    """

    def __init__(self, input_size, output_size, n, mapping='random', alpha=None, beta=None,
                 ste=True, clamp_luts=True, lm_tau=0.001):
        super().__init__()

        # Input Check
        if not input_size > 0:
            raise ValueError(f"input_size must be > 0, got {input_size}")
        if not output_size > 0:
            raise ValueError(f"output_size must be > 0, got {output_size}")
        if not n > 0:
            raise ValueError(f"n must be > 0, got {n}")
        if not (
            mapping in ('arange', 'random', 'learnable') or (
                isinstance(mapping, torch.Tensor) and
                mapping.dtype == torch.int32 and
                mapping.shape == torch.Size([output_size, n])
            )
        ):
            raise ValueError(
                f"mapping must be 'arange', 'random', 'learnable', or a (output_size, n) "
                f"int32 Tensor; got {mapping!r}"
            )
        if not isinstance(ste, bool):
            raise ValueError(f"ste must be a bool, got {type(ste)}")
        if not isinstance(clamp_luts, bool):
            raise ValueError(f"clamp_luts must be a bool, got {type(clamp_luts)}")

        # Vars
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.n = int(n)
        self.ste = ste
        self.clamp_luts = clamp_luts

        # Alpha and beta
        if alpha is None:
            alpha = 0.5 * 0.75**(n-1)
        if beta is None:
            beta = 0.25/0.75
        self.alpha = torch.tensor(alpha)
        self.beta = torch.tensor(beta)
        if self.alpha.dtype not in (torch.float16, torch.float32, torch.float64):
            raise ValueError(f"alpha must be a float tensor, got dtype {self.alpha.dtype}")
        if self.beta.dtype not in (torch.float16, torch.float32, torch.float64):
            raise ValueError(f"beta must be a float tensor, got dtype {self.beta.dtype}")

        # Mapping
        if isinstance(mapping, torch.Tensor):
            self.mapping = torch.nn.Parameter(mapping, requires_grad=False)
        elif mapping == 'learnable':
            self.mapping = LearnableMapping(input_size, output_size * n, tau=lm_tau)
            self.__dummy_mapping = torch.nn.Parameter(
                torch.arange(output_size * n).reshape(output_size, n).int(),
                requires_grad=False
            )
        else:
            self.mapping = torch.nn.Parameter(
                layer_mapping(input_size, n, output_size, random=(mapping == 'random')),
                requires_grad=False
            )

        # LUTs
        luts = torch.rand(output_size, 2**n, dtype=torch.float32) * 2 - 1
        self.luts = torch.nn.Parameter(luts, requires_grad=True)

    def forward(self, x):

        # Clamp LUTs
        if self.training and self.clamp_luts:
            with torch.no_grad():
                self.luts.clamp_(-1, 1)

        # Learnable Mapping
        if isinstance(self.mapping, LearnableMapping):
            x = self.mapping(x)
            mapping = self.__dummy_mapping
        else:
            mapping = self.mapping

        # EFD
        x = EFDFunction.apply(x, mapping, self.luts, self.alpha, self.beta)

        # STE
        if self.ste:
            x = STEFunction.apply(x)

        return x

    def get_lut_contents(self) -> torch.Tensor:
        """Binarised LUT contents for hardware export. Shape: (output_size, 2^n)."""
        return (self.luts > 0).int()

    def get_input_indices(self) -> torch.Tensor:
        """Hard mapping indices. Shape: (output_size, n)."""
        if isinstance(self.mapping, LearnableMapping):
            return self.mapping.weights.argmax(dim=0).view(self.output_size, self.n)
        return self.mapping.data
