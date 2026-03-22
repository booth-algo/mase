# Copied from torch_dwn v1.1.1 (https://github.com/alanbacellar/DWN)
# Author: Alan T. L. Bacellar <alanbacellar@gmail.com>
# Licensed under the MIT License.
# Reproduced here for research use within MASE.

import torch


def pad_if_needed(x, n):
    if not (x.size(-1) % n == 0):
        pad = torch.zeros(len(x.shape), dtype=torch.int)
        pad[-1] = n - (x.size(-1) % n)
        pad = tuple(pad.numpy().tolist())
        x = torch.nn.functional.pad(x, pad)
    return x


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x > 0).float()

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad  # torch.nn.functional.hardtanh(output_grad)


class STE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return STEFunction.apply(x)
