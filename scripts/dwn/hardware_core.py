"""Shared DWN hardware core wrapper for RTL emission scripts."""

import torch.nn as nn


class DWNHardwareCore(nn.Module):
    """Wrapper that contains only the LUT layers for hardware emission."""

    def __init__(self, lut_layers):
        super().__init__()
        self.lut_layers = nn.ModuleList(lut_layers)

    def forward(self, x):
        for layer in self.lut_layers:
            x = layer(x)
        return x
