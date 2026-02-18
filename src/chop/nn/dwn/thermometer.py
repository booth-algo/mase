# Copied from torch_dwn v1.1.1 (https://github.com/alanbacellar/DWN)
# Author: Alan T. L. Bacellar <alanbacellar@gmail.com>
# Licensed under the MIT License.
# Reproduced here for research use within MASE.
# Source file: binarization.py

import torch
import torch.nn as nn


class DistributiveThermometer(nn.Module):
    """
    Distributive thermometer encoding.

    Maps each continuous feature to num_bits binary values by comparing against
    evenly-spaced quantile thresholds computed from training data.

    Ported from the reference implementation (Bacellar et al., ICML 2024).

    Args:
        num_bits:     Number of thresholds T per feature.
        feature_wise: If True, compute separate thresholds per feature.
    """

    def __init__(self, num_bits: int, feature_wise: bool = True):
        super().__init__()
        self.num_bits = num_bits
        self.feature_wise = feature_wise
        self.register_buffer("thresholds", None)

    def _get_thresholds(self, x: torch.Tensor) -> torch.Tensor:
        """Compute evenly-spaced quantile thresholds via sort (matches reference)."""
        T = self.num_bits
        if self.feature_wise:
            data = torch.sort(x, dim=0)[0]         # (N, F) sorted per feature
            N = data.shape[0]
            indices = torch.tensor(
                [int(N * i / (T + 1)) for i in range(1, T + 1)],
                dtype=torch.long,
            )
            thresholds = data[indices]              # (T, F)
            return thresholds.permute(*range(1, thresholds.ndim), 0)  # (F, T)
        else:
            data = torch.sort(x.flatten())[0]      # (N*F,)
            N = data.shape[0]
            indices = torch.tensor(
                [int(N * i / (T + 1)) for i in range(1, T + 1)],
                dtype=torch.long,
            )
            return data[indices]                    # (T,)

    def fit(self, x: torch.Tensor, verbose: bool = False) -> "DistributiveThermometer":
        """
        Fit thresholds from training data.

        Args:
            x:       Training data (N, F).
            verbose: Print progress.
        Returns:
            self (for chaining: thermometer.fit(x_train))
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)

        T = self.num_bits
        if verbose:
            N, F = x.shape
            print(
                f"  Computing {T} quantile thresholds for {F} features "
                f"over {N} samples...",
                flush=True,
            )

        self.thresholds = self._get_thresholds(x)

        if verbose:
            print(f"  Done. Thresholds shape: {self.thresholds.shape}", flush=True)

        return self

    def binarize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply thermometer encoding.

        Args:
            x: Input (*, F).
        Returns:
            Binary tensor (*, F*T).
        """
        if self.thresholds is None:
            raise RuntimeError("Must call fit() before binarize()/forward().")

        *batch_dims, F = x.shape
        T = self.num_bits

        # x unsqueeze(-1): (*, F, 1)  thresholds: (F, T) or (T,)
        out = (x.unsqueeze(-1) > self.thresholds).float()  # (*, F, T)
        return out.reshape(*batch_dims, F * T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.binarize(x)
