import torch
import torch.nn as nn

from .thermometer import DistributiveThermometer
from .lut_layer import LUTLayer, spectral_reg_loss
from .group_sum import GroupSum


class DWNModel(nn.Module):
    """
    Differentiable Weightless Neural Network (DWN).

    Architecture:
        DistributiveThermometer -> LUTLayer x L -> GroupSum

    Matches the reference paper (Bacellar et al., ICML 2024).
    Reference MNIST config:
        LUTLayer(I, 2000, n=6, mapping='learnable')
        LUTLayer(2000, 1000, n=6)
        GroupSum(k=10, tau=1/0.3)

    Args:
        input_features: Continuous input width (F).
        num_classes:    Output classes (C).
        num_bits:       Thermometer bits per feature (T). Paper uses 3.
        hidden_sizes:   List of LUT layer output widths, e.g. [2000, 1000].
                        If None, uses [hidden_size] * num_layers.
        hidden_size:    Uniform width when hidden_sizes is None (default 2000).
        num_layers:     Number of LUT layers when hidden_sizes is None.
        lut_n:          LUT fan-in (default 6).
        mapping_first:  Mapping for first layer (default 'learnable').
        mapping_rest:   Mapping for subsequent layers (default 'random').
        tau:            GroupSum temperature. Paper uses 1/0.3 ≈ 3.33.
        lambda_reg:     Spectral reg weight (0 = disabled, matches paper default).
    """

    def __init__(
        self,
        input_features: int,
        num_classes: int,
        num_bits: int = 3,
        hidden_sizes=None,
        hidden_size: int = 2000,
        num_layers: int = 2,
        lut_n=6,
        mapping_first: str = "learnable",
        mapping_rest: str = "random",
        tau: float = 1.0 / 0.3,
        lambda_reg: float = 0.0,
    ):
        super().__init__()

        self.input_features = input_features
        self.num_classes = num_classes
        self.num_bits = num_bits
        self.lambda_reg = lambda_reg

        # Resolve layer widths
        if hidden_sizes is None:
            hidden_sizes = [hidden_size] * num_layers
        self.hidden_sizes = list(hidden_sizes)

        # Resolve lut_n to a per-layer list (int → same for all layers)
        if isinstance(lut_n, int):
            lut_ns = [lut_n] * len(self.hidden_sizes)
        else:
            lut_ns = list(lut_n)
            assert len(lut_ns) == len(self.hidden_sizes), (
                f"lut_n list length ({len(lut_ns)}) must match number of layers "
                f"({len(self.hidden_sizes)})"
            )
        self.lut_n = lut_ns  # store as list

        # Thermometer encoder
        self.thermometer = DistributiveThermometer(num_bits=num_bits, feature_wise=True)

        # LUT layers
        thermo_out = input_features * num_bits
        lut_layers = []
        in_sz = thermo_out
        for i, out_sz in enumerate(self.hidden_sizes):
            mp = mapping_first if i == 0 else mapping_rest
            lut_layers.append(LUTLayer(input_size=in_sz, output_size=out_sz, n=lut_ns[i], mapping=mp))
            in_sz = out_sz
        self.lut_layers = nn.ModuleList(lut_layers)

        # GroupSum classifier
        self.group_sum = GroupSum(k=num_classes, tau=tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.thermometer(x)
        for layer in self.lut_layers:
            x = layer(x)
        return self.group_sum(x)

    def fit_thermometer(self, x: torch.Tensor, verbose: bool = False) -> None:
        """Fit thermometer thresholds on training data."""
        self.thermometer.fit(x, verbose=verbose)

    def get_spectral_reg_loss(self) -> torch.Tensor:
        """Spectral regularisation loss (disabled by default, lambda_reg=0)."""
        if self.lambda_reg == 0.0:
            return torch.tensor(0.0)
        return sum(spectral_reg_loss(layer, self.lambda_reg) for layer in self.lut_layers)

