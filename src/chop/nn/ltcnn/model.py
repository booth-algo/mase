import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import QuantizationEncoder, ThermometerEncoder
from .conv_layer import LTConvLayer
from .ff_layer import LTNFeedForwardLayer
from chop.nn.dwn.group_sum import GroupSum


class LTCNN(nn.Module):
    """
    Look-Up Table Convolutional Neural Network.

    Architecture: encoder -> LUT conv layers + pooling -> LTN feed-forward -> GroupSum.
    Based on the ICLR 2026 paper on differentiable weightless neural networks.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        image_size: int = 28,
        bit_depth: int = 2,
        encoding: str = "quantization",
        n: int = 4,
        conv_channels: list = None,
        kernel_size: int = 3,
        ff_hidden_sizes: list = None,
        tau: float = 10.0,
        learnable_mapping: bool = False,
        Q: int = None,
    ):
        super().__init__()

        if conv_channels is None:
            conv_channels = [4, 8]
        if ff_hidden_sizes is None:
            ff_hidden_sizes = [500, 200]
        if not conv_channels or not ff_hidden_sizes:
            raise ValueError("conv_channels and ff_hidden_sizes must be non-empty")

        self.num_classes = num_classes
        self.n = n

        self.encoder = self._build_encoder(encoding, bit_depth)
        encoded_channels = in_channels * bit_depth

        self.conv_layers = self._build_conv_stack(
            encoded_channels, conv_channels, kernel_size, n, Q
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        flat_size = self._compute_flat_size(image_size, len(conv_channels), conv_channels[-1])
        self.ff_layers, self.group_sum = self._build_classifier(
            flat_size, ff_hidden_sizes, n, num_classes, tau, learnable_mapping
        )

    @staticmethod
    def _build_encoder(encoding: str, bit_depth: int) -> nn.Module:
        if encoding == "thermometer":
            return ThermometerEncoder(bit_depth)
        return QuantizationEncoder(bit_depth)

    @staticmethod
    def _build_conv_stack(
        in_ch: int, channels: list, kernel_size: int, n: int, Q: int
    ) -> nn.ModuleList:
        layers = nn.ModuleList()
        ch_in = in_ch
        for ch_out in channels:
            layers.append(
                LTConvLayer(ch_in, ch_out, kernel_size, n=n, stride=1, padding=1, Q=Q)
            )
            ch_in = ch_out
        return layers

    @staticmethod
    def _compute_flat_size(image_size: int, num_conv_layers: int, last_channels: int) -> int:
        # Each conv layer preserves spatial dims (stride=1, padding=1),
        # followed by MaxPool2d(2,2) which halves them.
        spatial = image_size
        for _ in range(num_conv_layers):
            spatial = spatial // 2
        return last_channels * spatial * spatial

    @staticmethod
    def _build_classifier(
        flat_size: int,
        ff_hidden_sizes: list,
        n: int,
        num_classes: int,
        tau: float,
        learnable_mapping: bool,
    ) -> tuple[nn.ModuleList, GroupSum]:
        layers = []
        in_size = flat_size
        for out_size in ff_hidden_sizes:
            layers.append(LTNFeedForwardLayer(in_size, out_size, n, learnable_mapping))
            in_size = out_size

        group_size = max(1, in_size // num_classes)
        final_out = num_classes * group_size
        layers.append(LTNFeedForwardLayer(in_size, final_out, n, learnable_mapping))

        return nn.ModuleList(layers), GroupSum(k=num_classes, tau=tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)

        for conv in self.conv_layers:
            x = conv(x)
            x = self.pool(x)
            x = x.clamp(0.0, 1.0)

        x = x.view(x.shape[0], -1)

        for ff in self.ff_layers:
            x = ff(x)

        return F.log_softmax(self.group_sum(x), dim=-1)
