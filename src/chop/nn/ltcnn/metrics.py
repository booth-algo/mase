"""LUT area estimation for LTCNN models."""

from .model import LTCNN


def compute_area_luts(model: LTCNN) -> int:
    """Total LUT entries: sum of num_nodes * 2^n_inputs across all LUT nodes."""
    area = 0

    for conv_layer in model.conv_layers:
        for kernel in conv_layer.kernels:
            for lut_nodes in kernel.layers:
                area += lut_nodes.num_nodes * lut_nodes.n_entries

    for ff_layer in model.ff_layers:
        lut_nodes = ff_layer.lut_layer
        area += lut_nodes.num_nodes * lut_nodes.n_entries

    return area


def compute_parameter_count(model: LTCNN) -> int:
    """Total trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
