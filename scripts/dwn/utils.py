"""Shared utilities for DWN training and search scripts."""

import torch


def compute_area_loss(model):
    """
    Hardware-aware regularization for DWN.

    Two components:
    1. Mapping entropy loss (differentiable): for LUT layers with LearnableMapping,
       penalize high entropy of mapping attention weights. High entropy means each
       LUT input attends to many source features (high routing complexity).
       Low entropy = concentrated connections = fewer effective routing resources.

    2. Area metric (non-differentiable, logged only): sum_l(output_size_l * 2^n_l)
       = total LUT table storage across all layers. Logged as 'area_luts' for
       Pareto analysis between area and accuracy.

    Uses chunked softmax computation to avoid OOM on large LearnableMapping layers

    Returns:
        (entropy_loss: Tensor, area_luts: int)
    """
    from chop.nn.dwn.mapping import LearnableMapping

    entropy_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    area_luts = 0

    for layer in model.lut_layers:
        # Area metric (logged, not gradient-flowed)
        area_luts += layer.output_size * (2 ** layer.n)

        # Mapping entropy regularization (differentiable)
        if isinstance(layer.mapping, LearnableMapping):
            # weights: (input_size, output_size * n)
            # Each column is a distribution over input_size source features
            W = layer.mapping.weights   # (input_size, output_size * n)
            # Temperature: use the layer's mapping tau (softmax temperature)
            tau = getattr(layer.mapping, 'tau', 0.001)
            # Chunked entropy to avoid materializing full (input_size, output_size*n) tensor
            chunk_size = 2000
            num_cols = W.shape[1]
            col_entropies = []
            for start in range(0, num_cols, chunk_size):
                end = min(start + chunk_size, num_cols)
                W_chunk = W[:, start:end]
                probs_chunk = torch.softmax(W_chunk / tau, dim=0)
                log_probs_chunk = torch.log(probs_chunk + 1e-10)
                col_entropies.append(-(probs_chunk * log_probs_chunk).sum(dim=0))
            entropy = torch.cat(col_entropies).mean()  # scalar
            entropy_loss = entropy_loss + entropy

    return entropy_loss, area_luts
