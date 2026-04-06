import torch

from .mapping import LearnableMapping


def compute_area_luts(model) -> int:
    """Non-differentiable LUT area metric."""
    area_luts = 0
    for layer in model.lut_layers:
        area_luts += layer.output_size * (2 ** layer.n)
    return area_luts


def compute_entropy_loss(model) -> torch.Tensor:
    """Differentiable entropy regularization on LearnableMapping weights."""
    entropy_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    chunk_size = 2000

    for layer in model.lut_layers:
        if isinstance(layer.mapping, LearnableMapping):
            W = layer.mapping.weights  # (input_size, output_size * n)
            tau = getattr(layer.mapping, "tau", 0.001)
            num_cols = W.shape[1]
            col_entropies = []
            for start in range(0, num_cols, chunk_size):
                end = min(start + chunk_size, num_cols)
                W_chunk = W[:, start:end]
                probs_chunk = torch.softmax(W_chunk / tau, dim=0)
                log_probs_chunk = torch.log(probs_chunk + 1e-10)
                col_entropies.append(-(probs_chunk * log_probs_chunk).sum(dim=0))
            entropy_loss = entropy_loss + torch.cat(col_entropies).mean()

    return entropy_loss


def compute_area_loss(model) -> tuple:
    """Backward-compatible wrapper: returns (entropy_loss, area_luts)."""
    return compute_entropy_loss(model), compute_area_luts(model)
