"""
Shared utilities for DWN integration tests.

Extracted from the individual test runners to eliminate code duplication.
All function signatures and behaviour are preserved exactly.
"""

import os
import sys
import types


# Environment setup helpers

def setup_sys_path():
    """
    Add the repo root and src/ to sys.path, and install lightweight stubs
    for ``chop`` and ``chop.nn`` so that downstream imports of
    ``chop.nn.dwn.*`` succeed without triggering the heavy torchvision
    dependency chain from ``chop/__init__.py``.
    """
    _SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../src"))
    _REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
    for p in [_SRC, _REPO]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # Stub chop and chop.nn to prevent chop/__init__.py from importing torchvision
    for _pkg in ["chop", "chop.nn"]:
        if _pkg not in sys.modules:
            _mod = types.ModuleType(_pkg)
            _mod.__path__ = [os.path.join(_SRC, *_pkg.split("."))]
            _mod.__package__ = _pkg
            sys.modules[_pkg] = _mod


def setup_conda_path():
    """
    Ensure the active conda environment's ``bin/`` directory is on PATH so
    that subprocess calls to verilator, cocotb, etc. resolve correctly.
    """
    _CONDA_ENV_BIN = os.path.join(os.environ.get("CONDA_PREFIX", ""), "bin")
    if os.path.isdir(_CONDA_ENV_BIN) and _CONDA_ENV_BIN not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _CONDA_ENV_BIN + os.pathsep + os.environ.get("PATH", "")


# Pure-Python golden-model helpers

def sw_forward(x_bits, lut_layers):
    """
    Evaluate DWN LUT stack via direct table lookup.

    Args:
        x_bits     : list[int] of 0/1, length = input_size of first layer
        lut_layers : list of LUTLayer (eval mode, CPU)

    Returns:
        list[int] of 0/1, length = output_size of last layer
    """
    for layer in lut_layers:
        indices  = layer.get_input_indices().tolist()   # (out, n)
        contents = layer.get_lut_contents().tolist()    # (out, 2^n)
        out = []
        for i in range(layer.output_size):
            addr = sum(x_bits[indices[i][k]] << k for k in range(layer.n))
            out.append(int(contents[i][addr]))
        x_bits = out
    return x_bits


def group_sum_forward(lut_bits, num_classes):
    """
    Python GroupSum: count 1s per group (raw count, exactly matches RTL).

    Args:
        lut_bits   : list[int] of 0/1, length = total output bits
        num_classes: number of output classes

    Returns:
        list[int] of raw counts, one per class (values 0 to group_size)
    """
    n = len(lut_bits)
    group_size = n // num_classes
    return [
        sum(lut_bits[g * group_size:(g + 1) * group_size])
        for g in range(num_classes)
    ]


# Data loading

_DEFAULT_MNIST_CACHE = os.path.expanduser("~/.cache/dwn/mnist/mnist_features.pt")


def load_mnist_test(num_samples):
    """
    Load MNIST test set from cache (last 10 000 of 70 000 standard split).

    Returns:
        list of (img_flat, label) where img_flat is float tensor [1, 784]
    """
    import torch

    cache_path = os.environ.get("DWN_MNIST_CACHE", _DEFAULT_MNIST_CACHE)
    assert os.path.exists(cache_path), (
        f"MNIST cache not found: {cache_path}\n"
        f"Run the DWN training script once with --dataset mnist to populate it."
    )
    cached = torch.load(cache_path, map_location="cpu", weights_only=True)
    X_all, y_all = cached["X"], cached["y"]

    # Standard split: first 60 000 = train, last 10 000 = test
    X_test = X_all[60000:]
    y_test = y_all[60000:]

    n = min(num_samples, len(X_test))
    samples = []
    for i in range(n):
        img_flat = X_test[i].unsqueeze(0).float()   # [1, 784]
        label    = int(y_test[i].item())
        samples.append((img_flat, label))
    return samples
