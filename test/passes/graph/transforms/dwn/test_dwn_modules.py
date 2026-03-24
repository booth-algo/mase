"""Unit tests for DWN PyTorch modules."""
import pytest
import torch
import torch.nn as nn
import sys
import os
import types

# Ensure we can import from src/chop without triggering full MASE import chain
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../src'))
sys.path.insert(0, _src)
for _pkg in ['chop', 'chop.nn']:
    if _pkg not in sys.modules:
        _mod = types.ModuleType(_pkg)
        _mod.__path__ = [os.path.join(_src, *_pkg.split('.'))]
        _mod.__package__ = _pkg
        sys.modules[_pkg] = _mod

CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="EFDFunction requires CUDA")


def test_thermometer_fit_and_binarize():
    """Test DistributiveThermometer fit and binarize."""
    from chop.nn.dwn import DistributiveThermometer

    enc = DistributiveThermometer(num_bits=4, feature_wise=True)

    torch.manual_seed(42)
    X_train = torch.randn(100, 3)
    enc.fit(X_train)
    assert enc.thresholds is not None

    X_test = torch.randn(10, 3)
    output = enc.binarize(X_test)

    assert output.shape == (10, 12), f"Expected (10, 12), got {output.shape}"
    assert output.min() >= 0.0
    assert output.max() <= 1.0
    assert len(output.unique()) <= 2, f"Expected binary output, got {output.unique()}"


def test_thermometer_forward():
    """Test DistributiveThermometer forward == binarize."""
    from chop.nn.dwn import DistributiveThermometer

    enc = DistributiveThermometer(num_bits=4, feature_wise=True)
    X = torch.randn(50, 5)
    enc.fit(X)

    x_test = torch.randn(8, 5)
    assert torch.allclose(enc.forward(x_test), enc.binarize(x_test))


# LUTLayer tests (require CUDA — EFDFunction is GPU-only per paper)

@requires_cuda
def test_lut_layer_shape_random_mapping():
    """Test LUTLayer output shape with random mapping."""
    from chop.nn.dwn import LUTLayer
    device = torch.device('cuda')

    layer = LUTLayer(input_size=32, output_size=64, n=4, mapping='random').to(device)
    x = torch.randint(0, 2, (8, 32)).float().to(device)
    out = layer(x)

    assert out.shape == (8, 64), f"Expected (8, 64), got {out.shape}"


@requires_cuda
def test_lut_layer_shape_learnable_mapping():
    """Test LUTLayer output shape with learnable mapping."""
    from chop.nn.dwn import LUTLayer
    device = torch.device('cuda')

    layer = LUTLayer(input_size=32, output_size=64, n=4, mapping='learnable').to(device)
    x = torch.randint(0, 2, (8, 32)).float().to(device)
    out = layer(x)

    assert out.shape == (8, 64), f"Expected (8, 64), got {out.shape}"


@requires_cuda
def test_lut_layer_binary_output():
    """Test LUTLayer output is binary {0, 1}."""
    from chop.nn.dwn import LUTLayer
    device = torch.device('cuda')

    layer = LUTLayer(input_size=16, output_size=32, n=2, mapping='random').to(device)
    layer.eval()
    x = torch.randint(0, 2, (4, 16)).float().to(device)
    out = layer(x)

    unique_vals = out.unique()
    assert set(unique_vals.tolist()).issubset({0.0, 1.0}), \
        f"Expected binary output, got {unique_vals}"


@requires_cuda
def test_lut_layer_gradient_flow():
    """Test that gradients flow through LUTLayer (EFD + STE)."""
    from chop.nn.dwn import LUTLayer
    device = torch.device('cuda')

    layer = LUTLayer(input_size=16, output_size=32, n=2, mapping='random', ste=True).to(device)
    x = torch.rand(4, 16, device=device)

    out = layer(x)
    loss = out.sum()
    loss.backward()

    # Gradient should exist for luts (paper uses .luts not .lut_weights)
    assert layer.luts.grad is not None, "No gradient for luts"
    assert not torch.isnan(layer.luts.grad).any(), "NaN gradient"
    assert layer.luts.grad.abs().max() > 0, "All-zero gradient"


@requires_cuda
def test_lut_layer_get_contents():
    """Test get_lut_contents returns binary array of correct shape."""
    from chop.nn.dwn import LUTLayer

    layer = LUTLayer(input_size=16, output_size=32, n=4, mapping='random')
    contents = layer.get_lut_contents()

    assert contents.shape == (32, 16), f"Expected (32, 16), got {contents.shape}"  # 2^4=16


@requires_cuda
def test_spectral_reg_loss():
    """Test spectral_reg_loss returns a scalar."""
    from chop.nn.dwn import LUTLayer, spectral_reg_loss

    layer = LUTLayer(input_size=16, output_size=32, n=4, mapping='random')
    loss = spectral_reg_loss(layer, lambda_reg=1e-4)

    assert loss.dim() == 0, "spectral_reg_loss should return a scalar"
    assert loss.item() > 0, "spectral_reg_loss should be positive"


# GroupSum tests (CPU only)

def test_group_sum_shape():
    """Test GroupSum output shape."""
    from chop.nn.dwn import GroupSum

    gs = GroupSum(k=10, tau=1.0)
    x = torch.randint(0, 2, (8, 100)).float()
    out = gs(x)

    assert out.shape == (8, 10), f"Expected (8, 10), got {out.shape}"


def test_group_sum_popcount():
    """Test GroupSum correctly counts ones per group."""
    from chop.nn.dwn import GroupSum

    gs = GroupSum(k=2, tau=1.0)
    x = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    out = gs(x)

    assert out[0, 0].item() == 2.0, f"Expected 2, got {out[0,0].item()}"
    assert out[0, 1].item() == 0.0, f"Expected 0, got {out[0,1].item()}"


# DWNModel tests (require CUDA)

@requires_cuda
def test_dwn_model_forward():
    """Test DWNModel forward pass on CUDA."""
    from chop.nn.dwn import DWNModel
    device = torch.device('cuda')

    torch.manual_seed(42)
    model = DWNModel(
        input_features=10, num_classes=4, num_bits=4,
        hidden_size=40, num_layers=2, lut_n=2,
        mapping_first='random', mapping_rest='random', tau=1.0,
    )
    X_train = torch.randn(100, 10)
    model.fit_thermometer(X_train)
    model = model.to(device)

    x = torch.randn(8, 10, device=device)
    out = model(x)

    assert out.shape == (8, 4), f"Expected (8, 4), got {out.shape}"


@requires_cuda
def test_dwn_model_training_step():
    """Test a single training step with DWNModel on CUDA."""
    from chop.nn.dwn import DWNModel
    device = torch.device('cuda')

    torch.manual_seed(42)
    model = DWNModel(
        input_features=10, num_classes=4, num_bits=4,
        hidden_size=40, num_layers=2, lut_n=2,
        mapping_first='random', mapping_rest='random', tau=1.0,
        lambda_reg=1e-4,
    )
    X_train = torch.randn(100, 10)
    model.fit_thermometer(X_train)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(16, 10, device=device)
    y = torch.randint(0, 4, (16,), device=device)

    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y) + model.get_spectral_reg_loss()
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss), f"Loss is NaN"
    assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
