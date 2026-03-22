"""
Pytest runner for DWN MNIST full simulation.

Loads the trained MNIST n2 DWN model checkpoint and uses it as a "golden
reference" to generate test vectors for a UVM-style cocotb simulation of
the emitted dwn_top.sv RTL.

Key features:
  - sys.modules stub trick avoids torchvision chain import error
  - Loads MNIST test set from /data/datasets/mnist/mnist_features.pt cache
  - Uses sw_forward (pure-Python LUT lookup) for RTL-equivalent golden model
  - Uses the actual trained thermometer thresholds from the checkpoint
  - UVM-style testbench: Sequencer / Driver / Monitor / Scoreboard / Coverage
  - Verifies emitted RTL output matches sw_forward bit-for-bit for every sample
  - Reports per-class accuracy comparison (SW model vs RTL)

Usage:
    cd src/mase_components/dwn_layers/test
    conda run -n plena2 python -m pytest test_dwn_mnist_fullsim.py -v -s

Environment overrides:
    DWN_CKPT            : path to checkpoint (default: mase_output/dwn/mnist_n2.pt)
    DWN_RTL_DIR         : path to emitted RTL dir
                          (default: mase_output/dwn/mnist_n2_rtl/hardware/rtl)
    DWN_UVM_NUM_SAMPLES : number of MNIST test samples to simulate (default: 500)
    DWN_MNIST_CACHE     : path to MNIST cache .pt file
                          (default: /data/datasets/mnist/mnist_features.pt)
"""

import json
import os
import sys
import types

# ---------------------------------------------------------------------------
# sys.path + sys.modules stub (avoids torchvision chain import error)
# ---------------------------------------------------------------------------

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

_CONDA_ENV_BIN = os.path.join(os.environ.get("CONDA_PREFIX", ""), "bin")
if os.path.isdir(_CONDA_ENV_BIN) and _CONDA_ENV_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _CONDA_ENV_BIN + os.pathsep + os.environ.get("PATH", "")

import torch
import cocotb_test.simulator as simulator

# RTL component directory (fixed_dwn_lut_neuron.sv, fixed_dwn_lut_layer.sv)
RTL_COMPONENT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../rtl")
)

# Default checkpoint and RTL directories for MNIST n2
_DEFAULT_CKPT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../mase_output/dwn/mnist_n2.pt")
)
_DEFAULT_RTL_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../../mase_output/dwn/mnist_n2_rtl/hardware/rtl",
    )
)
_DEFAULT_MNIST_CACHE = "/data/datasets/mnist/mnist_features.pt"

# Number of MNIST test samples to simulate
NUM_TEST_SAMPLES = int(os.environ.get("DWN_UVM_NUM_SAMPLES", "500"))


# ---------------------------------------------------------------------------
# Pure-Python SW forward (matches RTL exactly — no CUDA/EFD required)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Golden reference builder
# ---------------------------------------------------------------------------

def build_golden_reference(ckpt_path: str):
    """
    Load trained DWNModel from checkpoint.

    Returns:
        model      : full DWNModel (thermometer + lut_layers + group_sum)
        hw_forward : callable(x_flat [1,784]) →
                       (thermo_packed int, hw_output_packed int, sw_pred int)
        cfg        : model config dict from checkpoint
    """
    from chop.nn.dwn.model import DWNModel

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = ckpt["model_config"]
    print(f"[golden] Checkpoint config: {cfg}")

    model_kwargs = {k: v for k, v in cfg.items()
                    if k not in ("area_lambda", "lambda_reg")}
    model = DWNModel(**model_kwargs)
    # fit_thermometer must be called before load_state_dict to register the buffer
    model.fit_thermometer(torch.zeros(2, cfg["input_features"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    lut_layers = list(model.lut_layers)

    def hw_forward(x_flat: torch.Tensor):
        """
        Run x_flat through the golden model (thermometer + sw_forward + groupsum).

        Uses sw_forward (pure-Python LUT lookup) for the hardware core so that
        the result exactly matches the RTL — no CUDA/EFD dependency.

        Args:
            x_flat: float tensor [1, input_features] (MNIST pixels in [0,1])

        Returns:
            thermo_packed    (int): packed thermometer bits → RTL data_in_0
            hw_output_packed (int): expected RTL output    → expected data_out_0
            sw_pred          (int): torch model prediction (for reference)
        """
        with torch.no_grad():
            # Step 1: thermometer encoding (CPU, no EFD needed)
            thermo = model.thermometer(x_flat)       # [1, thermo_width], float {0,1}
            thermo_bits = thermo[0].int().tolist()   # list of 0/1
            thermo_packed = sum(b << i for i, b in enumerate(thermo_bits))

            # Step 2: LUT layers — use sw_forward (mirrors RTL exactly)
            hw_bits = sw_forward(thermo_bits, lut_layers)
            hw_output_packed = sum(b << i for i, b in enumerate(hw_bits))

            # Step 3: GroupSum → predicted class (reference only)
            h = torch.tensor(hw_bits, dtype=torch.float32).unsqueeze(0)
            logits  = model.group_sum(h)             # [1, num_classes]
            sw_pred = int(logits.argmax(dim=1).item())

        return thermo_packed, hw_output_packed, sw_pred

    return model, hw_forward, cfg


# ---------------------------------------------------------------------------
# MNIST data loader (from pre-cached .pt file, no torchvision required)
# ---------------------------------------------------------------------------

def load_mnist_test(num_samples: int):
    """
    Load MNIST test set from cache (last 10 000 of 70 000 standard split).

    Returns:
        list of (img_flat, label) where img_flat is float tensor [1, 784]
    """
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


# ---------------------------------------------------------------------------
# Pytest test
# ---------------------------------------------------------------------------

def test_dwn_mnist_fullsim():
    """
    Full simulation of emitted MNIST n2 dwn_top RTL vs torch DWN golden reference.

    Test flow:
      1. Load trained DWNModel from mnist_n2.pt checkpoint
      2. Load real MNIST test images from /data/datasets/mnist/mnist_features.pt
      3. For each image run the golden model to produce:
           - thermometer-encoded input  (packed int → RTL data_in_0)
           - expected hardware output   (packed int from sw_forward → expected data_out_0)
           - predicted class            (GroupSum of sw_forward output, for reference)
      4. Write UVM config JSON with all transactions
      5. Run UVM-style cocotb simulation (Verilator backend)
      6. Assert 100% RTL vs SW match (any mismatch = RTL correctness bug)
    """
    ckpt_path = os.environ.get("DWN_CKPT",    _DEFAULT_CKPT)
    rtl_dir   = os.environ.get("DWN_RTL_DIR", _DEFAULT_RTL_DIR)

    assert os.path.exists(ckpt_path), (
        f"Checkpoint not found: {ckpt_path}\n"
        "Train with:\n"
        "  conda run -n plena2 python test/passes/graph/transforms/dwn/"
        "run_dwn_training.py --dataset mnist --hidden-sizes 2000 1000 --lut-n 2"
    )
    assert os.path.exists(rtl_dir), (
        f"RTL directory not found: {rtl_dir}\n"
        "Emit RTL with:\n"
        "  conda run -n plena2 python scripts/emit_dwn_rtl.py --ckpt-name mnist_n2"
    )

    dwn_top_sv = os.path.join(rtl_dir, "dwn_top.sv")
    assert os.path.exists(dwn_top_sv), f"dwn_top.sv not found in: {rtl_dir}"

    # ---- Build golden reference ----
    print(f"\n[test] Loading golden model from: {ckpt_path}")
    model, hw_forward, cfg = build_golden_reference(ckpt_path)
    thermo_width = cfg["input_features"] * cfg["num_bits"]
    num_outputs  = cfg["hidden_sizes"][-1]
    num_classes  = cfg["num_classes"]

    print(
        f"[test] Architecture: "
        f"{thermo_width} thermo bits → "
        f"hidden={cfg['hidden_sizes']} → "
        f"{num_outputs} output bits  (LUT_N={cfg['lut_n']})"
    )

    # ---- Load real MNIST test data ----
    print(f"[test] Loading {NUM_TEST_SAMPLES} MNIST test samples...")
    samples = load_mnist_test(NUM_TEST_SAMPLES)
    print(f"[test] Loaded {len(samples)} samples")

    # ---- Generate transactions via golden model ----
    print(f"[test] Running golden model (sw_forward) on {len(samples)} samples...")
    transactions = []
    sw_correct   = 0
    for i, (img_flat, label) in enumerate(samples):
        thermo_packed, hw_output_packed, sw_pred = hw_forward(img_flat)
        if sw_pred == label:
            sw_correct += 1
        transactions.append({
            "thermo_packed":    thermo_packed,
            "hw_output_packed": hw_output_packed,
            "label":            label,
            "sw_pred":          sw_pred,
        })
        if (i + 1) % 100 == 0:
            print(f"  ... processed {i + 1}/{len(samples)} samples")

    sw_acc = 100.0 * sw_correct / len(samples)
    print(f"[test] sw_forward accuracy on {len(samples)} samples: {sw_acc:.1f}%")
    print(f"[test] (Expected ~97.4% for mnist_n2 checkpoint at epoch 29)")

    # ---- Write UVM config JSON ----
    config_path = os.path.join(os.path.dirname(__file__), "dwn_mnist_uvm_config.json")
    config = {
        "num_inputs":   thermo_width,
        "num_outputs":  num_outputs,
        "num_classes":  num_classes,
        "transactions": transactions,
    }
    with open(config_path, "w") as f:
        json.dump(config, f)
    os.environ["DWN_MNIST_UVM_CONFIG"] = config_path
    print(f"[test] UVM config written to: {config_path}")

    # ---- Run cocotb UVM simulation ----
    print(f"[test] Starting cocotb UVM simulation (Verilator)...")
    # make_args fixes a Verilator/GCC PCH bug: CFG_CXXFLAGS_PCH_I is empty in
    # the conda-packaged Verilator, causing GCC to treat the .fast PCH hint as
    # a linker input instead of a precompiled-header prefix.
    simulator.run(
        verilog_sources=[
            os.path.join(RTL_COMPONENT_DIR, "fixed_dwn_lut_neuron.sv"),
            os.path.join(RTL_COMPONENT_DIR, "fixed_dwn_lut_layer.sv"),
            dwn_top_sv,
        ],
        toplevel="dwn_top",
        module="dwn_mnist_uvm_tb",
        simulator="verilator",
        waves=False,
        build_dir=os.path.join(os.path.dirname(__file__), "sim_build_dwn_mnist_uvm"),
        python_search_path=[os.path.dirname(__file__)],
        extra_args=["--Wno-TIMESCALEMOD"],
        make_args=["CFG_CXXFLAGS_PCH_I=-include"],
    )
