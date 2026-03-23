"""
Pytest runner for DWN paper-scope full simulation.

Loads the trained baseline_n6 DWN model checkpoint and uses it as a "golden
reference" to generate test vectors for a UVM-style cocotb simulation of
the emitted dwn_top_paper_scope.sv RTL (4-cycle clocked pipeline).

Key features:
  - sys.modules stub trick avoids torchvision chain import error
  - Loads MNIST test set from /data/datasets/mnist/mnist_features.pt cache
  - Uses sw_forward (pure-Python LUT lookup) for LUT layer golden model
  - Uses group_sum_forward (pure-Python count) for GroupSum golden model
  - Both exactly match the clocked RTL pipeline (dwn_top_clocked + groupsum_pipelined)
  - UVM-style testbench: Sequencer / Driver / Monitor / Scoreboard / Coverage
  - Verifies RTL output scores match SW golden scores element-wise for every sample
  - Reports per-class accuracy comparison (SW model vs RTL)

Usage:
    cd src/mase_components/dwn_layers/test
    conda run -n plena2 python -m pytest test_dwn_paper_scope_fullsim.py -v -s

Environment overrides:
    DWN_CKPT                    : checkpoint path (default: baseline_n6.pt)
    DWN_RTL_DIR                 : RTL dir (default: baseline_n6_rtl/hardware/rtl)
    DWN_PAPER_SCOPE_NUM_SAMPLES : number of test samples (default: 200)
    DWN_MNIST_CACHE             : MNIST cache path
"""

import json
import os
import sys

from dwn_test_utils import (
    setup_sys_path, setup_conda_path, sw_forward, group_sum_forward, load_mnist_test,
)

setup_sys_path()
setup_conda_path()

import torch
import cocotb_test.simulator as simulator

# RTL component directory (fixed_dwn_lut_neuron.sv, fixed_dwn_lut_layer_clocked.sv, etc.)
RTL_COMPONENT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../rtl")
)

# Default checkpoint and RTL directories for mnist_n2
# (baseline_n6 exceeds Verilator's ~65536-bit parameter width limit:
#  n=6 INPUT_INDICES = 2000×6×12 = 144000 bits. n=2 = 48000 bits — within limit.)
_DEFAULT_CKPT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../mase_output/dwn/mnist_n2.pt")
)
_DEFAULT_RTL_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../../mase_output/dwn/mnist_n2_rtl/hardware/rtl",
    )
)
# Number of MNIST test samples to simulate
NUM_TEST_SAMPLES = int(os.environ.get("DWN_PAPER_SCOPE_NUM_SAMPLES", "200"))


# Golden reference builder

def build_golden_reference(ckpt_path: str):
    """
    Load trained DWNModel from checkpoint.

    Returns:
        model      : full DWNModel (thermometer + lut_layers + group_sum)
        hw_forward : callable(x_flat [1, input_features]) →
                       (thermo_packed int, expected_scores list[int], sw_pred int)
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
    num_classes = cfg["num_classes"]

    def hw_forward_paper_scope(x_flat: torch.Tensor):
        """
        Run x_flat through the paper-scope golden model.

        Uses sw_forward (pure-Python LUT lookup) for LUT layers and
        group_sum_forward (pure-Python count) for GroupSum so that the
        result exactly matches the clocked RTL pipeline.

        Args:
            x_flat: float tensor [1, input_features]

        Returns:
            thermo_packed    (int)      : packed thermometer bits → RTL data_in_0
            expected_scores  (list[int]): raw group counts per class → expected scores
            sw_pred          (int)      : predicted class from scores (for reference)
        """
        with torch.no_grad():
            # Step 1: thermometer encoding (CPU, no EFD needed)
            thermo = model.thermometer(x_flat)       # [1, thermo_width], float {0,1}
            thermo_bits = thermo[0].int().tolist()   # list of 0/1
            thermo_packed = sum(b << i for i, b in enumerate(thermo_bits))

            # Step 2: LUT layers — use sw_forward (mirrors RTL exactly)
            lut_bits = sw_forward(thermo_bits, lut_layers)

            # Step 3: GroupSum — raw count per class (mirrors fixed_dwn_groupsum_pipelined)
            expected_scores = group_sum_forward(lut_bits, num_classes)

            # Step 4: predicted class from scores (for reference)
            sw_pred = int(max(range(num_classes), key=lambda c: expected_scores[c]))

        return thermo_packed, expected_scores, sw_pred

    return model, hw_forward_paper_scope, cfg


# Pytest test

def test_dwn_paper_scope_fullsim():
    """
    Full simulation of emitted baseline_n6 dwn_top_paper_scope RTL vs SW golden reference.

    Test flow:
      1. Load trained DWNModel from baseline_n6.pt checkpoint
      2. Load real MNIST test images from /data/datasets/mnist/mnist_features.pt
      3. For each image run the golden model to produce:
           - thermometer-encoded input  (packed int → RTL data_in_0)
           - expected class scores      (list[int] from group_sum_forward → expected RTL output)
           - predicted class            (argmax of expected_scores, for reference)
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

    dwn_top_clocked_sv     = os.path.join(rtl_dir, "dwn_top_clocked.sv")
    dwn_top_paper_scope_sv = os.path.join(rtl_dir, "dwn_top_paper_scope.sv")
    assert os.path.exists(dwn_top_clocked_sv), \
        f"dwn_top_clocked.sv not found in: {rtl_dir}"
    assert os.path.exists(dwn_top_paper_scope_sv), \
        f"dwn_top_paper_scope.sv not found in: {rtl_dir}"

    # ---- Build golden reference ----
    print(f"\n[test] Loading golden model from: {ckpt_path}")
    model, hw_forward, cfg = build_golden_reference(ckpt_path)
    thermo_width = cfg["input_features"] * cfg["num_bits"]
    num_classes  = cfg["num_classes"]

    print(
        f"[test] Architecture: "
        f"{thermo_width} thermo bits → "
        f"hidden={cfg['hidden_sizes']} → "
        f"{num_classes} class scores  (LUT_N={cfg['lut_n']})"
    )

    # ---- Load real MNIST test data ----
    print(f"[test] Loading {NUM_TEST_SAMPLES} MNIST test samples...")
    samples = load_mnist_test(NUM_TEST_SAMPLES)
    print(f"[test] Loaded {len(samples)} samples")

    # ---- Generate transactions via golden model ----
    print(f"[test] Running golden model (sw_forward + group_sum_forward) on {len(samples)} samples...")
    transactions = []
    sw_correct   = 0
    for i, (img_flat, label) in enumerate(samples):
        thermo_packed, expected_scores, sw_pred = hw_forward(img_flat)
        if sw_pred == label:
            sw_correct += 1
        transactions.append({
            "thermo_packed":    thermo_packed,
            "expected_scores":  expected_scores,
            "label":            label,
            "sw_pred":          sw_pred,
        })
        if (i + 1) % 100 == 0:
            print(f"  ... processed {i + 1}/{len(samples)} samples")

    sw_acc = 100.0 * sw_correct / len(samples)
    print(f"[test] sw_forward accuracy on {len(samples)} samples: {sw_acc:.1f}%")
    print(f"[test] (Expected ~97-98% for mnist_n2 checkpoint)")

    # ---- Write UVM config JSON ----
    config_path = os.path.join(os.path.dirname(__file__), "dwn_paper_scope_uvm_config.json")
    config = {
        "num_inputs":   thermo_width,
        "num_classes":  num_classes,
        "transactions": transactions,
    }
    with open(config_path, "w") as f:
        json.dump(config, f)
    os.environ["DWN_PAPER_SCOPE_UVM_CONFIG"] = config_path
    print(f"[test] UVM config written to: {config_path}")

    # ---- Run cocotb UVM simulation ----
    print(f"[test] Starting cocotb UVM simulation (Verilator)...")
    # make_args fixes a Verilator/GCC PCH bug: CFG_CXXFLAGS_PCH_I is empty in
    # the conda-packaged Verilator, causing GCC to treat the .fast PCH hint as
    # a linker input instead of a precompiled-header prefix.
    simulator.run(
        verilog_sources=[
            os.path.join(RTL_COMPONENT_DIR, "fixed_dwn_lut_neuron.sv"),
            os.path.join(RTL_COMPONENT_DIR, "fixed_dwn_lut_layer_clocked.sv"),
            os.path.join(RTL_COMPONENT_DIR, "fixed_dwn_groupsum_pipelined.sv"),
            os.path.join(RTL_COMPONENT_DIR, "dwn_paper_scope_sim_wrapper.sv"),
            dwn_top_clocked_sv,
            dwn_top_paper_scope_sv,
        ],
        toplevel="dwn_paper_scope_sim_wrapper",
        module="dwn_paper_scope_uvm_tb",
        simulator="verilator",
        waves=False,
        build_dir=os.path.join(os.path.dirname(__file__), "sim_build_paper_scope"),
        python_search_path=[os.path.dirname(__file__)],
        extra_args=["--Wno-TIMESCALEMOD"],
        make_args=["CFG_CXXFLAGS_PCH_I=-include"],
    )
