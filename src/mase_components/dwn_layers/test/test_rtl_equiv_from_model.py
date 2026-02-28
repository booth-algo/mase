"""
Pytest runner for DWN RTL functional equivalence test using actual trained model weights.

Creates a LUTLayer with a fixed seed, extracts packed hardware parameters, writes
them to a JSON config file, then runs the cocotb testbench (dwn_lut_layer_equiv_tb)
against the RTL via Verilator.
"""
import json
import os
import sys

import cocotb_test.simulator as simulator
import pytest
import torch

# ---------------------------------------------------------------------------
# sys.path setup — mirrors test_rtl_sim.py so chop packages are importable
# ---------------------------------------------------------------------------

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Ensure verilator is discoverable by shutil.which (cocotb_test uses it internally).
_CONDA_ENV_BIN = os.path.join(os.environ.get("CONDA_PREFIX", ""), "bin")
if os.path.isdir(_CONDA_ENV_BIN) and _CONDA_ENV_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _CONDA_ENV_BIN + os.pathsep + os.environ.get("PATH", "")

# ---------------------------------------------------------------------------
# RTL sources — same as test_rtl_sim.py
# ---------------------------------------------------------------------------

RTL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../rtl"))
VERILOG_SOURCES = [
    os.path.join(RTL_DIR, f) for f in [
        "fixed_dwn_lut_layer.sv",
        "fixed_dwn_lut_neuron.sv",
        "fixed_dwn_thermometer.sv",
        "fixed_dwn_groupsum.sv",
        "fixed_dwn_flatten.sv",
    ]
]


# ---------------------------------------------------------------------------
# Parameter packing helper
# ---------------------------------------------------------------------------

def _pack_lut_params(lut_layer):
    indices = lut_layer.get_input_indices()  # (output_size, n)
    output_size = lut_layer.output_size
    lut_n = lut_layer.n
    packed_indices = 0
    for i in range(output_size):
        for k in range(lut_n):
            idx = int(indices[i, k].item()) & 0xFF
            packed_indices |= idx << ((i * lut_n + k) * 8)
    contents = lut_layer.get_lut_contents()  # (output_size, 2^n)
    lut_entries = 2 ** lut_n
    packed_contents = 0
    for i in range(output_size):
        for j in range(lut_entries):
            bit = int(contents[i, j].item()) & 1
            packed_contents |= bit << (i * lut_entries + j)
    return packed_indices, packed_contents


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_rtl_lut_layer_from_trained_model():
    """RTL/SW equivalence using LUTLayer weights initialised with torch.manual_seed(42).

    Uses input_size=4, output_size=2, n=2 so INPUT_INDICES fits in 32 bits
    (2 outputs × 2 inputs × 8 bits = 32 bits) — Verilator's -G flag cannot
    reliably pass integers wider than 32 bits as decimals.
    """
    from chop.nn.dwn import LUTLayer  # noqa: import inside test to allow sys.path setup

    torch.manual_seed(42)
    lut_layer = LUTLayer(input_size=4, output_size=2, n=2, mapping="random")
    lut_layer.eval()

    packed_indices, packed_contents = _pack_lut_params(lut_layer)

    config = {
        "input_size": lut_layer.input_size,
        "output_size": lut_layer.output_size,
        "lut_n": lut_layer.n,
        "input_indices_packed": packed_indices,
        "lut_contents_packed": packed_contents,
    }

    config_path = os.path.join(os.path.dirname(__file__), "_equiv_test_config.json")
    try:
        with open(config_path, "w") as f:
            json.dump(config, f)

        os.environ["DWN_EQUIV_CONFIG"] = config_path

        simulator.run(
            verilog_sources=VERILOG_SOURCES,
            toplevel="fixed_dwn_lut_layer",
            module="dwn_lut_layer_equiv_tb",
            simulator="verilator",
            waves=False,
            parameters={
                "INPUT_SIZE": lut_layer.input_size,
                "OUTPUT_SIZE": lut_layer.output_size,
                "LUT_N": lut_layer.n,
                "INPUT_INDICES": packed_indices,
                "LUT_CONTENTS": packed_contents,
            },
        )
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)
