"""
Pytest runner for DWN RTL cocotb testbenches.

Each test function calls simulator.run() which invokes cocotb with Verilator
and executes the @cocotb.test() coroutines in the corresponding *_tb.py file.
"""
import os
import pytest
import cocotb_test.simulator as simulator

# Ensure verilator is discoverable by shutil.which (cocotb_test uses it internally).
# CONDA_PREFIX is set automatically when a conda environment is activated.
_CONDA_ENV_BIN = os.path.join(os.environ.get("CONDA_PREFIX", ""), "bin")
if os.path.isdir(_CONDA_ENV_BIN) and _CONDA_ENV_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _CONDA_ENV_BIN + os.pathsep + os.environ.get("PATH", "")

# ---------------------------------------------------------------------------
# RTL source directories
# ---------------------------------------------------------------------------
FIXED_RTL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../rtl/fixed"))
STRUCTURAL_RTL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../rtl/structural"))

VERILOG_SOURCES = [
    os.path.join(FIXED_RTL_DIR, f) for f in [
        "fixed_dwn_lut_layer.sv",
        "fixed_dwn_lut_neuron.sv",
        "fixed_dwn_thermometer.sv",
        "fixed_dwn_groupsum.sv",
        "fixed_dwn_flatten.sv",
    ]
]

STRUCTURAL_VERILOG_SOURCES = [
    os.path.join(STRUCTURAL_RTL_DIR, f) for f in [
        "sim_lut6.sv",
        "structural_dwn_lut_neuron.sv",
        "structural_dwn_lut_layer.sv",
    ]
]

# Verilator flags to suppress width warnings on parameterised structural modules.
_STRUCTURAL_EXTRA_ARGS = [
    "--Wno-WIDTHTRUNC",
    "--Wno-WIDTHEXPAND",
]

# ---------------------------------------------------------------------------
# Fixed RTL tests
# ---------------------------------------------------------------------------

def test_rtl_groupsum():
    """Run cocotb testbench for fixed_dwn_groupsum."""
    simulator.run(
        verilog_sources=VERILOG_SOURCES,
        toplevel="fixed_dwn_groupsum",
        module="fixed_dwn_groupsum_tb",
        simulator="verilator",
        waves=False,
    )


def test_rtl_thermometer():
    """Run cocotb testbench for fixed_dwn_thermometer."""
    thresholds = (0x60 << 24) | (0x20 << 16) | (0x80 << 8) | 0x40

    simulator.run(
        verilog_sources=VERILOG_SOURCES,
        toplevel="fixed_dwn_thermometer",
        module="fixed_dwn_thermometer_tb",
        simulator="verilator",
        waves=False,
        parameters={
            "NUM_FEATURES": 2,
            "FEATURE_WIDTH": 8,
            "NUM_THRESHOLDS": 2,
            "THRESHOLDS": thresholds,
        },
    )


def test_rtl_lut_layer():
    """Run cocotb testbench for fixed_dwn_lut_layer."""
    # INPUT_INDICES packed with INDEX_BITS=2 (for INPUT_SIZE=4):
    # LUT0 reads [idx0=0, idx1=1], LUT1 reads [idx0=2, idx1=3]
    # Packed: 0b11_10_01_00 = 0xE4
    input_indices = 0xE4
    lut_contents = 0xAC

    simulator.run(
        verilog_sources=VERILOG_SOURCES,
        toplevel="fixed_dwn_lut_layer",
        module="fixed_dwn_lut_layer_tb",
        simulator="verilator",
        waves=False,
        parameters={
            "INPUT_SIZE": 4,
            "OUTPUT_SIZE": 2,
            "LUT_N": 2,
            "INPUT_INDICES": input_indices,
            "LUT_CONTENTS": lut_contents,
        },
    )


# ---------------------------------------------------------------------------
# Structural RTL tests (use behavioral LUT6 sim stub for Verilator)
# ---------------------------------------------------------------------------

def test_structural_rtl_lut_layer():
    """Run cocotb testbench against structural_dwn_lut_layer.

    The structural variant has the same data/valid/ready ports but no clk/rst
    (purely combinational), so it uses a dedicated Timer-based testbench.
    """
    # INPUT_INDICES packed with INDEX_BITS=2 (for INPUT_SIZE=4):
    # LUT0 reads [idx0=0, idx1=1], LUT1 reads [idx0=2, idx1=3]
    # Packed: 0b11_10_01_00 = 0xE4
    input_indices = 0xE4
    lut_contents = 0xAC

    simulator.run(
        verilog_sources=STRUCTURAL_VERILOG_SOURCES,
        toplevel="structural_dwn_lut_layer",
        module="structural_dwn_lut_layer_tb",
        simulator="verilator",
        waves=False,
        extra_args=_STRUCTURAL_EXTRA_ARGS,
        parameters={
            "INPUT_SIZE": 4,
            "OUTPUT_SIZE": 2,
            "LUT_N": 2,
            "INPUT_INDICES": input_indices,
            "LUT_CONTENTS": lut_contents,
        },
    )


def test_structural_rtl_lut_neuron():
    """Exhaustive test of structural_dwn_lut_neuron with LUT_N=2.

    LUT_CONTENTS=4'b1010 implements out = data_in_0[0] (identity of bit 0).
    All 2^LUT_N=4 input patterns are checked.
    """
    simulator.run(
        verilog_sources=STRUCTURAL_VERILOG_SOURCES,
        toplevel="structural_dwn_lut_neuron",
        module="structural_dwn_lut_neuron_tb",
        simulator="verilator",
        waves=False,
        extra_args=_STRUCTURAL_EXTRA_ARGS,
        parameters={
            "LUT_N": 2,
            "LUT_CONTENTS": 0b1010,
        },
    )
