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
    input_indices = 0x03020100
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
