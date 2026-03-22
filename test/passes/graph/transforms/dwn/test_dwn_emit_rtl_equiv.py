"""End-to-end functional equivalence test: emitted dwn_top.sv vs Python LUTLayer.

Builds a tiny TinyDWN model, runs the full emit pipeline, then drives the
generated Verilog through Verilator via cocotb and checks that every output
matches the Python reference model.
"""
import json
import os
import shutil
import sys
import tempfile

import pytest
import torch
import torch.nn as nn

_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../src"))
sys.path.insert(0, _src)

CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="EFDFunction requires CUDA"
)

INPUT_SIZE = 8
OUTPUT_SIZE = 4
LUT_N = 2


# ---------------------------------------------------------------------------
# Cocotb testbench source (written to tempdir at test-time)
# ---------------------------------------------------------------------------

_TB_SOURCE = '''\
"""Cocotb testbench for dwn_top equivalence check.

Reads TEST_VECTORS_PATH env var -> JSON list of {input: int, expected: int}.
Drives data_in_0 for each vector and asserts data_out_0 matches expected.
"""
import json
import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge


async def _reset(dut):
    await cocotb.start(Clock(dut.clk, 10, units="ns").start())
    dut.rst.value = 1
    dut.data_in_0_valid.value = 0
    dut.data_out_0_ready.value = 1
    await ClockCycles(dut.clk, 4)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_equiv(dut):
    vectors_path = os.environ["TEST_VECTORS_PATH"]
    with open(vectors_path) as f:
        vectors = json.load(f)

    await _reset(dut)

    passed = 0
    failed = 0
    for v in vectors:
        inp = v["input"]
        expected = v["expected"]

        dut.data_in_0.value = inp
        dut.data_in_0_valid.value = 1
        dut.data_out_0_ready.value = 1
        await RisingEdge(dut.clk)
        await ClockCycles(dut.clk, 1)   # combinational output is stable
        dut.data_in_0_valid.value = 0

        got = int(dut.data_out_0.value)
        if got == expected:
            passed += 1
        else:
            failed += 1
            assert False, (
                f"MISMATCH: input={inp:#010b} expected={expected:#06b} got={got:#06b}"
            )

    cocotb.log.info(f"[EQUIV] {passed} passed, {failed} failed")
'''


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@requires_cuda
def test_emitted_top_functional_equiv():
    """Functional equivalence: emitted dwn_top.sv output must match Python LUTLayer."""

    # ------------------------------------------------------------------
    # Skip if Verilator not available
    # ------------------------------------------------------------------
    if shutil.which("verilator") is None:
        # Also check conda env bin
        _conda_bin = os.path.join(os.environ.get("CONDA_PREFIX", ""), "bin")
        if not os.path.isfile(os.path.join(_conda_bin, "verilator")):
            pytest.skip("verilator not found on PATH")
        os.environ["PATH"] = _conda_bin + os.pathsep + os.environ.get("PATH", "")

    import cocotb_test.simulator as sim_runner

    from chop.nn.dwn import LUTLayer
    from chop.ir.graph.mase_graph import MaseGraph
    from chop.passes.graph.analysis.init_metadata import init_metadata_analysis_pass
    from chop.passes.graph.analysis.add_metadata.add_common_metadata import (
        add_common_metadata_analysis_pass,
    )
    from chop.passes.graph.analysis.add_metadata.add_hardware_metadata import (
        add_hardware_metadata_analysis_pass,
    )
    from chop.passes.graph.transforms.verilog.emit_top import (
        emit_verilog_top_transform_pass,
    )
    from chop.passes.graph.transforms.verilog.emit_internal import (
        emit_internal_rtl_transform_pass,
    )
    from mase_components.dwn_layers.passes import dwn_hardware_metadata_pass

    device = torch.device("cuda")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    class TinyDWN(nn.Module):
        def __init__(self):
            super().__init__()
            self.lut = LUTLayer(
                input_size=INPUT_SIZE,
                output_size=OUTPUT_SIZE,
                n=LUT_N,
                mapping="random",
            )

        def forward(self, x):
            return self.lut(x)

    torch.manual_seed(42)
    model = TinyDWN().to(device)
    model.eval()

    dummy_input = torch.randint(0, 2, (1, INPUT_SIZE)).float().to(device)

    # ------------------------------------------------------------------
    # Build MaseGraph
    # ------------------------------------------------------------------
    graph = MaseGraph(
        model,
        custom_ops={
            "modules": {
                LUTLayer: {
                    "args": {"x": "data_in"},
                    "module": "fixed_dwn_lut_layer",
                    "dependence_files": [
                        "dwn_layers/rtl/fixed_dwn_lut_neuron.sv",
                        "dwn_layers/rtl/fixed_dwn_lut_layer.sv",
                    ],
                }
            },
            "functions": {},
        },
    )

    # ------------------------------------------------------------------
    # Metadata passes
    # ------------------------------------------------------------------
    graph, _ = init_metadata_analysis_pass(graph)
    graph, _ = add_common_metadata_analysis_pass(
        graph,
        pass_args={
            "dummy_in": {"x": dummy_input},
            "add_value": False,
        },
    )
    graph, _ = add_hardware_metadata_analysis_pass(graph)
    graph, _ = dwn_hardware_metadata_pass(graph)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # ------------------------------------------------------------------
        # Emit Verilog
        # ------------------------------------------------------------------
        graph, _ = emit_verilog_top_transform_pass(
            graph,
            pass_args={
                "project_dir": tmp_dir,
                "top_name": "dwn_top",
            },
        )

        sv_top = os.path.join(tmp_dir, "hardware", "rtl", "dwn_top.sv")
        assert os.path.exists(sv_top), f"dwn_top.sv not generated at {sv_top}"

        # ------------------------------------------------------------------
        # Copy internal RTL dependencies
        # ------------------------------------------------------------------
        graph, _ = emit_internal_rtl_transform_pass(
            graph,
            pass_args={"project_dir": tmp_dir},
        )

        rtl_dir = os.path.join(tmp_dir, "hardware", "rtl")
        sv_files = sorted(
            os.path.join(rtl_dir, f)
            for f in os.listdir(rtl_dir)
            if f.endswith(".sv")
        )
        assert sv_files, f"No .sv files found in {rtl_dir}"

        # ------------------------------------------------------------------
        # Extract Python model predictions
        # ------------------------------------------------------------------
        lut_layer = model.lut
        lut_layer.eval()

        torch.manual_seed(0)
        test_inputs = torch.randint(0, 2, (256, INPUT_SIZE)).float().to(device)

        with torch.no_grad():
            py_outputs = lut_layer(test_inputs)  # (256, OUTPUT_SIZE)

        # Pack input bits into integer (bit i = input feature i)
        def _pack_input(row):
            val = 0
            for i, b in enumerate(row):
                val |= (int(b.item()) & 1) << i
            return val

        # Pack output bits into integer (bit i = output neuron i)
        def _pack_output(row):
            val = 0
            for i, b in enumerate(row):
                val |= (int(b.item()) & 1) << i
            return val

        vectors = []
        for inp_row, out_row in zip(test_inputs, py_outputs):
            vectors.append(
                {
                    "input": _pack_input(inp_row),
                    "expected": _pack_output(out_row),
                }
            )

        # ------------------------------------------------------------------
        # Write test vectors JSON and cocotb testbench
        # ------------------------------------------------------------------
        vectors_path = os.path.join(tmp_dir, "test_vectors.json")
        with open(vectors_path, "w") as f:
            json.dump(vectors, f)

        tb_path = os.path.join(tmp_dir, "_dwn_top_equiv_tb.py")
        with open(tb_path, "w") as f:
            f.write(_TB_SOURCE)

        # ------------------------------------------------------------------
        # Run Verilator simulation via cocotb
        # ------------------------------------------------------------------
        sim_runner.run(
            verilog_sources=sv_files,
            toplevel="dwn_top",
            module="_dwn_top_equiv_tb",
            simulator="verilator",
            waves=False,
            extra_env={"TEST_VECTORS_PATH": vectors_path},
            # cocotb_test searches this list for the testbench module
            python_search=[tmp_dir],
        )
