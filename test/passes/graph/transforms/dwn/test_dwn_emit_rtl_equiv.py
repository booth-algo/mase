"""End-to-end functional equivalence test: emitted dwn_top.sv vs Python LUTLayer.

Builds a DWN model from parametrized config, runs the full emit pipeline, then
drives the generated Verilog through Verilator via cocotb and checks that every
output matches the Python reference model (sw_forward).

Parametrized over single-layer and multi-layer configurations.
"""
import json
import os
import random
import shutil
import sys
import tempfile

import pytest
import torch
import torch.nn as nn

_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../src"))
sys.path.insert(0, _src)

# Also add the integration test directory so we can reuse dwn_test_utils
_integration_dir = os.path.abspath(
    os.path.join(
        _src,
        "mase_components/dwn_layers/test/integration",
    )
)
if _integration_dir not in sys.path:
    sys.path.insert(0, _integration_dir)

from dwn_test_utils import setup_sys_path, setup_conda_path, sw_forward

setup_sys_path()
setup_conda_path()

CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="EFDFunction requires CUDA"
)


# Cocotb testbench source (written to tempdir at test-time)

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


class DWNHardwareCore(nn.Module):
    """Wrapper that chains one or more LUTLayers for the emit pipeline."""

    def __init__(self, lut_layers):
        super().__init__()
        self.lut_layers = nn.ModuleList(lut_layers)

    def forward(self, x):
        for layer in self.lut_layers:
            x = layer(x)
        return x


@requires_cuda
@pytest.mark.parametrize(
    "model_config",
    [
        {
            "name": "single_layer",
            "input_size": 8,
            "hidden_sizes": [4],
            "lut_n": 2,
            "seed": 42,
        },
        {
            "name": "multi_layer",
            "input_size": 16,
            "hidden_sizes": [8, 4],
            "lut_n": 2,
            "seed": 42,
        },
    ],
    ids=["1-layer-8in", "2-layer-16in"],
)
def test_emitted_top_functional_equiv(model_config):
    """Functional equivalence: emitted dwn_top.sv output must match Python sw_forward."""

    # ------------------------------------------------------------------
    # Skip if Verilator not available
    # ------------------------------------------------------------------
    if shutil.which("verilator") is None:
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

    input_size = model_config["input_size"]
    hidden_sizes = model_config["hidden_sizes"]
    lut_n = model_config["lut_n"]
    seed = model_config["seed"]

    device = torch.device("cuda")

    # ------------------------------------------------------------------
    # Build model: chain of LUTLayers from input_size through hidden_sizes
    # ------------------------------------------------------------------
    torch.manual_seed(seed)
    sizes = [input_size] + hidden_sizes
    lut_layers = [
        LUTLayer(
            input_size=sizes[i],
            output_size=sizes[i + 1],
            n=lut_n,
            mapping="random",
        )
        for i in range(len(sizes) - 1)
    ]
    model = DWNHardwareCore(lut_layers)
    model.eval()
    model = model.to(device)

    output_size = hidden_sizes[-1]
    dummy_input = torch.randint(0, 2, (1, input_size)).float().to(device)

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
                        "dwn_layers/rtl/fixed/fixed_dwn_lut_neuron.sv",
                        "dwn_layers/rtl/fixed/fixed_dwn_lut_layer.sv",
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
        # Generate SW test vectors using sw_forward (pure-Python LUT cascade)
        # ------------------------------------------------------------------
        # Move layers to CPU for sw_forward
        cpu_layers = [layer.cpu() for layer in lut_layers]

        rng = random.Random(seed)
        max_in = (1 << input_size) - 1
        vectors = []
        for _ in range(256):
            val = rng.randint(0, max_in)
            bits = [(val >> i) & 1 for i in range(input_size)]
            out_bits = sw_forward(bits, cpu_layers)
            out_packed = sum(b << i for i, b in enumerate(out_bits))
            vectors.append({"input": val, "expected": out_packed})

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
            python_search=[tmp_dir],
        )
