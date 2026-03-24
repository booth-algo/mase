"""
Pytest runner for full DWN network RTL equivalence test.

Emits a tiny DWNHardwareCore to dwn_top.sv, then simulates it against
a pure-Python SW LUT model to verify end-to-end correctness.
"""
import json
import os
import random
import tempfile

from dwn_test_utils import setup_sys_path, setup_conda_path, sw_forward

setup_sys_path()
setup_conda_path()

import cocotb_test.simulator as simulator
import torch
import torch.nn as nn

RTL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../rtl"))


# Tiny DWNHardwareCore (LUT-layers only, no thermometer / groupsum)

INPUT_SIZE = 16
HIDDEN_SIZE = 8
OUTPUT_SIZE = 4
LUT_N = 2


class DWNHardwareCore(nn.Module):
    def __init__(self, lut_layers):
        super().__init__()
        self.lut_layers = nn.ModuleList(lut_layers)

    def forward(self, x):
        for layer in self.lut_layers:
            x = layer(x)
        return x


# Test

def test_rtl_dwn_top_equiv():
    """Emit tiny dwn_top RTL and verify against SW LUT golden model."""
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

    # ----- build tiny model -----
    torch.manual_seed(42)
    lut_layers = [
        LUTLayer(input_size=INPUT_SIZE, output_size=HIDDEN_SIZE, n=LUT_N, mapping="random"),
        LUTLayer(input_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, n=LUT_N, mapping="random"),
    ]
    hw_model = DWNHardwareCore(lut_layers)
    hw_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hw_model = hw_model.to(device)
    dummy_input = torch.zeros(1, INPUT_SIZE, device=device)

    # ----- emit RTL -----
    with tempfile.TemporaryDirectory() as tmpdir:
        graph = MaseGraph(
            hw_model,
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
        graph, _ = init_metadata_analysis_pass(graph)
        graph, _ = add_common_metadata_analysis_pass(
            graph, pass_args={"dummy_in": {"x": dummy_input}, "add_value": False}
        )
        graph, _ = add_hardware_metadata_analysis_pass(graph)
        graph, _ = dwn_hardware_metadata_pass(graph)
        graph, _ = emit_verilog_top_transform_pass(
            graph, pass_args={"project_dir": tmpdir, "top_name": "dwn_top"}
        )
        graph, _ = emit_internal_rtl_transform_pass(
            graph, pass_args={"project_dir": tmpdir}
        )

        emitted_rtl_dir = os.path.join(tmpdir, "hardware", "rtl")
        dwn_top_sv = os.path.join(emitted_rtl_dir, "dwn_top.sv")
        assert os.path.exists(dwn_top_sv), f"Expected emitted file: {dwn_top_sv}"

        # ----- generate SW test vectors -----
        rng = random.Random(42)
        test_vectors = []
        max_in = (1 << INPUT_SIZE) - 1
        for _ in range(256):
            val = rng.randint(0, max_in)
            bits = [(val >> i) & 1 for i in range(INPUT_SIZE)]
            out_bits = sw_forward(bits, lut_layers)
            out_packed = sum(b << i for i, b in enumerate(out_bits))
            test_vectors.append({"input_packed": val, "output_packed": out_packed})

        # ----- write config JSON -----
        config_path = os.path.join(tmpdir, "dwn_top_equiv_config.json")
        config = {
            "num_inputs": INPUT_SIZE,
            "num_outputs": OUTPUT_SIZE,
            "test_vectors": test_vectors,
        }
        with open(config_path, "w") as f:
            json.dump(config, f)
        old_val = os.environ.get("DWN_TOP_EQUIV_CONFIG")
        try:
            os.environ["DWN_TOP_EQUIV_CONFIG"] = config_path

            # ----- run cocotb simulation -----
            simulator.run(
                verilog_sources=[
                    os.path.join(RTL_DIR, "fixed", "fixed_dwn_lut_neuron.sv"),
                    os.path.join(RTL_DIR, "fixed", "fixed_dwn_lut_layer.sv"),
                    dwn_top_sv,
                ],
                toplevel="dwn_top",
                module="dwn_top_equiv_tb",
                simulator="verilator",
                waves=False,
                build_dir=os.path.join(os.path.dirname(__file__), "sim_build_dwn_top"),
                python_search_path=[os.path.dirname(__file__)],
                extra_args=["--Wno-TIMESCALEMOD"],
            )
        finally:
            if old_val is None:
                os.environ.pop("DWN_TOP_EQUIV_CONFIG", None)
            else:
                os.environ["DWN_TOP_EQUIV_CONFIG"] = old_val
