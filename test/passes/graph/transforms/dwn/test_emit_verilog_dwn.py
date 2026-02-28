"""Integration test: emit_verilog end-to-end with DWN LUTLayer."""
import os
import re
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


@requires_cuda
def test_emit_verilog_dwn_lut_layer():
    """
    End-to-end: build a tiny LUTLayer model, run the metadata + emit_verilog
    passes, and verify the generated Verilog is syntactically correct.

    Checks:
    - Parameters use lowercase node-name prefix (e.g. lut_INPUT_SIZE, not LUT_INPUT_SIZE)
    - INPUT_INDICES and LUT_CONTENTS are raw Verilog hex literals, NOT quoted strings
    - Expected parameters (INPUT_SIZE, OUTPUT_SIZE, LUT_N) are present
    """
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
    from mase_components.dwn_layers.passes import dwn_hardware_metadata_pass

    device = torch.device("cuda")

    # ------------------------------------------------------------------ model
    INPUT_SIZE = 8
    OUTPUT_SIZE = 4
    LUT_N = 2

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

    model = TinyDWN().to(device)
    model.eval()

    dummy_input = torch.randint(0, 2, (1, INPUT_SIZE)).float().to(device)

    # ------------------------------------------------------------------ graph
    # custom_ops "modules" keys must be class objects (not strings)
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

    # ------------------------------------------------------------------ metadata
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

    # ------------------------------------------------------------------ emit
    with tempfile.TemporaryDirectory() as tmp_dir:
        graph, _ = emit_verilog_top_transform_pass(
            graph,
            pass_args={
                "project_dir": tmp_dir,
                "top_name": "dwn_top",
            },
        )

        sv_path = os.path.join(tmp_dir, "hardware", "rtl", "dwn_top.sv")
        assert os.path.exists(sv_path), f"Verilog file not generated at {sv_path}"

        sv = open(sv_path).read()

    # ------------------------------------------------------------------ checks
    # Bug 1 fix: parameter references must use lowercase node prefix
    assert "lut_INPUT_SIZE" in sv, "Expected 'lut_INPUT_SIZE' parameter in Verilog"
    assert "lut_OUTPUT_SIZE" in sv, "Expected 'lut_OUTPUT_SIZE' parameter in Verilog"
    assert "lut_LUT_N" in sv, "Expected 'lut_LUT_N' parameter in Verilog"
    # Must NOT have uppercase LUT_ prefix for these parameters
    assert "LUT_INPUT_SIZE" not in sv, (
        "Found uppercase 'LUT_INPUT_SIZE' — case mismatch bug still present"
    )
    assert "LUT_OUTPUT_SIZE" not in sv, (
        "Found uppercase 'LUT_OUTPUT_SIZE' — case mismatch bug still present"
    )

    # Bug 2 fix: hex literals must NOT be quoted strings
    assert '"' not in sv or all(
        '"' not in line
        for line in sv.splitlines()
        if "INPUT_INDICES" in line or "LUT_CONTENTS" in line
    ), "INPUT_INDICES or LUT_CONTENTS is wrapped in quotes (should be bare hex literal)"

    # Hex literals should be present in bit-vector form
    assert re.search(r"\d+'h[0-9a-fA-F]+", sv), (
        "Expected at least one Verilog bit-vector hex literal in output"
    )

    # Parameter declarations should exist
    assert "parameter lut_INPUT_SIZE" in sv
    assert "parameter lut_OUTPUT_SIZE" in sv
    assert "parameter lut_LUT_N" in sv
    assert "parameter lut_INPUT_INDICES" in sv
    assert "parameter lut_LUT_CONTENTS" in sv

    print(f"\n[PASS] Generated Verilog excerpt (parameters):")
    for line in sv.splitlines():
        if "parameter" in line:
            print(" ", line.strip())
