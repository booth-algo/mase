#!/usr/bin/env python3
"""
Emit synthesisable SystemVerilog from a trained DWN checkpoint.

Usage:
    python scripts/emit_dwn_rtl.py --ckpt-name baseline_n6 --output-dir hw/baseline_n6
    python scripts/emit_dwn_rtl.py --ckpt-name mixed_n6_2  --output-dir hw/mixed_n6_2

Output directory will contain:
    rtl/dwn_top.sv
    rtl/fixed_dwn_lut_layer.sv
    rtl/fixed_dwn_lut_neuron.sv
    (and other RTL dependencies)
"""
import argparse
import os
import sys

_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, _src)

import torch
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ckpt-name", type=str, default="best",
                        help="Checkpoint stem (looks in mase_output/dwn/<name>.pt)")
    parser.add_argument("--ckpt",      type=str, default=None,
                        help="Full checkpoint path (overrides --ckpt-name)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for emitted RTL (default: mase_output/dwn/<ckpt-name>_rtl)")
    return parser.parse_args()


def main():
    args = parse_args()

    ckpt_dir = os.path.join(os.path.dirname(__file__), "../mase_output/dwn")
    ckpt_path = args.ckpt or os.path.join(ckpt_dir, f"{args.ckpt_name}.pt")
    output_dir = args.output_dir or os.path.join(ckpt_dir, f"{args.ckpt_name}_rtl")

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["model_config"]
    print(f"  Config: hidden_sizes={cfg['hidden_sizes']}, lut_n={cfg['lut_n']}, "
          f"num_bits={cfg['num_bits']}, mapping_first={cfg['mapping_first']}")

    from chop.nn.dwn import DWNModel, LUTLayer
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Reconstruct model to get trained LUT layer weights
    model = DWNModel(**cfg).to(device)

    # fit_thermometer must be called before load_state_dict so that the
    # thermometer.thresholds buffer is registered (otherwise load_state_dict
    # rejects it as an unexpected key).  The placeholder data is immediately
    # overwritten by the checkpoint values in the load_state_dict call below.
    model.fit_thermometer(torch.zeros(2, cfg["input_features"]).to(device))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Build a hardware-only wrapper: just the LUT layers.
    # The thermometer encoding is a software preprocessing step (not hardware),
    # and GroupSum is a simple integer adder not requiring MASE RTL emission.
    # The hardware input is thermometer-encoded binary: shape (B, input_features * num_bits).
    class DWNHardwareCore(nn.Module):
        """LUT-layer stack only — the synthesisable hardware core."""
        def __init__(self, lut_layers):
            super().__init__()
            self.lut_layers = nn.ModuleList(lut_layers)

        def forward(self, x):
            for layer in self.lut_layers:
                x = layer(x)
            return x

    hw_model = DWNHardwareCore(list(model.lut_layers)).to(device)
    hw_model.eval()

    # Dummy input for the hardware core: already thermometer-encoded binary.
    thermo_width = cfg["input_features"] * cfg["num_bits"]
    dummy_input = torch.zeros(1, thermo_width).to(device)

    # Build MaseGraph from the hardware core only
    graph = MaseGraph(
        hw_model,
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

    graph, _ = init_metadata_analysis_pass(graph)
    graph, _ = add_common_metadata_analysis_pass(
        graph,
        pass_args={"dummy_in": {"x": dummy_input}, "add_value": False},
    )
    graph, _ = add_hardware_metadata_analysis_pass(graph)
    graph, _ = dwn_hardware_metadata_pass(graph)

    # Emit RTL
    graph, _ = emit_verilog_top_transform_pass(
        graph,
        pass_args={"project_dir": output_dir, "top_name": "dwn_top"},
    )
    graph, _ = emit_internal_rtl_transform_pass(
        graph,
        pass_args={"project_dir": output_dir},
    )

    rtl_dir = os.path.join(output_dir, "hardware", "rtl")
    sv_files = [f for f in os.listdir(rtl_dir) if f.endswith(".sv")]
    print(f"\nEmitted RTL to: {rtl_dir}")
    for f in sorted(sv_files):
        print(f"  {f}")
    print(f"\nSynthesis command (on Vivado server):")
    print(f"  vivado -mode batch -source scripts/synth_dwn.tcl "
          f"-tclargs {os.path.abspath(rtl_dir)} {os.path.abspath(output_dir)}/synth_results")


if __name__ == "__main__":
    main()
