#!/usr/bin/env python3
"""
Emit synthesisable SystemVerilog from a trained DiffLogic checkpoint.

Usage:
    python scripts/emit_difflogic_rtl.py --ckpt-name mnist_8k
    python scripts/emit_difflogic_rtl.py --ckpt-name cifar10_8k --output-dir hw/cifar10_8k

Output directory will contain:
    hardware/rtl/top.sv
    hardware/rtl/fixed_difflogic_logic.sv
    hardware/rtl/fixed_difflogic_logic_neuron.sv
    hardware/rtl/fixed_difflogic_groupsum.sv
    hardware/rtl/fixed_difflogic_flatten.sv
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
                        help="Checkpoint stem (looks in mase_output/difflogic/<name>.pt)")
    parser.add_argument("--ckpt",      type=str, default=None,
                        help="Full checkpoint path (overrides --ckpt-name)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for emitted RTL (default: mase_output/difflogic/<ckpt-name>_rtl)")
    return parser.parse_args()


def main():
    args = parse_args()

    ckpt_dir = os.path.join(os.path.dirname(__file__), "../mase_output/difflogic")
    ckpt_path = args.ckpt or os.path.join(ckpt_dir, f"{args.ckpt_name}.pt")
    output_dir = args.output_dir or os.path.join(ckpt_dir, f"{args.ckpt_name}_rtl")

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = ckpt["model_config"]
    print(f"  Config: num_neurons={cfg['num_neurons']}, num_layers={cfg['num_layers']}, "
          f"tau={cfg['tau']}, connections={cfg['connections']}")

    try:
        from difflogic import LogicLayer, GroupSum
    except ImportError:
        print("ERROR: 'difflogic' package not found.")
        print("Install with:  pip install difflogic")
        sys.exit(1)

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
    from mase_components.difflogic_layers.passes import (
        difflogic_hardware_metadata_optimize_pass,
        difflogic_hardware_force_fixed_flatten_pass,
    )

    device = torch.device("cpu")
    print(f"Device: {device}")

    # Reconstruct model from config
    input_features = cfg["input_features"]
    num_classes = cfg["num_classes"]
    num_neurons = cfg["num_neurons"]
    num_layers = cfg["num_layers"]
    grad_factor = cfg.get("grad_factor", 1.0)
    tau = cfg["tau"]
    connections = cfg.get("connections", "random")

    # Build identical model structure (must use 'cpu' and 'python' impl for RTL emit)
    layers = [nn.Flatten()]
    layers.append(LogicLayer(
        input_features, num_neurons,
        device='cpu', grad_factor=grad_factor,
        implementation='python', connections=connections,
    ))
    for _ in range(num_layers - 1):
        layers.append(LogicLayer(
            num_neurons, num_neurons,
            device='cpu', grad_factor=grad_factor,
            implementation='python', connections=connections,
        ))
    layers.append(GroupSum(k=num_classes, tau=tau, device='cpu'))
    model = nn.Sequential(*layers)

    # Load trained weights (connection indices + gate weights)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # Custom ops mapping for MASE graph — same as the demo notebook
    CUSTOM_OPS = {
        "modules": {
            LogicLayer: {
                "args": {"input": "data_in"},
                "toolchain": "INTERNAL_RTL",
                "module": "fixed_difflogic_logic",
                "dependence_files": [
                    "difflogic_layers/rtl/fixed_difflogic_logic.sv",
                    "difflogic_layers/rtl/fixed_difflogic_logic_neuron.sv",
                ],
            },
            GroupSum: {
                "args": {"input": "data_in"},
                "toolchain": "INTERNAL_RTL",
                "module": "fixed_difflogic_groupsum",
                "dependence_files": [
                    "difflogic_layers/rtl/fixed_difflogic_groupsum.sv",
                ],
            },
        },
        "functions": {},
    }

    # Build MaseGraph
    graph = MaseGraph(model, custom_ops=CUSTOM_OPS)

    # Dummy input: single flattened sample
    dummy_input = torch.zeros(1, input_features).to(device)

    # Run MASE analysis and hardware metadata passes
    graph, _ = init_metadata_analysis_pass(graph)
    graph, _ = add_common_metadata_analysis_pass(
        graph,
        pass_args={"dummy_in": {"input_1": dummy_input}},
    )
    graph, _ = difflogic_hardware_metadata_optimize_pass(graph)
    graph, _ = add_hardware_metadata_analysis_pass(graph)
    graph, _ = difflogic_hardware_force_fixed_flatten_pass(graph)

    # Emit RTL
    graph, _ = emit_verilog_top_transform_pass(
        graph,
        pass_args={"project_dir": output_dir, "top_name": "difflogic_top"},
    )
    graph, _ = emit_internal_rtl_transform_pass(
        graph,
        pass_args={"project_dir": output_dir},
    )

    rtl_dir = os.path.join(output_dir, "hardware", "rtl")
    if os.path.isdir(rtl_dir):
        sv_files = [f for f in os.listdir(rtl_dir) if f.endswith(".sv")]
    else:
        sv_files = []
        print(f"WARNING: RTL directory not found at {rtl_dir}")

    print(f"\nEmitted RTL to: {rtl_dir}")
    for f in sorted(sv_files):
        print(f"  {f}")

    # Count logic gates for summary
    total_gates = sum(
        l.out_dim for l in model if isinstance(l, LogicLayer)
    )
    print(f"\nModel summary:")
    print(f"  Logic gates : {total_gates:,}")
    print(f"  Layers      : {num_layers}")
    print(f"  Input dim   : {input_features}")
    print(f"  Classes     : {num_classes}")

    print(f"\nSynthesis command (on Vivado server):")
    print(f"  vivado -mode batch -source scripts/synth_difflogic.tcl "
          f"-tclargs {os.path.abspath(rtl_dir)} {os.path.abspath(output_dir)}/synth_results")


if __name__ == "__main__":
    main()
