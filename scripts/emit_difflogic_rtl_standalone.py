#!/usr/bin/env python3
"""
Standalone DiffLogic RTL emitter — bypasses MaseGraph to avoid torchvision::nms import error.

Usage:
    python scripts/emit_difflogic_rtl_standalone.py --ckpt-name difflogic_mnist
    python scripts/emit_difflogic_rtl_standalone.py --ckpt-name difflogic_mnist --output-dir hw/mnist_rtl

Output directory contains:
    hardware/rtl/difflogic_top.sv
    hardware/rtl/fixed_difflogic_logic.sv
    hardware/rtl/fixed_difflogic_logic_neuron.sv
    hardware/rtl/fixed_difflogic_groupsum.sv
    hardware/rtl/fixed_difflogic_flatten.sv
"""

import argparse
import math
import os
import shutil
import sys
import types

# ---------------------------------------------------------------------------
# Stub torchvision so that importing difflogic (which may pull in chop which
# pulls in torchvision) does not crash on the torchvision::nms C-extension.
# ---------------------------------------------------------------------------
for _mod in ["torchvision", "torchvision.ops", "torchvision.ops.nms"]:
    sys.modules.setdefault(_mod, types.ModuleType(_mod))

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clog2(n: int) -> int:
    """Equivalent to SystemVerilog $clog2(n)."""
    if n <= 1:
        return 1
    return math.ceil(math.log2(n))


def _sv_array(values, width: int) -> str:
    """Format a Python list as a SystemVerilog packed-array literal.

    Example: '{4'd12, 4'd7, ...}
    """
    items = ", ".join(f"{width}'d{v}" for v in values)
    return "'{" + items + "}"


# ---------------------------------------------------------------------------
# Top-level SV generation
# ---------------------------------------------------------------------------

def emit_top_sv(layers_info: list, input_width: int, num_classes: int) -> str:
    """Generate difflogic_top.sv.

    layers_info: list of dicts with keys:
        in_dim, out_dim, ind_a (list[int]), ind_b (list[int]), op_codes (list[int])
    """
    num_layers = len(layers_info)
    last_out = layers_info[-1]["out_dim"]

    lines = []
    lines.append("`timescale 1ns / 1ps")
    lines.append(f"module difflogic_top (")
    lines.append(f"    input  wire clk,")
    lines.append(f"    input  wire rst,")
    lines.append(f"    input  wire [{input_width-1}:0] data_in_0,")
    lines.append(f"    input  wire data_in_0_valid,")
    lines.append(f"    output wire data_in_0_ready,")
    gs_w = _clog2(last_out // num_classes)
    lines.append(f"    output wire [{gs_w}:0] data_out_0 [0:{num_classes-1}],")
    lines.append(f"    output wire data_out_0_valid,")
    lines.append(f"    input  wire data_out_0_ready")
    lines.append(f");")
    lines.append("")

    # Inter-layer wires
    for i, info in enumerate(layers_info):
        out_dim = info["out_dim"]
        lines.append(f"  wire [{out_dim-1}:0] layer_{i}_out;")
        lines.append(f"  wire layer_{i}_out_valid;")
        lines.append(f"  wire layer_{i}_out_ready;")
    lines.append("")

    # Instantiate logic layers
    for i, info in enumerate(layers_info):
        in_dim  = info["in_dim"]
        out_dim = info["out_dim"]
        ind_a   = info["ind_a"]
        ind_b   = info["ind_b"]
        op_codes = info["op_codes"]
        idx_w   = _clog2(in_dim)

        in_data   = "data_in_0"      if i == 0 else f"layer_{i-1}_out"
        in_valid  = "data_in_0_valid" if i == 0 else f"layer_{i-1}_out_valid"
        in_ready  = "data_in_0_ready" if i == 0 else f"layer_{i-1}_out_ready"

        lines.append(f"  fixed_difflogic_logic #(")
        lines.append(f"    .DATA_IN_0_TENSOR_SIZE_DIM_0({in_dim}),")
        lines.append(f"    .DATA_OUT_0_TENSOR_SIZE_DIM_0({out_dim}),")
        lines.append(f"    .LAYER_OP_CODES({_sv_array(op_codes, 4)}),")
        lines.append(f"    .IND_A({_sv_array(ind_a, idx_w)}),")
        lines.append(f"    .IND_B({_sv_array(ind_b, idx_w)})")
        lines.append(f"  ) layer_{i}_inst (")
        lines.append(f"    .clk(clk),")
        lines.append(f"    .rst(rst),")
        lines.append(f"    .data_in_0({in_data}),")
        lines.append(f"    .data_in_0_valid({in_valid}),")
        lines.append(f"    .data_in_0_ready({in_ready}),")
        lines.append(f"    .data_out_0(layer_{i}_out),")
        lines.append(f"    .data_out_0_valid(layer_{i}_out_valid),")
        lines.append(f"    .data_out_0_ready(layer_{i}_out_ready)")
        lines.append(f"  );")
        lines.append("")

    # GroupSum
    last_i = num_layers - 1
    lines.append(f"  fixed_difflogic_groupsum #(")
    lines.append(f"    .DATA_IN_0_TENSOR_SIZE_DIM_0({last_out}),")
    lines.append(f"    .DATA_OUT_0_TENSOR_SIZE_DIM_0({num_classes})")
    lines.append(f"  ) groupsum_inst (")
    lines.append(f"    .clk(clk),")
    lines.append(f"    .rst(rst),")
    lines.append(f"    .data_in_0(layer_{last_i}_out),")
    lines.append(f"    .data_in_0_valid(layer_{last_i}_out_valid),")
    lines.append(f"    .data_in_0_ready(layer_{last_i}_out_ready),")
    lines.append(f"    .data_out_0(data_out_0),")
    lines.append(f"    .data_out_0_valid(data_out_0_valid),")
    lines.append(f"    .data_out_0_ready(data_out_0_ready)")
    lines.append(f"  );")
    lines.append("")
    lines.append("endmodule")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt-name", type=str, default="difflogic_mnist",
                   help="Checkpoint stem (looks in mase_output/difflogic/<name>.pt)")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Full checkpoint path (overrides --ckpt-name)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory (default: mase_output/difflogic/<ckpt-name>_rtl)")
    return p.parse_args()


def main():
    args = parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ckpt_dir  = os.path.join(repo_root, "mase_output", "difflogic")
    ckpt_path = args.ckpt or os.path.join(ckpt_dir, f"{args.ckpt_name}.pt")
    output_dir = args.output_dir or os.path.join(ckpt_dir, f"{args.ckpt_name}_rtl")
    rtl_dir = os.path.join(output_dir, "hardware", "rtl")

    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg  = ckpt["model_config"]
    print(f"  num_neurons={cfg['num_neurons']}, num_layers={cfg['num_layers']}, "
          f"input_features={cfg['input_features']}, num_classes={cfg['num_classes']}, "
          f"tau={cfg['tau']}, connections={cfg['connections']}")

    # ------------------------------------------------------------------
    # Import difflogic (torchvision already stubbed above)
    # ------------------------------------------------------------------
    try:
        from difflogic import LogicLayer, GroupSum
    except ImportError:
        print("ERROR: 'difflogic' package not found. Install with: pip install difflogic",
              file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Reconstruct model
    # ------------------------------------------------------------------
    input_features = cfg["input_features"]
    num_classes    = cfg["num_classes"]
    num_neurons    = cfg["num_neurons"]
    num_layers     = cfg["num_layers"]
    grad_factor    = cfg.get("grad_factor", 1.0)
    tau            = cfg["tau"]
    connections    = cfg.get("connections", "random")

    layers = [nn.Flatten()]
    layers.append(LogicLayer(
        input_features, num_neurons,
        device="cpu", grad_factor=grad_factor,
        implementation="python", connections=connections,
    ))
    for _ in range(num_layers - 1):
        layers.append(LogicLayer(
            num_neurons, num_neurons,
            device="cpu", grad_factor=grad_factor,
            implementation="python", connections=connections,
        ))
    layers.append(GroupSum(k=num_classes, tau=tau, device="cpu"))
    model = nn.Sequential(*layers)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # ------------------------------------------------------------------
    # Extract per-layer indices and op codes
    # ------------------------------------------------------------------
    layers_info = []
    for layer in model:
        if not isinstance(layer, LogicLayer):
            continue
        ind_a, ind_b = layer.indices          # each shape [num_neurons]
        op_codes = torch.argmax(layer.weights, dim=1)  # shape [num_neurons]
        layers_info.append({
            "in_dim":   layer.in_dim,
            "out_dim":  layer.out_dim,
            "ind_a":    ind_a.tolist(),
            "ind_b":    ind_b.tolist(),
            "op_codes": op_codes.tolist(),
        })

    if not layers_info:
        print("ERROR: No LogicLayer found in model.", file=sys.stderr)
        sys.exit(1)

    print(f"  Extracted {len(layers_info)} LogicLayer(s)")

    # ------------------------------------------------------------------
    # Emit top-level SV
    # ------------------------------------------------------------------
    os.makedirs(rtl_dir, exist_ok=True)

    top_sv = emit_top_sv(layers_info, input_width=input_features, num_classes=num_classes)
    top_path = os.path.join(rtl_dir, "difflogic_top.sv")
    with open(top_path, "w") as f:
        f.write(top_sv)
    print(f"  Wrote: {top_path}")

    # ------------------------------------------------------------------
    # Copy component SV files
    # ------------------------------------------------------------------
    components_src = os.path.join(
        repo_root, "src", "mase_components", "difflogic_layers", "rtl"
    )
    component_files = [
        "fixed_difflogic_logic.sv",
        "fixed_difflogic_logic_neuron.sv",
        "fixed_difflogic_groupsum.sv",
        "fixed_difflogic_flatten.sv",
    ]
    for fname in component_files:
        src = os.path.join(components_src, fname)
        dst = os.path.join(rtl_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied: {fname}")
        else:
            print(f"  WARNING: component not found: {src}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_neurons = sum(info["out_dim"] for info in layers_info)
    lut_estimate  = math.ceil(total_neurons / 16) * len(layers_info)

    print(f"\nEmitted RTL to: {rtl_dir}")
    for f in sorted(os.listdir(rtl_dir)):
        if f.endswith(".sv"):
            print(f"  {f}")

    print(f"\nModel summary:")
    print(f"  Logic layers  : {len(layers_info)}")
    print(f"  Input width   : {input_features} bits")
    print(f"  Neurons/layer : {[info['out_dim'] for info in layers_info]}")
    print(f"  Total neurons : {total_neurons:,}")
    print(f"  Classes       : {num_classes}")

    print(f"\nAnalytical LUT estimate: ceil({total_neurons}/16) * {len(layers_info)} = {lut_estimate:,} LUTs")

    synth_results = os.path.join(output_dir, "synth_results")
    synth_tcl     = os.path.join(repo_root, "scripts", "synth_difflogic.tcl")
    print(f"\nSynthesis command (on Vivado server):")
    print(f"  vivado -mode batch -source {synth_tcl} \\")
    print(f"         -tclargs {os.path.abspath(rtl_dir)} {os.path.abspath(synth_results)}")


if __name__ == "__main__":
    main()
