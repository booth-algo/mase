#!/usr/bin/env python3
"""
Standalone RTL emission for a trained DWN checkpoint.

Does NOT import MaseGraph (avoids torchvision version mismatch).
Uses sys.modules stubs to load only the DWN model classes.

Usage:
    python scripts/emit_rtl_standalone.py --ckpt-name baseline_n6
    python scripts/emit_rtl_standalone.py --ckpt /path/to/custom.pt
    python scripts/emit_rtl_standalone.py --ckpt-name baseline_n6 --pipelined
"""
import sys
import os
import types
import argparse
import shutil
from datetime import datetime

# ---------------------------------------------------------------------------
# sys.modules stubs — prevent the heavy chop/__init__.py and
# mase_components/__init__.py from running (they trigger torchvision::nms).
# ---------------------------------------------------------------------------
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, _src)

for _pkg in [
    "chop",
    "chop.nn",
    "mase_components",
]:
    if _pkg not in sys.modules:
        _mod = types.ModuleType(_pkg)
        _mod.__path__ = [os.path.join(_src, *_pkg.split("."))]
        _mod.__package__ = _pkg
        sys.modules[_pkg] = _mod

import torch

from chop.nn.dwn import DWNModel


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Emit SystemVerilog RTL from a DWN checkpoint (standalone, no MaseGraph).",
    )
    parser.add_argument(
        "--ckpt-name",
        type=str,
        default="baseline_n6",
        help="Checkpoint stem — looks in mase_output/dwn/<name>.pt",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Full checkpoint path (overrides --ckpt-name)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: mase_output/dwn/<ckpt-name>_rtl)",
    )
    parser.add_argument(
        "--pipelined",
        action="store_true",
        help="Also emit a pipelined (clocked) variant: dwn_top_clocked.sv",
    )
    return parser.parse_args()


def load_model(ckpt_path: str) -> "DWNModel":
    """Load checkpoint and reconstruct trained DWNModel."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["model_config"]
    print(
        f"  Config: hidden_sizes={cfg['hidden_sizes']}, lut_n={cfg['lut_n']}, "
        f"num_bits={cfg['num_bits']}, mapping_first={cfg['mapping_first']}"
    )

    # Strip training-only keys not accepted by DWNModel.__init__
    model_kwargs = {
        k: v for k, v in cfg.items() if k not in ("area_lambda", "lambda_reg")
    }
    model = DWNModel(**model_kwargs)

    # fit_thermometer must be called before load_state_dict so that the
    # thermometer.thresholds buffer is registered; values are overwritten
    # immediately by load_state_dict.
    model.fit_thermometer(torch.zeros(2, cfg["input_features"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def pack_lut_params(lut_layer):
    """
    Pack a trained LUTLayer's indices and contents into Verilog hex literals.

    Returns (input_size, output_size, lut_n, indices_hex, contents_hex) where
    indices_hex and contents_hex are strings of the form "<N>'h<HEX>".

    Replicates _pack_dwn_lut_params from mase_components/dwn_layers/passes.py.
    """
    indices = lut_layer.get_input_indices()  # (output_size, n) int32 tensor
    output_size = lut_layer.output_size
    lut_n = lut_layer.n
    lut_entries = 2 ** lut_n

    # Pack INPUT_INDICES: indices[i,k] -> bits [(i*lut_n+k)*8 +: 8]
    packed_indices = 0
    for i in range(output_size):
        for k in range(lut_n):
            idx = int(indices[i, k].item()) & 0xFF
            packed_indices |= idx << ((i * lut_n + k) * 8)

    # Pack LUT_CONTENTS: contents[i,j] -> bit [i*lut_entries + j]
    contents = lut_layer.get_lut_contents()  # (output_size, 2^n) int tensor
    packed_contents = 0
    for i in range(output_size):
        for j in range(lut_entries):
            bit = int(contents[i, j].item()) & 1
            packed_contents |= bit << (i * lut_entries + j)

    # Format as Verilog hex literals with explicit bit-width
    indices_bits = output_size * lut_n * 8
    contents_bits = output_size * lut_entries
    indices_hex = f"{indices_bits}'h{packed_indices:0{(indices_bits + 3) // 4}X}"
    contents_hex = f"{contents_bits}'h{packed_contents:0{(contents_bits + 3) // 4}X}"

    return lut_layer.input_size, output_size, lut_n, indices_hex, contents_hex


def emit_dwn_top_sv(layers_params, top_name="dwn_top"):
    """
    Generate dwn_top.sv SystemVerilog source as a string.

    layers_params: list of (input_size, output_size, lut_n, indices_hex, contents_hex)
    """
    n_layers = len(layers_params)
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    first_input_size_param = "lut_layers_0_INPUT_SIZE"
    last_output_size_param = f"lut_layers_{n_layers - 1}_OUTPUT_SIZE"

    lines = []

    # Header
    lines.append("// =====================================")
    lines.append("//     Mase Hardware")
    lines.append(f"//     Model: {top_name}")
    lines.append(f"//     {timestamp}")
    lines.append("// =====================================")
    lines.append("`timescale 1ns/1ps")

    # Module declaration with parameters
    lines.append(f"module {top_name} #(")
    param_lines = []
    for i, (in_sz, out_sz, lut_n, idx_hex, cont_hex) in enumerate(layers_params):
        prefix = f"lut_layers_{i}"
        param_lines.append(f"    parameter {prefix}_INPUT_SIZE = {in_sz}")
        param_lines.append(f"    parameter {prefix}_OUTPUT_SIZE = {out_sz}")
        param_lines.append(f"    parameter {prefix}_LUT_N = {lut_n}")
        param_lines.append(f"    parameter {prefix}_INPUT_INDICES = {idx_hex}")
        param_lines.append(f"    parameter {prefix}_LUT_CONTENTS = {cont_hex}")
    lines.append(",\n".join(param_lines))

    # Ports
    lines.append(") (")
    lines.append("    input clk,")
    lines.append("    input rst,")
    lines.append("")
    lines.append(f"\tinput [{first_input_size_param}-1:0] data_in_0,")
    lines.append(f"\tinput  data_in_0_valid,")
    lines.append(f"\toutput data_in_0_ready,")
    lines.append(f"\toutput [{last_output_size_param}-1:0] data_out_0,")
    lines.append(f"\toutput  data_out_0_valid,")
    lines.append(f"\tinput data_out_0_ready")
    lines.append(");")
    lines.append("")

    # Internal wires for each layer
    for i in range(n_layers):
        prefix = f"lut_layers_{i}"
        lines.append("// --------------------------")
        lines.append(f"//   {prefix} signals")
        lines.append("// --------------------------")
        lines.append(f"logic [{prefix}_INPUT_SIZE-1:0] {prefix}_data_in_0;")
        lines.append(f"logic {prefix}_data_in_0_valid;")
        lines.append(f"logic {prefix}_data_in_0_ready;")
        lines.append(f"logic [{prefix}_OUTPUT_SIZE-1:0] {prefix}_data_out_0;")
        lines.append(f"logic {prefix}_data_out_0_valid;")
        lines.append(f"logic {prefix}_data_out_0_ready;")

    lines.append("")
    lines.append("// --------------------------")
    lines.append("//   Component instantiation")
    lines.append("// --------------------------")

    # Instantiate each layer
    for i in range(n_layers):
        prefix = f"lut_layers_{i}"
        lines.append("")
        lines.append(f"// {prefix}")
        lines.append("fixed_dwn_lut_layer #(")
        lines.append(f"    .INPUT_SIZE({prefix}_INPUT_SIZE),")
        lines.append(f"    .OUTPUT_SIZE({prefix}_OUTPUT_SIZE),")
        lines.append(f"    .LUT_N({prefix}_LUT_N),")
        lines.append(f"    .INPUT_INDICES({prefix}_INPUT_INDICES),")
        lines.append(f"    .LUT_CONTENTS({prefix}_LUT_CONTENTS)")
        lines.append(f") {prefix}_inst (")
        lines.append(f"    .clk(clk),")
        lines.append(f"    .rst(rst),")
        lines.append(f"")
        lines.append(f"    .data_in_0({prefix}_data_in_0),")
        lines.append(f"    .data_in_0_valid({prefix}_data_in_0_valid),")
        lines.append(f"    .data_in_0_ready({prefix}_data_in_0_ready),")
        lines.append(f"        ")
        lines.append(f"    .data_out_0({prefix}_data_out_0),")
        lines.append(f"    .data_out_0_valid({prefix}_data_out_0_valid),")
        lines.append(f"    .data_out_0_ready({prefix}_data_out_0_ready)")
        lines.append(f");")

    lines.append("")
    lines.append("")
    lines.append("// --------------------------")
    lines.append("//   Interconnections")
    lines.append("// --------------------------")
    lines.append("    ")

    last = n_layers - 1

    # First layer <- top input
    lines.append("assign data_in_0_ready = lut_layers_0_data_in_0_ready;")
    lines.append("assign lut_layers_0_data_in_0_valid    = data_in_0_valid;")
    lines.append("assign lut_layers_0_data_in_0    = data_in_0;")
    lines.append("")

    # Last layer -> top output
    lines.append(f"assign data_out_0_valid = lut_layers_{last}_data_out_0_valid;")
    lines.append(f"assign lut_layers_{last}_data_out_0_ready    = data_out_0_ready;")
    lines.append(f"assign data_out_0 = lut_layers_{last}_data_out_0;")

    # Chain adjacent layers
    for i in range(n_layers - 1):
        lines.append("")
        lines.append(f"assign lut_layers_{i}_data_out_0_ready  = lut_layers_{i+1}_data_in_0_ready;")
        lines.append(f"assign lut_layers_{i+1}_data_in_0_valid    = lut_layers_{i}_data_out_0_valid;")
        lines.append(f"assign lut_layers_{i+1}_data_in_0 = lut_layers_{i}_data_out_0;")

    lines.append("")
    lines.append("endmodule")

    return "\n".join(lines)


def main():
    args = parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ckpt_dir = os.path.join(repo_root, "mase_output", "dwn")

    ckpt_name = args.ckpt_name
    ckpt_path = args.ckpt or os.path.join(ckpt_dir, f"{ckpt_name}.pt")

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(ckpt_dir, f"{ckpt_name}_rtl")
    rtl_dir = os.path.join(output_dir, "hardware", "rtl")
    os.makedirs(rtl_dir, exist_ok=True)

    model = load_model(ckpt_path)

    # Extract parameters from each LUT layer
    layers_params = []
    for i, layer in enumerate(model.lut_layers):
        print(f"  Packing layer {i}: input_size={layer.input_size}, "
              f"output_size={layer.output_size}, n={layer.n}")
        params = pack_lut_params(layer)
        layers_params.append(params)

    # Generate dwn_top.sv
    top_sv = emit_dwn_top_sv(layers_params, top_name="dwn_top")
    top_path = os.path.join(rtl_dir, "dwn_top.sv")
    with open(top_path, "w") as f:
        f.write(top_sv)
    print(f"\nEmitted: {top_path}")

    # Copy static RTL component files
    static_rtl_dir = os.path.join(
        repo_root, "src", "mase_components", "dwn_layers", "rtl"
    )
    static_files = [
        "fixed_dwn_lut_layer.sv",
        "fixed_dwn_lut_neuron.sv",
        "fixed_dwn_groupsum.sv",
        "fixed_dwn_thermometer.sv",
    ]
    for fname in static_files:
        src_path = os.path.join(static_rtl_dir, fname)
        if os.path.exists(src_path):
            dst_path = os.path.join(rtl_dir, fname)
            shutil.copy(src_path, dst_path)
            print(f"  Copied: {fname}")

    # Pipelined variant
    if args.pipelined:
        clocked_sv = top_sv.replace("module dwn_top", "module dwn_top_clocked")
        clocked_sv = clocked_sv.replace(
            "fixed_dwn_lut_layer ", "fixed_dwn_lut_layer_clocked "
        )
        clocked_top_path = os.path.join(rtl_dir, "dwn_top_clocked.sv")
        with open(clocked_top_path, "w") as f:
            f.write(clocked_sv)
        print(f"\n  Emitted: dwn_top_clocked.sv")

        clocked_layer_src = os.path.join(static_rtl_dir, "fixed_dwn_lut_layer_clocked.sv")
        if os.path.exists(clocked_layer_src):
            shutil.copy(clocked_layer_src, os.path.join(rtl_dir, "fixed_dwn_lut_layer_clocked.sv"))
            print(f"  Copied: fixed_dwn_lut_layer_clocked.sv")

    # Summary
    sv_files = sorted(f for f in os.listdir(rtl_dir) if f.endswith(".sv"))
    print(f"\nRTL directory: {rtl_dir}")
    print(f"Files ({len(sv_files)}):")
    for f in sv_files:
        print(f"  {f}")
    print(f"\nSynthesis command (on Vivado server):")
    print(f"  vivado -mode batch -source scripts/synth_dwn.tcl "
          f"-tclargs {os.path.abspath(rtl_dir)} "
          f"{os.path.abspath(output_dir)}/synth_results")


if __name__ == "__main__":
    main()
