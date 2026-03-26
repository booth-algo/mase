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
import re
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
    parser.add_argument("--emit-blif", action="store_true",
                        help="Also emit a BLIF file for ABC Boolean minimisation")
    parser.add_argument("--pipelined", action="store_true",
                        help="Also emit a pipelined (clocked) variant: dwn_top_clocked.sv")
    parser.add_argument("--full-pipeline", action="store_true",
                        help="Also emit full pipeline wrappers (thermometer + LUT + groupsum)")
    parser.add_argument("--feature-width", type=int, default=8,
                        help="Bits per input feature for full-pipeline thermometer encoding")
    return parser.parse_args()


def main():
    args = parse_args()

    if hasattr(args, 'ckpt_name') and args.ckpt_name and not re.match(r'^[\w\-]+$', args.ckpt_name):
        print(f"ERROR: --ckpt-name must be alphanumeric/hyphens/underscores, got: {args.ckpt_name}")
        sys.exit(1)

    ckpt_dir = os.path.join(os.path.dirname(__file__), "../mase_output/dwn")
    ckpt_path = args.ckpt or os.path.join(ckpt_dir, f"{args.ckpt_name}.pt")
    output_dir = args.output_dir or os.path.join(ckpt_dir, f"{args.ckpt_name}_rtl")

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")

    from mase_components.dwn_layers.emit import emit_dwn_rtl
    result = emit_dwn_rtl(
        ckpt_path=ckpt_path,
        output_dir=output_dir,
        emit_pipelined=args.pipelined,
        emit_blif=args.emit_blif,
        full_pipeline=args.full_pipeline,
        feature_width=args.feature_width,
    )

    print(f"\nEmitted RTL to: {result['rtl_dir']}")
    for f in result['sv_files']:
        print(f"  {f}")
    print(f"\nSynthesis command (on Vivado server):")
    print(f"  vivado -mode batch -source scripts/synth_dwn.tcl "
          f"-tclargs {result['rtl_dir']} {result['output_dir']}/synth_results")


if __name__ == "__main__":
    main()
