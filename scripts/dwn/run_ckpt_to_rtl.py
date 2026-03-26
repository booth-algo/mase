#!/usr/bin/env python3
"""
Load a trained DWN checkpoint and emit Verilog RTL using the MASE pipeline.

[Example usage]
    python run_ckpt_to_rtl.py --ckpt mase_output/dwn/best.pt
    python run_ckpt_to_rtl.py --ckpt mase_output/dwn/best.pt --out-dir /tmp/dwn_rtl --top-name my_dwn
"""
import sys
import os
import re
import types
import argparse

#
# sys.path setup — must come before any chop/mase_components imports
#

_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
if _src not in sys.path:
    sys.path.insert(0, _src)

import logging as _logging
import torch.nn as _nn
# Stub packages whose __init__.py has heavy/missing deps (onnx, einops, py3.10+)
_STUBS = [
    'chop', 'chop.nn', 'chop.nn.quantized', 'chop.nn.quantized.functional',
    'chop.nn.quantized.modules', 'chop.nn.quantizers', 'chop.nn.modules',
    'chop.tools', 'chop.ir',
    'chop.passes', 'chop.passes.graph',
    'chop.passes.graph.analysis', 'chop.passes.graph.analysis.add_metadata',
    'chop.passes.graph.transforms', 'chop.passes.graph.transforms.verilog',
    'chop.passes.graph.transforms.verilog.logicnets',
    'chop.passes.graph.transforms.verilog.logicnets.emit_linear',
]
for _pkg in _STUBS:
    if _pkg not in sys.modules:
        _mod = types.ModuleType(_pkg)
        _mod.__path__ = [os.path.join(_src, *_pkg.split('.'))]
        _mod.__package__ = _pkg
        sys.modules[_pkg] = _mod
# Provide attrs required by leaf modules without importing py3.10+/heavy code
sys.modules['chop.nn'].MASE_LEAF_LAYERS = ()
sys.modules['chop.nn.quantized'].quantized_func_map = {}
sys.modules['chop.nn.quantized'].quantized_module_map = {}
sys.modules['chop.nn.quantized.modules'].quantized_module_map = {}
sys.modules['chop.nn.quantized.functional'].quantized_func_map = {}
sys.modules['chop.tools'].get_logger = lambda name: _logging.getLogger(name)
sys.modules['chop.tools'].get_hf_dummy_in = None
sys.modules['chop.nn.modules'].GroupedQueryAttention = type(
    'GroupedQueryAttention', (_nn.Module,), {'forward': lambda self, x: x}
)
# logicnets uses quantized linear — DWN doesn't need it, stub it out
sys.modules['chop.passes.graph.transforms.verilog.logicnets'].LogicNetsLinearVerilog = None
sys.modules['chop.passes.graph.transforms.verilog.logicnets.emit_linear'].LogicNetsLinearVerilog = None

import torch
import torch.nn as nn

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a trained DWN checkpoint and emit Verilog RTL via MASE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="Path to .pt checkpoint file produced by run_dwn_training.py",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory (default: mase_output/dwn/<ckpt_stem>_rtl)",
    )
    parser.add_argument(
        "--top-name", type=str, default="dwn_top",
        help="Verilog top module name",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for model evaluation",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', args.top_name):
        print(f"ERROR: --top-name must be a valid Verilog identifier, got: {args.top_name}")
        return 1

    torch.manual_seed(args.seed)

    ckpt_path = os.path.abspath(args.ckpt)
    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        return 1

    if args.out_dir is not None:
        out_dir = os.path.abspath(args.out_dir)
    else:
        ckpt_stem = os.path.splitext(os.path.basename(ckpt_path))[0]
        _default_base = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../mase_output/dwn")
        )
        out_dir = os.path.join(_default_base, f"{ckpt_stem}_rtl")

    print(f"Loading checkpoint: {ckpt_path}")
    print(f"Output directory: {out_dir}")

    from mase_components.dwn_layers.emit import emit_dwn_rtl
    result = emit_dwn_rtl(
        ckpt_path=ckpt_path,
        output_dir=out_dir,
        top_name=args.top_name,
        device=args.device,
    )

    print()
    print("=== RTL Emitted Successfully ===")
    print(f"  Output : {result['output_dir']}/")
    print("  Files:")
    for f in result['sv_files']:
        print(f"    {f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
