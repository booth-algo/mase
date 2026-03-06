#!/usr/bin/env python3
"""
Load a trained DWN checkpoint and emit Verilog RTL using the MASE pipeline.

[Example usage]
    python run_ckpt_to_rtl.py --ckpt mase_output/dwn/best.pt
    python run_ckpt_to_rtl.py --ckpt mase_output/dwn/best.pt --out-dir /tmp/dwn_rtl --top-name my_dwn
"""
import sys
import os
import types
import argparse

# ---------------------------------------------------------------------------
# sys.path setup — must come before any chop/mase_components imports
# ---------------------------------------------------------------------------

_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../src'))
if _src not in sys.path:
    sys.path.insert(0, _src)

for _pkg in ['chop', 'chop.nn']:
    if _pkg not in sys.modules:
        _mod = types.ModuleType(_pkg)
        _mod.__path__ = [os.path.join(_src, *_pkg.split('.'))]
        _mod.__package__ = _pkg
        sys.modules[_pkg] = _mod

import torch
import torch.nn as nn

from chop.nn.dwn import DWNModel


# ---------------------------------------------------------------------------
# DWNHardwareCore — LUT layers only (no thermometer, no group_sum)
# ---------------------------------------------------------------------------

class DWNHardwareCore(nn.Module):
    """Thin wrapper around the LUT-layer stack for RTL emission."""

    def __init__(self, lut_layers):
        super().__init__()
        self.lut_layers = nn.ModuleList(lut_layers)

    def forward(self, x):
        for layer in self.lut_layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
        "--fit-data", type=int, default=640,
        help="(Unused — thermometer state is restored from checkpoint) "
             "Number of random samples that *would* be used for fitting if needed",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # --- device selection ---
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}" + (
        f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    ))

    # --- load checkpoint ---
    ckpt_path = os.path.abspath(args.ckpt)
    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        return 1

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    epoch = ckpt.get("epoch", "?")
    acc   = ckpt.get("acc",   float("nan"))
    cfg   = ckpt.get("model_config")

    if cfg is None:
        print("ERROR: checkpoint has no 'model_config' key — cannot reconstruct model.")
        return 1

    print(f"  Epoch : {epoch}")
    print(f"  Acc   : {acc:.4f}")
    print(f"  Config: input_features={cfg['input_features']}, num_classes={cfg['num_classes']}, "
          f"hidden_sizes={cfg['hidden_sizes']}, lut_n={cfg['lut_n']}, "
          f"num_bits={cfg['num_bits']}, mapping_first={cfg.get('mapping_first', 'learnable')}")

    # --- reconstruct model (handle missing keys in older checkpoints) ---
    _DWN_VALID_KEYS = {"input_features", "num_classes", "num_bits", "hidden_sizes", "lut_n", "mapping_first", "mapping_rest", "tau", "lambda_reg"}
    model_kwargs = {
        "input_features":  cfg["input_features"],
        "num_classes":     cfg["num_classes"],
        "num_bits":        cfg["num_bits"],
        "hidden_sizes":    cfg["hidden_sizes"],
        "lut_n":           cfg["lut_n"],
        "mapping_first":   cfg.get("mapping_first", "learnable"),
        "mapping_rest":    cfg.get("mapping_rest",  "random"),
        "tau":             cfg.get("tau",            3.333),
        "lambda_reg":      cfg.get("lambda_reg",     0.0),
    }
    # Filter to only valid DWNModel keys (area_lambda is not supported)
    model_kwargs = {k: v for k, v in model_kwargs.items() if k in _DWN_VALID_KEYS}

    model = DWNModel(**model_kwargs)

    # Load state dict BEFORE any thermometer fitting so that saved thresholds
    # are restored directly from the checkpoint (no re-fitting needed).
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # --- extract LUT layers and build hardware core ---
    lut_layers = list(model.lut_layers)
    hw_model = DWNHardwareCore(lut_layers)
    hw_model.eval()

    # dummy_input: binary tensor (1, input_features * num_bits)
    # This is what the first LUT layer receives (post-thermometer encoding).
    input_features  = cfg["input_features"]
    num_bits        = cfg["num_bits"]
    binary_input_size = input_features * num_bits
    dummy_input = torch.zeros(1, binary_input_size, device=device)

    # --- output directory ---
    if args.out_dir is not None:
        out_dir = os.path.abspath(args.out_dir)
    else:
        ckpt_stem = os.path.splitext(os.path.basename(ckpt_path))[0]
        _default_base = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../../../mase_output/dwn")
        )
        out_dir = os.path.join(_default_base, f"{ckpt_stem}_rtl")

    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # --- emit RTL ---
    print("Running MASE emit pipeline...")

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
        graph, pass_args={"dummy_in": {"x": dummy_input}, "add_value": False}
    )
    graph, _ = add_hardware_metadata_analysis_pass(graph)
    graph, _ = dwn_hardware_metadata_pass(graph)
    graph, _ = emit_verilog_top_transform_pass(
        graph, pass_args={"project_dir": out_dir, "top_name": args.top_name}
    )
    graph, _ = emit_internal_rtl_transform_pass(
        graph, pass_args={"project_dir": out_dir}
    )

    # --- report ---
    rtl_dir = os.path.join(out_dir, "hardware", "rtl")
    top_sv  = os.path.join(rtl_dir, f"{args.top_name}.sv")

    print()
    print("=== RTL Emitted Successfully ===")
    print(f"  Checkpoint : {ckpt_path} (epoch {epoch}, acc {acc:.3f})")
    print(f"  Config     : hidden_sizes={cfg['hidden_sizes']}, lut_n={cfg['lut_n']}, "
          f"num_bits={num_bits}")
    print(f"  Output     : {out_dir}/")
    print("  Files:")

    # List files that should have been emitted
    expected_files = [
        os.path.join("hardware", "rtl", f"{args.top_name}.sv"),
        os.path.join("hardware", "rtl", "fixed_dwn_lut_layer.sv"),
        os.path.join("hardware", "rtl", "fixed_dwn_lut_neuron.sv"),
    ]
    for rel in expected_files:
        abs_path = os.path.join(out_dir, rel)
        status = "" if os.path.exists(abs_path) else "  [NOT FOUND]"
        print(f"    {rel}{status}")

    if not os.path.exists(top_sv):
        print(f"\nWARNING: expected top-level file not found: {top_sv}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
