#!/usr/bin/env python3
"""
Standalone BLIF export for a trained DWN checkpoint.

Does NOT import MaseGraph (avoids torchvision::nms conflict).
Optionally runs ABC Boolean minimisation if ABC is available.

Usage:
    python scripts/emit_blif_standalone.py --ckpt-name baseline_n6
    python scripts/emit_blif_standalone.py --ckpt /path/to/custom.pt
"""
import sys
import os
import types
import argparse
import shutil
import subprocess

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
from mase_components.dwn_layers.blif import emit_network_blif


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Export a DWN checkpoint to BLIF and optionally minimise with ABC.",
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
    return parser.parse_args()


def load_model(ckpt_path: str) -> "DWNModel":
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


def run_abc(blif_path: str, min_blif_path: str) -> None:
    abc_bin = shutil.which("abc") or os.path.expanduser("~/.local/bin/abc")
    if not os.path.exists(abc_bin):
        print("ABC not found — skipping Boolean minimisation.")
        print(
            "  Install ABC and place it at ~/.local/bin/abc (or anywhere on PATH)."
        )
        return

    print(f"\nRunning ABC minimisation: {abc_bin}")
    # strash converts to AIG; dc2 is technology-independent optimisation;
    # write_blif writes the minimised network back.  'map' is omitted because
    # it requires a cell library (.lib) which is not available here.
    cmd = (
        f"read_blif {blif_path}; "
        f"strash; "
        f"dc2; "
        f"write_blif {min_blif_path}; "
        f"print_stats"
    )
    result = subprocess.run(
        [abc_bin, "-c", cmd],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        print(f"ABC exited with code {result.returncode}")
    else:
        print(f"Minimised BLIF written to: {min_blif_path}")


def main():
    args = parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ckpt_dir = os.path.join(repo_root, "mase_output", "dwn")

    ckpt_name = args.ckpt_name
    ckpt_path = args.ckpt or os.path.join(ckpt_dir, f"{ckpt_name}.pt")

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    model = load_model(ckpt_path)

    # Output directory and file paths
    out_dir = os.path.join(ckpt_dir, f"{ckpt_name}_blif")
    os.makedirs(out_dir, exist_ok=True)

    blif_path = os.path.join(out_dir, f"{ckpt_name}.blif")
    min_blif_path = os.path.join(out_dir, f"{ckpt_name}_minimised.blif")

    emit_network_blif(model, blif_path)
    print(f"BLIF written to: {os.path.abspath(blif_path)}")

    run_abc(blif_path, min_blif_path)


if __name__ == "__main__":
    main()
