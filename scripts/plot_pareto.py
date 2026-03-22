"""
Pareto comparison plot: DWN (n=2,4,6) vs DiffLogic across MNIST, NID, JSC.

X axis: CLB LUT count (Vivado-measured or analytically estimated)
Y axis: Test accuracy (%)

Usage:
    python scripts/plot_pareto.py [--output pareto_plot.pdf]
"""

import argparse
import math
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def read_acc(path):
    """Return accuracy float from a checkpoint, or None if missing."""
    if not os.path.exists(path):
        warnings.warn(f"Checkpoint not found: {path}")
        return None
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        return ckpt.get("acc", ckpt.get("best_acc"))
    except Exception as e:
        warnings.warn(f"Failed to load {path}: {e}")
        return None


def read_model_config(path):
    """Return model_config dict from a checkpoint, or None."""
    if not os.path.exists(path):
        return None
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        return ckpt.get("model_config")
    except Exception as e:
        warnings.warn(f"Failed to load model_config from {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# LUT-count estimators
# ---------------------------------------------------------------------------

def dwn_lut_count(hidden_sizes, lut_n_list):
    """
    Estimate CLB LUT count for a DWN model.

    Each DWN LUT needs 2^n bits; a Xilinx LUT6 stores 64 bits.
    CLB LUTs per layer ≈ ceil(output_size * 2^n / 64)
    """
    # Normalise lut_n_list to match number of layers
    n_layers = len(hidden_sizes)
    if len(lut_n_list) == 1:
        lut_n_list = lut_n_list * n_layers
    elif len(lut_n_list) < n_layers:
        # Pad last value
        lut_n_list = lut_n_list + [lut_n_list[-1]] * (n_layers - len(lut_n_list))

    total = 0
    for size, n in zip(hidden_sizes, lut_n_list):
        total += math.ceil(size * (2 ** n) / 64)
    return total


def difflogic_lut_count(num_neurons, num_layers):
    """
    Estimate CLB LUT count for a DiffLogic model.

    Each LogicLayer gate uses a 4-bit truth table.
    CLB LUTs ≈ ceil(num_neurons * 4 / 64) = ceil(num_neurons / 16)
    """
    return math.ceil(num_neurons / 16) * num_layers


def parse_lut_n(cfg):
    """Parse lut_n from model_config (int or comma-separated string)."""
    raw = cfg.get("lut_n", 6)
    if isinstance(raw, int):
        return [raw]
    return [int(x) for x in str(raw).split(",")]


# ---------------------------------------------------------------------------
# Known Vivado-measured counts (from project MEMORY.md)
# ---------------------------------------------------------------------------

VIVADO_MEASURED = {
    ("mnist", "dwn_n6"):   1256,
    ("mnist", "dwn_n62"):   889,   # mixed n=[6,2]
    ("mnist", "dwn_n642"):  705,   # mixed n=[6,4,2]
}


# ---------------------------------------------------------------------------
# Build per-dataset point lists
# ---------------------------------------------------------------------------

DWN_DIR = "/home/khl22/mase-fork/mase_output/dwn"
DL_DIR  = "/home/khl22/mase-fork/mase_output/difflogic"


def collect_points():
    """
    Returns a dict:
        { dataset: [ {label, acc, lut_count, measured, marker_style, color}, ... ] }
    """
    datasets = ["mnist", "nid", "jsc"]
    results  = {ds: [] for ds in datasets}

    # -----------------------------------------------------------------------
    # DWN checkpoints
    # -----------------------------------------------------------------------
    dwn_specs = {
        "mnist": [
            ("mnist_n2.pt",      "DWN n=2", "blue",  "o"),
            # n=4 checkpoint not present — skip gracefully
            ("mnist_n4.pt",      "DWN n=4", "green", "s"),
            ("baseline_n6.pt",   "DWN n=6", "red",   "D"),
        ],
        "nid": [
            ("nid_n2.pt",  "DWN n=2", "blue",  "o"),
            ("nid_n4.pt",  "DWN n=4", "green", "s"),
            ("nid_n6.pt",  "DWN n=6", "red",   "D"),
        ],
        "jsc": [
            ("jsc_n2.pt",              "DWN n=2", "blue",  "o"),
            ("jsc_n4.pt",              "DWN n=4", "green", "s"),
            # Use the best JSC n=6 checkpoint available
            ("jsc_learnable_100ep.pt", "DWN n=6", "red",   "D"),
        ],
    }

    for ds, specs in dwn_specs.items():
        for fname, label, color, marker in specs:
            path = os.path.join(DWN_DIR, fname)
            acc  = read_acc(path)
            cfg  = read_model_config(path)
            if acc is None or cfg is None:
                print(f"  [SKIP] {ds}/{fname}: missing checkpoint")
                continue

            hidden_sizes = cfg["hidden_sizes"]
            lut_n_list   = parse_lut_n(cfg)
            est_luts     = dwn_lut_count(hidden_sizes, lut_n_list)

            # Check for Vivado measurement
            # Build a key suffix from lut_n_list
            n_str = "dwn_n" + "".join(str(n) for n in lut_n_list)
            vivado_key = (ds, n_str)
            measured_luts = VIVADO_MEASURED.get(vivado_key)
            lut_count = measured_luts if measured_luts is not None else est_luts
            measured  = measured_luts is not None

            results[ds].append({
                "label":    label,
                "acc":      acc * 100,
                "lut_count": lut_count,
                "measured": measured,
                "color":    color,
                "marker":   marker,
            })

    # -----------------------------------------------------------------------
    # DiffLogic checkpoints
    # -----------------------------------------------------------------------
    dl_specs = {
        "mnist": "difflogic_mnist.pt",
        "nid":   "difflogic_nid.pt",
        "jsc":   "difflogic_jsc.pt",
    }

    for ds, fname in dl_specs.items():
        path = os.path.join(DL_DIR, fname)
        acc  = read_acc(path)
        cfg  = read_model_config(path)
        if acc is None or cfg is None:
            print(f"  [SKIP] {ds}/{fname}: missing checkpoint")
            continue

        num_neurons = cfg.get("num_neurons", cfg.get("hidden_size", 256))
        num_layers  = cfg.get("num_layers", 1)
        est_luts    = difflogic_lut_count(num_neurons, num_layers)

        results[ds].append({
            "label":     "DiffLogic",
            "acc":       acc * 100,
            "lut_count": est_luts,
            "measured":  False,
            "color":     "purple",
            "marker":    "^",
        })

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(results, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "DWN vs DiffLogic Pareto: Accuracy vs CLB LUT Count",
        fontsize=14, fontweight="bold",
    )

    dataset_titles = {"mnist": "MNIST", "nid": "NID", "jsc": "JSC"}

    for ax, ds in zip(axes, ["mnist", "nid", "jsc"]):
        points = results[ds]
        ax.set_title(dataset_titles[ds], fontsize=12)
        ax.set_xlabel("CLB LUT Count", fontsize=10)
        ax.set_ylabel("Test Accuracy (%)", fontsize=10)
        ax.set_xscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

        if not points:
            ax.text(0.5, 0.5, "No data available",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="gray")
            continue

        for pt in points:
            luts     = pt["lut_count"]
            acc      = pt["acc"]
            measured = pt["measured"]
            color    = pt["color"]
            marker   = pt["marker"]
            label    = pt["label"]

            if measured:
                # Filled marker = Vivado-measured
                ax.scatter(luts, acc, marker=marker, color=color,
                           s=100, zorder=5, label=label)
                suffix = ""
            else:
                # Hollow marker = estimated
                ax.scatter(luts, acc, marker=marker, color=color,
                           s=100, zorder=5, facecolors="none",
                           edgecolors=color, linewidths=1.5, label=label + " (est)")
                suffix = " (est)"

            ax.annotate(
                f"{acc:.2f}%{suffix}",
                (luts, acc),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=7.5,
                color=color,
            )

        # De-duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            base = l.replace(" (est)", "")
            if base not in seen:
                seen[base] = h
        ax.legend(seen.values(), seen.keys(), fontsize=8, loc="lower right")

    # Global legend for filled vs hollow
    filled_patch  = mlines.Line2D([], [], marker="o", color="gray",
                                  markerfacecolor="gray", markersize=8,
                                  linestyle="None", label="Vivado-measured")
    hollow_patch  = mlines.Line2D([], [], marker="o", color="gray",
                                  markerfacecolor="none", markersize=8,
                                  linestyle="None", label="Estimated (analytical)")
    fig.legend(handles=[filled_patch, hollow_patch],
               loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot DWN vs DiffLogic Pareto frontier")
    parser.add_argument("--output", default="pareto_plot.pdf",
                        help="Output file path (PDF or PNG)")
    args = parser.parse_args()

    print("Collecting data points...")
    results = collect_points()

    total = sum(len(v) for v in results.values())
    print(f"  {total} points collected across {len(results)} datasets")

    plot(results, args.output)


if __name__ == "__main__":
    main()
