#!/usr/bin/env python3
"""
Pareto sweep for DWN hardware-aware regularization.

Sweeps `area_lambda` values to produce an area-accuracy Pareto front.
For each lambda, trains a DWNModel and records:
  - area_luts: sum_l(output_size_l * 2^n_l) — LUT table storage (integer, architecture-driven)
  - accuracy:  test accuracy after training

NOTE on metrics:
  `area_luts` captures LUT table *storage* (determined by architecture/fan-in),
  which stays roughly constant across lambdas since it depends on layer sizes and n,
  not on learned weights. The *real* hardware effect of area_lambda is on `entropy_loss`,
  which penalizes high-entropy routing connections (each LUT input attending to many
  source features). Lower entropy → more concentrated, routing-friendly connections →
  lower real routing resource usage on FPGA. The Pareto curve therefore primarily reflects
  the entropy-vs-accuracy trade-off; area_luts is included as a supplementary metric.

[Example usage]
    # Quick sweep with fake data
    python run_pareto_sweep.py --search-epochs 5

    # Real MNIST sweep
    python run_pareto_sweep.py --dataset mnist --search-epochs 15 \\
        --lambdas 0.0 1e-4 5e-4 1e-3 5e-3 1e-2

    # Custom lambdas and architecture
    python run_pareto_sweep.py --hidden-sizes 500 250 --lut-n 4 \\
        --lambdas 0 0.001 0.01 0.1
"""
import sys
import os
import types
import argparse
import csv

_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../src'))
sys.path.insert(0, _src)

for _pkg in ['chop', 'chop.nn']:
    if _pkg not in sys.modules:
        _mod = types.ModuleType(_pkg)
        _mod.__path__ = [os.path.join(_src, *_pkg.split('.'))]
        _mod.__package__ = _pkg
        sys.modules[_pkg] = _mod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from chop.nn.dwn import DWNModel


# ---------------------------------------------------------------------------
# Inline copy of compute_area_loss (duplicated to avoid sibling-module import)
# ---------------------------------------------------------------------------

def compute_area_loss(model):
    """
    Hardware-aware regularization for DWN.

    Two components:
    1. Mapping entropy loss (differentiable): for LUT layers with LearnableMapping,
       penalize high entropy of mapping attention weights. High entropy means each
       LUT input attends to many source features (high routing complexity).
       Low entropy = concentrated connections = fewer effective routing resources.

    2. Area metric (non-differentiable, logged only): sum_l(output_size_l * 2^n_l)
       = total LUT table storage across all layers. Logged as 'area_luts' for
       Pareto analysis between area and accuracy.

    Returns:
        (entropy_loss: Tensor, area_luts: int)
    """
    from chop.nn.dwn.mapping import LearnableMapping

    entropy_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    area_luts = 0

    for layer in model.lut_layers:
        # Area metric (logged, not gradient-flowed)
        area_luts += layer.output_size * (2 ** layer.n)

        # Mapping entropy regularization (differentiable)
        if isinstance(layer.mapping, LearnableMapping):
            # weights: (input_size, output_size * n)
            # Each column is a distribution over input_size source features
            W = layer.mapping.weights   # (input_size, output_size * n)
            # Temperature: use the layer's mapping tau (softmax temperature)
            tau = getattr(layer.mapping, 'tau', 0.001)
            probs = torch.softmax(W / tau, dim=0)   # (input_size, output_size*n)
            # Shannon entropy per mapping position, averaged
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum(dim=0).mean()  # scalar
            entropy_loss = entropy_loss + entropy

    return entropy_loss, area_luts


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

TABULAR_DATASETS = [
    "phoneme", "skin-seg", "higgs", "australian", "nomao",
    "segment", "miniboone", "christine", "jasmine", "sylvine", "blood",
]


def _fake_data(n_train=640):
    print(f"Using fake random data ({n_train} train samples, 200 test samples)")
    X_train = torch.randn(n_train, 784)
    y_train = torch.randint(0, 10, (n_train,))
    X_test  = torch.randn(200, 784)
    y_test  = torch.randint(0, 10, (200,))
    return X_train, y_train, X_test, y_test, 784, 10


def _load_vision(dataset_name, seed):
    try:
        from torchvision import datasets as tvdatasets, transforms
    except ImportError:
        print("torchvision not available; falling back to fake data")
        return _fake_data()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    cls = tvdatasets.MNIST if dataset_name == "mnist" else tvdatasets.FashionMNIST
    cache = os.path.expanduser(f"~/.cache/{dataset_name}")
    train_ds = cls(cache, train=True,  download=True, transform=transform)
    test_ds  = cls(cache, train=False, download=True, transform=transform)
    X_train = torch.stack([x for x, _ in train_ds])
    y_train = torch.tensor([y for _, y in train_ds])
    X_test  = torch.stack([x for x, _ in test_ds])
    y_test  = torch.tensor([y for _, y in test_ds])
    print(f"{dataset_name}: {len(X_train)} train, {len(X_test)} test, features=784, classes=10")
    return X_train, y_train, X_test, y_test, 784, 10


def _load_tabular(dataset_name, seed):
    try:
        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        import numpy as np
    except ImportError:
        raise ImportError("scikit-learn required for tabular datasets: pip install scikit-learn")

    print(f"Fetching {dataset_name} from OpenML...")
    data = fetch_openml(name=dataset_name, version=1, as_frame=False, parser="auto")
    X = data.data.astype("float32")
    le = LabelEncoder()
    y = le.fit_transform(data.target).astype("int64")
    num_classes = len(le.classes_)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    X_train = torch.tensor(X_tr)
    y_train = torch.tensor(y_tr, dtype=torch.long)
    X_test  = torch.tensor(X_te)
    y_test  = torch.tensor(y_te, dtype=torch.long)
    input_features = X_train.shape[1]
    print(f"{dataset_name}: {len(X_train)} train, {len(X_test)} test, "
          f"features={input_features}, classes={num_classes}")
    return X_train, y_train, X_test, y_test, input_features, num_classes


def load_data(dataset, seed):
    if dataset in ("mnist", "fashion_mnist"):
        return _load_vision(dataset, seed)
    elif dataset in TABULAR_DATASETS:
        return _load_tabular(dataset, seed)
    else:
        # Fallback: fake data (useful for quick testing)
        print(f"Unknown dataset {dataset!r}; using fake data")
        return _fake_data()


# ---------------------------------------------------------------------------
# Training for a single lambda
# ---------------------------------------------------------------------------

def train_one(
    area_lambda, lut_n, X_train, y_train, X_test, y_test,
    input_features, num_classes, args, device
):
    """Train one DWN model for `search_epochs` and return (area_luts, best_acc)."""
    torch.manual_seed(args.seed)

    model = DWNModel(
        input_features=input_features,
        num_classes=num_classes,
        num_bits=args.num_bits,
        hidden_sizes=args.hidden_sizes,
        lut_n=lut_n,
        mapping_first=args.mapping_first,
        mapping_rest="random",
        tau=args.tau,
        lambda_reg=0.0,
    )

    model.fit_thermometer(X_train.cpu(), verbose=False)
    model = model.to(device)

    Xtr = X_train.to(device)
    ytr = y_train.to(device)
    Xte = X_test.to(device)
    yte = y_test.to(device)

    loader    = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    n_batches = len(loader)

    best_acc   = 0.0
    area_luts_final = 0

    for epoch in range(1, args.search_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for i, (xb, yb) in enumerate(loader, 1):
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            if area_lambda > 0.0:
                entropy_loss, _ = compute_area_loss(model)
                loss = loss + area_lambda * entropy_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(
                f"    lambda={area_lambda:.5f}  epoch {epoch}/{args.search_epochs}"
                f"  batch {i}/{n_batches}  loss={epoch_loss/i:.4f}",
                end="\r", flush=True,
            )

        model.eval()
        with torch.no_grad():
            preds   = model(Xte).argmax(1)
            correct = (preds == yte).sum().item()
        acc = correct / len(yte)

        _, area_luts_final = compute_area_loss(model)

        if acc > best_acc:
            best_acc = acc

        print(
            f"    lambda={area_lambda:.5f}  epoch {epoch}/{args.search_epochs}"
            f"  acc={acc:.4f}  best={best_acc:.4f}  area_luts={area_luts_final:,}    "
        )

    return area_luts_final, best_acc


# ---------------------------------------------------------------------------
# Pareto computation
# ---------------------------------------------------------------------------

def compute_pareto_front(results):
    """
    Mark Pareto-optimal configs: not dominated on BOTH area_luts (lower is better)
    AND accuracy (higher is better).

    A config A dominates B iff A.area_luts <= B.area_luts AND A.acc >= B.acc
    with at least one strict inequality.
    """
    pareto = []
    for i, ri in enumerate(results):
        dominated = False
        for j, rj in enumerate(results):
            if i == j:
                continue
            # rj dominates ri?
            if (rj["area_luts"] <= ri["area_luts"] and rj["accuracy"] >= ri["accuracy"]
                    and (rj["area_luts"] < ri["area_luts"] or rj["accuracy"] > ri["accuracy"])):
                dominated = True
                break
        pareto.append(not dominated)
    return pareto


# ---------------------------------------------------------------------------
# ASCII scatter plot
# ---------------------------------------------------------------------------

def ascii_scatter(results, pareto_flags, width=60, height=12):
    """Print a simple ASCII scatter plot: area_luts on x-axis, accuracy on y-axis."""
    accs      = [r["accuracy"]  for r in results]
    areas     = [r["area_luts"] for r in results]

    min_area  = min(areas)
    max_area  = max(areas)
    min_acc   = max(0.0, min(accs) - 0.05)
    max_acc   = min(1.0, max(accs) + 0.05)

    # Avoid degenerate range
    area_range = max_area - min_area if max_area != min_area else 1
    acc_range  = max_acc  - min_acc  if max_acc  != min_acc  else 0.01

    grid = [[" "] * width for _ in range(height)]

    for r, is_pareto in zip(results, pareto_flags):
        col = int((r["area_luts"] - min_area) / area_range * (width - 1))
        row = height - 1 - int((r["accuracy"] - min_acc) / acc_range * (height - 1))
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))
        grid[row][col] = "*" if is_pareto else "."

    print()
    print("Area-Accuracy Pareto Front")
    print("Accuracy")
    for i, row in enumerate(grid):
        acc_label = max_acc - i / (height - 1) * acc_range
        print(f"  {acc_label:.2f} |{''.join(row)}")

    # x-axis
    print("       +" + "-" * width)
    # x-axis labels
    n_ticks = 4
    tick_positions = [int(k / (n_ticks - 1) * (width - 1)) for k in range(n_ticks)]
    tick_values    = [min_area + k / (n_ticks - 1) * area_range for k in range(n_ticks)]
    label_row = " " * 8
    for pos, val in zip(tick_positions, tick_values):
        label = f"{int(val):,}"
        label_row += label.ljust(width // n_ticks)
    print(label_row.rstrip())
    print("       (area_luts ->)")
    print("* = Pareto optimal   . = dominated")
    print()


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_table(results, pareto_flags):
    print("=== Pareto Sweep Results ===")
    print(f"{'area_lambda':>12} | {'area_luts':>10} | {'accuracy':>8} | {'pareto':>6}")
    print("-" * 12 + "-+-" + "-" * 10 + "-+-" + "-" * 8 + "-+-" + "-" * 6)
    for r, is_pareto in zip(results, pareto_flags):
        star = "*" if is_pareto else ""
        print(
            f"  {r['area_lambda']:>10.5f}  |  {r['area_luts']:>9,}  |  {r['accuracy']:>6.4f}  |  {star:>5}"
        )
    n_pareto = sum(pareto_flags)
    print(f"\nPareto front: {n_pareto} config{'s' if n_pareto != 1 else ''}")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _output_dir():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../../../mase_output/dwn")
    )


def save_csv(results, pareto_flags, lut_n_str):
    out_dir = _output_dir()
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "pareto_sweep_results.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["area_lambda", "lut_n", "area_luts", "accuracy", "is_pareto"]
        )
        writer.writeheader()
        for r, is_pareto in zip(results, pareto_flags):
            writer.writerow({
                "area_lambda": r["area_lambda"],
                "lut_n":       lut_n_str,
                "area_luts":   r["area_luts"],
                "accuracy":    f"{r['accuracy']:.6f}",
                "is_pareto":   int(is_pareto),
            })

    print(f"Results saved to: {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="DWN area-accuracy Pareto sweep over area_lambda",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="+", default=[2000, 1000],
        help="LUT layer output widths (one per layer)",
    )
    parser.add_argument(
        "--lut-n", type=str, default="6",
        help="LUT fan-in: single int (e.g. 6) or comma-separated per-layer (e.g. 6,4,2)",
    )
    parser.add_argument(
        "--dataset", type=str, default="mnist",
        help="Dataset: mnist, fashion_mnist, or a tabular dataset name",
    )
    parser.add_argument(
        "--search-epochs", type=int, default=15,
        help="Training epochs per lambda value",
    )
    parser.add_argument("--num-bits",      type=int,   default=3)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument(
        "--lambdas", type=float, nargs="+",
        default=[0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        help="area_lambda values to sweep",
    )
    parser.add_argument("--batch-size",    type=int,   default=32)
    parser.add_argument(
        "--mapping-first", type=str, default="learnable",
        choices=["learnable", "random", "arange"],
    )
    parser.add_argument(
        "--tau", type=float, default=1.0 / 0.3,
        help="GroupSum temperature (paper uses 1/0.3 ≈ 3.33)",
    )
    parser.add_argument("--lr", type=float, default=0.01)
    return parser.parse_args()


def _parse_lut_n(lut_n_str):
    parts = [int(x.strip()) for x in lut_n_str.split(",")]
    return parts[0] if len(parts) == 1 else parts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    lut_n     = _parse_lut_n(args.lut_n)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = str(device)
    if device.type == "cuda":
        device_str += f" ({torch.cuda.get_device_name(0)})"
    print(f"Device: {device_str}")

    print("\n=== DWN Pareto Sweep ===")
    print(f"  dataset={args.dataset}, search_epochs={args.search_epochs}")
    print(f"  hidden_sizes={args.hidden_sizes}, lut_n={args.lut_n}, num_bits={args.num_bits}")
    print(f"  tau={args.tau:.3f}, lr={args.lr}, batch_size={args.batch_size}")
    print(f"  lambdas={args.lambdas}")
    print(f"  mapping_first={args.mapping_first}, seed={args.seed}")
    print()

    # Load data once; reuse across lambdas
    X_train, y_train, X_test, y_test, input_features, num_classes = load_data(
        args.dataset, args.seed
    )

    results = []

    for idx, area_lambda in enumerate(args.lambdas):
        print(f"\n[{idx + 1}/{len(args.lambdas)}] area_lambda={area_lambda:.5f}")
        print("-" * 55)

        area_luts, best_acc = train_one(
            area_lambda=area_lambda,
            lut_n=lut_n,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            input_features=input_features,
            num_classes=num_classes,
            args=args,
            device=device,
        )

        results.append({
            "area_lambda": area_lambda,
            "area_luts":   area_luts,
            "accuracy":    best_acc,
        })
        print(f"  -> area_luts={area_luts:,}  best_acc={best_acc:.4f}")

    # Compute Pareto front
    pareto_flags = compute_pareto_front(results)

    # Print table
    print()
    print_table(results, pareto_flags)

    # ASCII scatter
    ascii_scatter(results, pareto_flags)

    # Save CSV
    csv_path = save_csv(results, pareto_flags, args.lut_n)

    return 0


if __name__ == "__main__":
    sys.exit(main())
