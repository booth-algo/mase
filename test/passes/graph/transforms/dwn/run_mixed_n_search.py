#!/usr/bin/env python3
"""
Mixed-N search for DWN models: sweep per-layer LUT fan-in configurations
to find accuracy/area Pareto trade-offs.

[Example usage]
    python run_mixed_n_search.py --hidden-sizes 2000 1000 --search-epochs 10 \
        --n-values 2 4 6 --max-configs 27

    # Fewer configs for quick testing
    python run_mixed_n_search.py --hidden-sizes 100 --search-epochs 5 \
        --n-values 2 4 --max-configs 9
"""
import sys
import os
import types
import argparse
import itertools
import random
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
# Inline copy of compute_area_loss from run_dwn_training.py
# ---------------------------------------------------------------------------

def compute_area_loss(model):
    """
    Hardware-aware regularization for DWN.

    Returns:
        (entropy_loss: Tensor, area_luts: int)
    """
    from chop.nn.dwn.mapping import LearnableMapping

    entropy_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    area_luts = 0

    for layer in model.lut_layers:
        area_luts += layer.output_size * (2 ** layer.n)

        if isinstance(layer.mapping, LearnableMapping):
            W = layer.mapping.weights   # (input_size, output_size * n)
            tau = getattr(layer.mapping, 'tau', 0.001)
            probs = torch.softmax(W / tau, dim=0)
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum(dim=0).mean()
            entropy_loss = entropy_loss + entropy

    return entropy_loss, area_luts


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _fake_data(n_train=640):
    print(f"Using fake random data ({n_train} train, 200 test, features=784, classes=10)")
    X_train = torch.randn(n_train, 784)
    y_train = torch.randint(0, 10, (n_train,))
    X_test  = torch.randn(200, 784)
    y_test  = torch.randint(0, 10, (200,))
    return X_train, y_train, X_test, y_test, 784, 10


def _load_vision(dataset_name, n_train=None):
    try:
        from torchvision import datasets as tvdatasets, transforms
    except ImportError:
        print("torchvision not available; falling back to fake data")
        return _fake_data(n_train or 640)

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


TABULAR_DATASETS = [
    "phoneme", "skin-seg", "higgs", "australian", "nomao",
    "segment", "miniboone", "christine", "jasmine", "sylvine", "blood",
]


def load_data(args):
    """Return (X_train, y_train, X_test, y_test, input_features, num_classes)."""
    dataset = args.dataset
    if dataset in ("mnist", "fashion_mnist"):
        # default to fake if --dataset mnist but no --real-data flag
        if not args.real_data:
            return _fake_data(args.n_train)
        return _load_vision(dataset, args.n_train)
    elif dataset in TABULAR_DATASETS:
        try:
            from sklearn.datasets import fetch_openml
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            import numpy as np
        except ImportError:
            raise ImportError("scikit-learn required for tabular datasets: pip install scikit-learn")
        print(f"Fetching {dataset} from OpenML...")
        data = fetch_openml(name=dataset, version=1, as_frame=False, parser="auto")
        X = data.data.astype("float32")
        le = LabelEncoder()
        y = le.fit_transform(data.target).astype("int64")
        num_classes = len(le.classes_)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=args.seed, stratify=y
        )
        return (
            torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.long),
            torch.tensor(X_te), torch.tensor(y_te, dtype=torch.long),
            X_tr.shape[1], num_classes,
        )
    else:
        raise ValueError(
            f"Unknown dataset: {dataset!r}. Choose from: mnist, fashion_mnist, "
            + ", ".join(TABULAR_DATASETS)
        )


# ---------------------------------------------------------------------------
# Pareto helpers
# ---------------------------------------------------------------------------

def compute_pareto_front(results):
    """
    Mark each result as Pareto-optimal.

    A config P is Pareto-optimal if no other config Q has:
        area_luts_Q <= area_luts_P  AND  acc_Q >= acc_P
    with at least one strict inequality.

    Args:
        results: list of dicts with keys 'area_luts' and 'accuracy'

    Returns:
        list of bools, same length as results
    """
    n = len(results)
    pareto = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            area_j = results[j]['area_luts']
            acc_j  = results[j]['accuracy']
            area_i = results[i]['area_luts']
            acc_i  = results[i]['accuracy']
            # Q=j dominates P=i?
            if area_j <= area_i and acc_j >= acc_i:
                if area_j < area_i or acc_j > acc_i:
                    pareto[i] = False
                    break
    return pareto


# ---------------------------------------------------------------------------
# Training a single config
# ---------------------------------------------------------------------------

def train_config(lut_n_list, input_features, num_classes, args,
                 X_train, y_train, X_test, y_test, device):
    """Train a DWNModel with the given per-layer lut_n list.

    Returns (best_acc: float, area_luts: int).
    """
    model = DWNModel(
        input_features=input_features,
        num_classes=num_classes,
        num_bits=args.num_bits,
        hidden_sizes=args.hidden_sizes,
        lut_n=lut_n_list,
        mapping_first=args.mapping_first,
        mapping_rest="random",
        tau=1.0 / 0.3,
        lambda_reg=0.0,
    )

    model.fit_thermometer(X_train, verbose=False)
    model = model.to(device)

    X_tr = X_train.to(device)
    y_tr = y_train.to(device)
    X_te = X_test.to(device)
    y_te = y_test.to(device)

    dataset  = TensorDataset(X_tr, y_tr)
    loader   = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    n_batches = len(loader)

    for epoch in range(1, args.search_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for i, (x_batch, y_batch) in enumerate(loader, 1):
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(
                f"    epoch {epoch}/{args.search_epochs}  "
                f"batch {i}/{n_batches}  "
                f"loss={epoch_loss/i:.4f}",
                end="\r", flush=True,
            )

        model.eval()
        with torch.no_grad():
            correct = (model(X_te).argmax(1) == y_te).sum().item()
        acc = correct / len(y_te)
        if acc > best_acc:
            best_acc = acc
        print(
            f"    epoch {epoch}/{args.search_epochs}  "
            f"acc={acc:.4f}  best={best_acc:.4f}".ljust(60),
        )

    _, area_luts = compute_area_loss(model)
    return best_acc, area_luts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Mixed-N search for DWN: sweep per-layer LUT fan-in to find accuracy/area trade-offs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hidden-sizes",  type=int,   nargs="+", default=[2000, 1000],
                        help="LUT layer hidden widths, e.g. --hidden-sizes 2000 1000")
    parser.add_argument("--dataset",       type=str,   default="mnist",
                        help="Dataset: mnist (fake by default), fashion_mnist, or tabular dataset name")
    parser.add_argument("--real-data",     action="store_true",
                        help="Download and use real MNIST/FashionMNIST data (requires torchvision)")
    parser.add_argument("--n-train",       type=int,   default=640,
                        help="Number of training samples for fake data mode")
    parser.add_argument("--search-epochs", type=int,   default=10,
                        help="Training epochs per configuration")
    parser.add_argument("--num-bits",      type=int,   default=3,
                        help="Thermometer bits per feature")
    parser.add_argument("--seed",          type=int,   default=42,
                        help="Random seed")
    parser.add_argument("--max-configs",   type=int,   default=27,
                        help="Maximum number of configurations to evaluate (random sample if exceeded)")
    parser.add_argument("--n-values",      type=int,   nargs="+", default=[2, 4, 6],
                        help="Allowed LUT fan-in values to search over, e.g. --n-values 2 4 6")
    parser.add_argument("--batch-size",    type=int,   default=32,
                        help="Training batch size")
    parser.add_argument("--mapping-first", type=str,   default="learnable",
                        choices=["learnable", "random", "arange"],
                        help="Mapping type for first LUT layer")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (
        f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    ))

    # Number of LUT layers = len(hidden_sizes)
    num_lut_layers = len(args.hidden_sizes)
    print(f"\n=== Mixed-N Search ===")
    print(f"  hidden_sizes={args.hidden_sizes}, num_lut_layers={num_lut_layers}")
    print(f"  n_values={args.n_values}, search_epochs={args.search_epochs}")
    print(f"  max_configs={args.max_configs}, seed={args.seed}")

    # Generate all candidate configs
    all_configs = list(itertools.product(args.n_values, repeat=num_lut_layers))
    total = len(all_configs)
    print(f"  Total candidate configs: {total}  (={len(args.n_values)}^{num_lut_layers})")

    if total > args.max_configs:
        print(f"  Sampling {args.max_configs} configs randomly (seed={args.seed})")
        all_configs = random.sample(all_configs, args.max_configs)
    else:
        print(f"  Evaluating all {total} configs")

    # Load data once
    print("\nLoading data...")
    X_train, y_train, X_test, y_test, input_features, num_classes = load_data(args)
    print(f"  input_features={input_features}, num_classes={num_classes}")

    # Evaluate each config
    results = []
    for cfg_idx, lut_n_tuple in enumerate(all_configs, 1):
        lut_n_list = list(lut_n_tuple)
        cfg_str = "-".join(str(n) for n in lut_n_list)
        expected_area = sum(
            hs * (2 ** n)
            for hs, n in zip(
                args.hidden_sizes,
                lut_n_list,
            )
        )
        print(f"\n[{cfg_idx}/{len(all_configs)}] N config: {cfg_str}  "
              f"(expected area ~{expected_area:,} LUTs)")

        # Re-seed per config for reproducibility
        torch.manual_seed(args.seed + cfg_idx)

        best_acc, area_luts = train_config(
            lut_n_list=lut_n_list,
            input_features=input_features,
            num_classes=num_classes,
            args=args,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            device=device,
        )
        results.append({
            'lut_n_config': cfg_str,
            'area_luts': area_luts,
            'accuracy': best_acc,
        })
        print(f"  -> best_acc={best_acc:.4f}  area_luts={area_luts:,}")

    # Sort by area ascending
    results.sort(key=lambda r: r['area_luts'])

    # Compute Pareto front
    pareto_flags = compute_pareto_front(results)
    for r, is_p in zip(results, pareto_flags):
        r['is_pareto'] = is_p

    # Print table
    print("\n=== Mixed-N Search Results ===")
    col_w = max(len(r['lut_n_config']) for r in results) + 2
    col_w = max(col_w, 10)
    print(
        f" {'N Config':<{col_w}} | {'AreaLUTs':>10} | {'Accuracy':>8} | {'Pareto':>6}"
    )
    print("-" * (col_w + 1) + "+" + "-" * 12 + "+" + "-" * 10 + "+" + "-" * 8)
    for r in results:
        pareto_mark = "  *" if r['is_pareto'] else ""
        print(
            f" {r['lut_n_config']:<{col_w}} | {r['area_luts']:>10,} | "
            f"{r['accuracy']:>8.4f} | {pareto_mark:>6}"
        )

    pareto_count = sum(1 for r in results if r['is_pareto'])
    print(f"\nPareto front: {pareto_count} configs")

    # Save CSV
    out_dir = os.path.join(
        os.path.dirname(__file__), "../../../../../mase_output/dwn"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mixed_n_search_results.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["lut_n_config", "area_luts", "accuracy", "is_pareto"]
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                'lut_n_config': r['lut_n_config'],
                'area_luts': r['area_luts'],
                'accuracy': r['accuracy'],
                'is_pareto': r['is_pareto'],
            })

    print(f"Saved to: {os.path.abspath(out_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
