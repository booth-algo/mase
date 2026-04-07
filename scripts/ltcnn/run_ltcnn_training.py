#!/usr/bin/env python3
"""
LTCNN training script - ICLR 2026 methodology.

Trains a Look-Up Table Convolutional Neural Network (LTCNN) on MNIST,
FashionMNIST, or CIFAR-10.

Usage:
    python scripts/ltcnn/run_ltcnn_training.py --dataset mnist
    python scripts/ltcnn/run_ltcnn_training.py --dataset cifar10 \
        --conv-channels 8 16 --ff-hidden-sizes 500 200 --epochs 50
    python scripts/ltcnn/run_ltcnn_training.py --dataset mnist --eval \
        --ckpt mase_output/ltcnn/ltcnn-mnist.pt
    python scripts/ltcnn/run_ltcnn_training.py --dataset mnist --baseline-cnn
"""

import sys
import os
import math
import time
import argparse

# Add src to path
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.insert(0, _src)

import types
for _pkg in ("chop", "chop.nn"):
    if _pkg not in sys.modules:
        _mod = types.ModuleType(_pkg)
        _mod.__path__ = [os.path.join(_src, *_pkg.split("."))]
        _mod.__package__ = _pkg
        sys.modules[_pkg] = _mod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from chop.nn.ltcnn import LTCNN
from chop.nn.ltcnn.metrics import compute_area_luts


# Paper reference values (ICLR 2026, Table 2)
PAPER_REFS = {
    "mnist": {
        "1e4":   (0.865, 0.777),
        "1e4.5": (0.940, 0.766),
        "1e5":   (0.950, 0.885),
        "1e5.5": (0.972, 0.825),
        "1e6":   (0.978, 0.851),
    },
    "fashion_mnist": {
        "1e4":   (0.803, 0.719),
        "1e4.5": (0.840, 0.714),
        "1e5":   (0.857, 0.798),
        "1e5.5": (0.874, 0.754),
        "1e6":   (0.877, 0.764),
    },
    "cifar10": {
        "1e4":   (0.432, 0.342),
        "1e4.5": (0.474, 0.365),
        "1e5":   (0.504, 0.416),
        "1e5.5": (0.522, 0.411),
        "1e6":   (0.489, 0.353),
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LTCNN training - ICLR 2026 methodology",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    p.add_argument(
        "--dataset", type=str, default="mnist",
        choices=["mnist", "fashion_mnist", "cifar10"],
        help="Dataset to train on.",
    )

    # Architecture
    p.add_argument(
        "--conv-channels", type=int, nargs="+", default=[4, 8],
        help="Output channels per conv layer.",
    )
    p.add_argument(
        "--ff-hidden-sizes", type=int, nargs="+", default=[500, 200],
        help="Hidden widths for feed-forward layers.",
    )
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--lut-n", type=int, default=4, help="LUT arity (fan-in).")
    p.add_argument("--bit-depth", type=int, default=2)
    p.add_argument(
        "--encoding", type=str, default="quantization",
        choices=["quantization", "thermometer"],
    )
    p.add_argument("--tau", type=float, default=None,
                   help="GroupSum temperature. Defaults to sqrt(nf/C) per paper.")
    p.add_argument("--learnable-mapping", action="store_true", default=False)
    p.add_argument("--Q", type=int, default=None, choices=[None, 1, 2, 4, 8])

    # Training hyper-parameters
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch-size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=0.02)
    p.add_argument("--lr-gamma",   type=float, default=0.95)
    p.add_argument("--patience",   type=int,   default=10)
    p.add_argument("--seed",       type=int,   default=42)

    # Checkpoint I/O
    p.add_argument("--ckpt-name", type=str, default=None,
                   help="Checkpoint filename stem (saved as mase_output/ltcnn/<name>.pt).")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Explicit checkpoint path for --eval.")

    # Modes
    p.add_argument("--eval", action="store_true",
                   help="Load a saved checkpoint and evaluate; skip training.")
    p.add_argument("--baseline-cnn", action="store_true",
                   help="Train a matched standard CNN baseline instead of LTCNN.")

    return p.parse_args()


# Dataset loading

def _data_dir() -> str:
    return os.environ.get("LTCNN_DATA_DIR", os.path.expanduser("~/.cache/ltcnn"))


def load_dataset(dataset_name: str):
    """Load an image classification dataset as image tensors.

    Returns X_train, y_train, X_test, y_test, in_channels, num_classes, image_size.
    """
    try:
        from torchvision import datasets as tvd, transforms
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required: pip install torchvision"
        ) from exc

    cache = os.path.join(_data_dir(), dataset_name)
    to_tensor = transforms.ToTensor()

    if dataset_name == "mnist":
        cls, in_channels, image_size, num_classes = tvd.MNIST, 1, 28, 10
    elif dataset_name == "fashion_mnist":
        cls, in_channels, image_size, num_classes = tvd.FashionMNIST, 1, 28, 10
    elif dataset_name == "cifar10":
        cls, in_channels, image_size, num_classes = tvd.CIFAR10, 3, 32, 10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}")

    train_ds = cls(cache, train=True,  download=True, transform=to_tensor)
    test_ds  = cls(cache, train=False, download=True, transform=to_tensor)

    print(f"Loading {dataset_name}: {len(train_ds)} train, {len(test_ds)} test")

    X_train = torch.stack([x for x, _ in train_ds])
    y_train = torch.tensor([y for _, y in train_ds], dtype=torch.long)
    X_test  = torch.stack([x for x, _ in test_ds])
    y_test  = torch.tensor([y for _, y in test_ds], dtype=torch.long)

    return X_train, y_train, X_test, y_test, in_channels, num_classes, image_size


# Baseline CNN

def build_baseline_cnn(
    in_channels: int,
    num_classes: int,
    image_size: int,
    conv_channels: list,
    kernel_size: int,
    ff_hidden_sizes: list,
) -> nn.Sequential:
    """Build a standard CNN matched to an LTCNN configuration."""
    layers = []
    ch_in = in_channels

    for ch_out in conv_channels:
        layers += [
            nn.Conv2d(ch_in, ch_out, kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        ch_in = ch_out

    spatial = image_size
    for _ in conv_channels:
        spatial = spatial // 2
    flat_size = ch_in * spatial * spatial

    layers.append(nn.Flatten())

    for hidden in ff_hidden_sizes:
        layers += [nn.Linear(flat_size, hidden), nn.ReLU(inplace=True)]
        flat_size = hidden

    layers.append(nn.Linear(flat_size, num_classes))
    return nn.Sequential(*layers)


# Training loop

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    """Run one training epoch and return the mean batch loss."""
    model.train()
    epoch_loss = 0.0
    n_batches = len(loader)

    for i, (x_batch, y_batch) in enumerate(loader, 1):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        log_probs = model(x_batch)
        loss = criterion(log_probs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        print(
            f"  Epoch {epoch}/{total_epochs}  batch {i}/{n_batches}"
            f"  loss={epoch_loss / i:.4f}",
            end="\r",
            flush=True,
        )

    return epoch_loss / n_batches


def evaluate(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> float:
    """Evaluate model accuracy on (X, y) in mini-batches."""
    model.eval()
    correct = 0
    n = len(y)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb = X[start : start + batch_size].to(device)
            yb = y[start : start + batch_size].to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()

    return correct / n


def _compute_tau(ff_hidden_sizes: list, num_classes: int) -> float:
    """Compute paper default tau = sqrt(nf / C)."""
    nf = ff_hidden_sizes[-1]
    return math.sqrt(nf / num_classes)


# Checkpoint helpers

def _ckpt_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "../../mase_output/ltcnn")


def _resolve_ckpt_path(args: argparse.Namespace) -> str:
    if args.ckpt:
        return args.ckpt
    name = args.ckpt_name or f"ltcnn-{args.dataset}"
    return os.path.join(_ckpt_dir(), f"{name}.pt")


def save_checkpoint(
    path: str,
    model: nn.Module,
    epoch: int,
    acc: float,
    loss: float,
    model_config: dict,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "acc": acc,
            "loss": loss,
            "model_config": model_config,
        },
        path,
    )


# Dataset helpers

def _dataset_in_channels(dataset: str) -> int:
    return 3 if dataset == "cifar10" else 1


def _dataset_image_size(dataset: str) -> int:
    return 32 if dataset == "cifar10" else 28


def _dataset_num_classes(dataset: str) -> int:
    _NUM_CLASSES = {"mnist": 10, "fashion_mnist": 10, "cifar10": 10}
    if dataset not in _NUM_CLASSES:
        raise ValueError(f"Unknown dataset {dataset!r}")
    return _NUM_CLASSES[dataset]


def _build_model_config(args: argparse.Namespace, tau: float) -> dict:
    return {
        "dataset": args.dataset,
        "in_channels": _dataset_in_channels(args.dataset),
        "num_classes": _dataset_num_classes(args.dataset),
        "image_size": _dataset_image_size(args.dataset),
        "bit_depth": args.bit_depth,
        "encoding": args.encoding,
        "n": args.lut_n,
        "conv_channels": args.conv_channels,
        "kernel_size": args.kernel_size,
        "ff_hidden_sizes": args.ff_hidden_sizes,
        "tau": tau,
        "learnable_mapping": args.learnable_mapping,
        "Q": args.Q,
    }


# Eval-only path

def eval_checkpoint(args: argparse.Namespace, device: torch.device) -> int:
    """Load a checkpoint and report test accuracy. Returns exit code."""
    ckpt_path = _resolve_ckpt_path(args)
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return 1

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    cfg = ckpt.get("model_config")
    if cfg is None:
        print("WARNING: checkpoint has no model_config - using CLI args.")
        nc = _dataset_num_classes(args.dataset)
        tau = _compute_tau(args.ff_hidden_sizes, nc) if args.tau is None else args.tau
        cfg = _build_model_config(args, tau)

    print(f"  Config: {cfg}")

    X_train, y_train, X_test, y_test, in_channels, num_classes, image_size = \
        load_dataset(cfg.get("dataset", args.dataset))

    if cfg.get("baseline_cnn", False):
        model = build_baseline_cnn(
            in_channels=cfg["in_channels"],
            num_classes=cfg["num_classes"],
            image_size=cfg["image_size"],
            conv_channels=cfg["conv_channels"],
            kernel_size=cfg["kernel_size"],
            ff_hidden_sizes=cfg["ff_hidden_sizes"],
        )
    else:
        model = LTCNN(
            in_channels=cfg["in_channels"],
            num_classes=cfg["num_classes"],
            image_size=cfg["image_size"],
            bit_depth=cfg["bit_depth"],
            encoding=cfg["encoding"],
            n=cfg["n"],
            conv_channels=cfg["conv_channels"],
            kernel_size=cfg["kernel_size"],
            ff_hidden_sizes=cfg["ff_hidden_sizes"],
            tau=cfg["tau"],
            learnable_mapping=cfg.get("learnable_mapping", False),
            Q=cfg.get("Q"),
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    acc = evaluate(model, X_test, y_test, device)
    print(f"\n# Eval results (epoch {ckpt.get('epoch', '?')})")
    print(f"  Test accuracy : {acc:.4f}  ({int(acc * len(y_test))}/{len(y_test)})")
    print(f"  Saved val acc : {ckpt.get('acc', float('nan')):.4f}")

    if not cfg.get("baseline_cnn", False):
        lut_area = compute_area_luts(model)
        print(f"  LUT area      : {lut_area:,} entries")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters    : {param_count:,}")
    return 0


# Main training entry point

def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    print(f"Device: {device}{dev_name}")

    if args.eval:
        return eval_checkpoint(args, device)

    X_train, y_train, X_test, y_test, in_channels, num_classes, image_size = \
        load_dataset(args.dataset)

    tau = args.tau
    if tau is None and not args.baseline_cnn:
        tau = _compute_tau(args.ff_hidden_sizes, num_classes)
        print(f"Using tau = sqrt({args.ff_hidden_sizes[-1]}/{num_classes}) = {tau:.4f}")

    if args.baseline_cnn:
        model = build_baseline_cnn(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
            conv_channels=args.conv_channels,
            kernel_size=args.kernel_size,
            ff_hidden_sizes=args.ff_hidden_sizes,
        )
        criterion = nn.CrossEntropyLoss()
        model_label = "Baseline CNN"
    else:
        model = LTCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
            bit_depth=args.bit_depth,
            encoding=args.encoding,
            n=args.lut_n,
            conv_channels=args.conv_channels,
            kernel_size=args.kernel_size,
            ff_hidden_sizes=args.ff_hidden_sizes,
            tau=tau,
            learnable_mapping=args.learnable_mapping,
            Q=args.Q,
        )
        criterion = nn.NLLLoss()  # pairs with log_softmax output
        model_label = "LTCNN"

    model = model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test,  y_test  = X_test.to(device),  y_test.to(device)

    param_count = sum(p.numel() for p in model.parameters())

    ckpt_name = args.ckpt_name or f"ltcnn-{args.dataset}"
    if args.baseline_cnn:
        ckpt_name = f"baseline-cnn-{args.dataset}"

    print(f"\n# {model_label} Training")
    print(f"  dataset={args.dataset}  in_channels={in_channels}"
          f"  image_size={image_size}  num_classes={num_classes}")
    if not args.baseline_cnn:
        print(f"  encoding={args.encoding}  bit_depth={args.bit_depth}"
              f"  lut_n={args.lut_n}  tau={tau:.4f}")
    print(f"  conv_channels={args.conv_channels}  kernel_size={args.kernel_size}")
    print(f"  ff_hidden_sizes={args.ff_hidden_sizes}")
    print(f"  epochs={args.epochs}  batch_size={args.batch_size}"
          f"  lr={args.lr}  lr_gamma={args.lr_gamma}  patience={args.patience}")
    print(f"  parameters={param_count:,}")

    if not args.baseline_cnn:
        lut_area_init = compute_area_luts(model)
        print(f"  LUT area={lut_area_init:,} entries")

    loader = DataLoader(TensorDataset(X_train, y_train),
                        batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_gamma)

    ckpt_path = _resolve_ckpt_path(args)
    if not ckpt_path.endswith(".pt"):
        ckpt_path += ".pt"
    os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)

    model_config = _build_model_config(args, tau if tau is not None else 1.0)
    if args.baseline_cnn:
        model_config["baseline_cnn"] = True

    best_val_acc = 0.0
    epochs_no_improve = 0
    start_time = time.time()

    header = f"{'Epoch':>6}  {'Loss':>8}  {'TrainAcc':>9}  {'ValAcc':>8}  {'BestVal':>8}  {'LR':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(
            model, loader, optimizer, criterion, device, epoch, args.epochs
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        train_acc = evaluate(model, X_train, y_train, device)
        val_acc   = evaluate(model, X_test,  y_test,  device)

        saved_flag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            save_checkpoint(ckpt_path, model, epoch, val_acc, avg_loss, model_config)
            saved_flag = "  <-- saved"
        else:
            epochs_no_improve += 1

        print(
            f"\r{epoch:>6}  {avg_loss:>8.4f}  {train_acc:>9.4f}"
            f"  {val_acc:>8.4f}  {best_val_acc:>8.4f}  {current_lr:>8.2e}{saved_flag}"
            .ljust(80)
        )

        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs.")
            break

    elapsed = time.time() - start_time

    log_params = math.log10(param_count) if param_count > 0 else 0
    scales = [(4.0, "1e4"), (4.5, "1e4.5"), (5.0, "1e5"), (5.5, "1e5.5"), (6.0, "1e6")]
    closest_scale = min(scales, key=lambda s: abs(s[0] - log_params))
    ref_data = PAPER_REFS.get(args.dataset, {}).get(closest_scale[1])

    # Summary
    print(f"\n# Results")
    print(f"  model         : {model_label}")
    print(f"  dataset       : {args.dataset}")
    print(f"  best val acc  : {best_val_acc:.4f}")
    print(f"  parameters    : {param_count:,}  (10^{log_params:.2f})")
    print(f"  training time : {elapsed:.1f}s  ({elapsed / 60:.1f} min)")
    if not args.baseline_cnn:
        lut_area = compute_area_luts(model)
        print(f"  LUT area      : {lut_area:,} entries")
    print(f"  checkpoint    : {os.path.abspath(ckpt_path)}")

    if ref_data and not args.baseline_cnn:
        paper_max, paper_mean = ref_data
        delta = best_val_acc - paper_max
        print(f"\n# Paper comparison (ICLR 2026 Table 2, ~{closest_scale[1]} params)")
        print(f"  paper max acc : {paper_max:.3f}")
        print(f"  paper mean acc: {paper_mean:.3f}")
        print(f"  your acc      : {best_val_acc:.4f}")
        if delta >= 0:
            print(f"  delta         : +{delta:.4f}")
        else:
            print(f"  delta         : {delta:.4f}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
