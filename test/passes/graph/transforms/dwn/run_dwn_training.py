#!/usr/bin/env python3
"""
Standalone DWN training runner.

[Example usage]
Reference config (Bacellar et al., ICML 2024) — targets ~98.3% on MNIST:
    python run_dwn_training.py --real-mnist --epochs 30 --lut-n 6 \
        --hidden-sizes 2000 1000 --num-bits 3 --batch-size 32 \
        --mapping-first learnable --lr 0.01 --lr-step 14

Multi-dataset examples:
    python run_dwn_training.py --dataset fashion_mnist --epochs 30 --lut-n 6 \
        --hidden-sizes 2000 1000 --num-bits 3 --batch-size 32
    python run_dwn_training.py --dataset phoneme --epochs 20 --hidden-sizes 256 128
"""
import sys
import os
import types
import argparse

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

from datasets import load_data
from utils import compute_area_loss


def parse_args():
    parser = argparse.ArgumentParser(
        description="DWN training runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--hidden-sizes",  type=int,   nargs="+", default=[2000, 1000],
                        help="LUT layer output widths (one per layer), e.g. --hidden-sizes 2000 1000")
    parser.add_argument("--lut-n",         type=str,   default="6",
                        help="LUT fan-in: single int (e.g. 6) or comma-separated per-layer list (e.g. 6,4,2)")
    parser.add_argument("--num-bits",      type=int,   default=3,
                        help="Thermometer bits per feature (paper uses 3)")
    parser.add_argument("--tau",           type=float, default=1.0/0.3,
                        help="GroupSum temperature (paper uses 1/0.3 ≈ 3.33)")
    parser.add_argument("--lr",            type=float, default=0.01)
    parser.add_argument("--lr-step",       type=int,   default=14,
                        help="StepLR step size (paper uses 14)")
    parser.add_argument("--lr-gamma",      type=float, default=0.1)
    parser.add_argument("--lr-milestones", type=int,   nargs="*", default=[],
                        help="Epoch milestones for MultiStepLR (e.g. --lr-milestones 10 20). "
                             "If provided, uses MultiStepLR instead of StepLR.")
    parser.add_argument("--augment",       action="store_true", default=False,
                        help="Apply data augmentation for cifar10 training data "
                             "(RandomCrop 32 pad=4 + RandomHorizontalFlip).")
    parser.add_argument("--augment-refit", action="store_true", default=False,
                        help="When --augment is set, fit thermometer thresholds on "
                             "UNAUGMENTED images (matching test distribution) while still "
                             "training on augmented images. Requires --augment.")
    parser.add_argument("--batch-size",    type=int,   default=32,
                        help="Batch size (paper uses 32)")
    parser.add_argument("--lambda-reg",    type=float, default=0.0,
                        help="Spectral reg weight (0 = disabled, matches paper)")
    parser.add_argument("--area-lambda",   type=float, default=0.0,
                        help="Weight for hardware area regularization. Adds lambda * mapping_entropy_loss "
                             "to encourage LUT inputs to concentrate on fewer features (lower routing complexity). "
                             "0 = disabled. Try 1e-4 to 1e-2.")
    parser.add_argument("--optimizer",     type=str,   default="adam",
                        choices=["adam", "sgd"],
                        help="Optimizer to use: adam (default) or sgd")
    parser.add_argument("--momentum",     type=float, default=0.9,
                        help="Momentum for SGD optimizer (default: 0.9)")
    parser.add_argument("--mapping-first", default="learnable",
                        choices=["learnable", "random", "arange"])
    parser.add_argument("--dataset",       type=str,   default="mnist",
                        help="Dataset to train on: mnist, fashion_mnist, cifar10, jsc, nid, kws, toyadmos, "
                             "or tabular dataset name (phoneme, skin-seg, higgs, australian, nomao, segment, "
                             "miniboone, christine, jasmine, sylvine, blood). Default: mnist")
    parser.add_argument("--real-mnist",    action="store_true",
                        help="(Deprecated) Use --dataset mnist. Equivalent to --dataset mnist with real data.")
    parser.add_argument("--n-train",       type=int,   default=640,
                        help="Training samples for fake data mode")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--patience",      type=int,   default=None,
                        help="Early stopping patience (default: disabled)")
    parser.add_argument("--eval",          action="store_true",
                        help="Load best checkpoint and evaluate, skip training")
    parser.add_argument("--ckpt",          type=str,   default=None,
                        help="Checkpoint path for --eval (overrides --ckpt-name)")
    parser.add_argument("--ckpt-name",     type=str,   default="best",
                        help="Checkpoint filename stem (saved as <name>.pt in mase_output/dwn/)")
    return parser.parse_args()


def _ckpt_dir():
    return os.path.join(os.path.dirname(__file__), "../../../../../mase_output/dwn")


def eval_checkpoint(args, device):
    ckpt_path = args.ckpt or os.path.join(_ckpt_dir(), f"{args.ckpt_name}.pt")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return 1

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    cfg = ckpt.get("model_config")
    if cfg is None:
        print("Checkpoint has no model_config — results may be wrong")
        cfg = {
            "input_features": 784, "num_classes": 10,
            "num_bits": args.num_bits, "hidden_sizes": args.hidden_sizes,
            "lut_n": args.lut_n, "mapping_first": args.mapping_first,
            "mapping_rest": "random", "tau": args.tau, "lambda_reg": args.lambda_reg,
        }
    else:
        print(f"  Config: hidden_sizes={cfg['hidden_sizes']}, lut_n={cfg['lut_n']}, "
              f"mapping_first={cfg['mapping_first']}, num_bits={cfg['num_bits']}")

    X_train, _, X_test, y_test, _, _, _ = load_data(args)
    _DWN_VALID_KEYS = {"input_features", "num_classes", "num_bits", "hidden_sizes", "lut_n", "mapping_first", "mapping_rest", "tau", "lambda_reg"}
    cfg_filtered = {k: v for k, v in cfg.items() if k in _DWN_VALID_KEYS}
    model = DWNModel(**cfg_filtered)
    model.fit_thermometer(X_train, verbose=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(1)
        correct = (preds == y_test).sum().item()
    acc = correct / len(y_test)

    print(f"\n=== Eval results (epoch {ckpt.get('epoch', '?')}) ===")
    print(f"  Test accuracy : {acc:.4f}  ({correct}/{len(y_test)})")
    print(f"  Saved loss    : {ckpt.get('loss', float('nan')):.4f}")
    return 0


def _parse_lut_n(lut_n_str: str):
    """Parse --lut-n argument: '6' -> 6, '6,4,2' -> [6, 4, 2]."""
    parts = [int(x.strip()) for x in lut_n_str.split(",")]
    return parts[0] if len(parts) == 1 else parts


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Resolve lut_n: int or list[int]
    args.lut_n = _parse_lut_n(args.lut_n)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    if args.eval:
        return eval_checkpoint(args, device)

    print("=== DWN Training ===")
    print(f"  dataset={args.dataset}, epochs={args.epochs}, hidden_sizes={args.hidden_sizes}, lut_n={args.lut_n}, "
          f"num_bits={args.num_bits}, tau={args.tau:.3f}, lr={args.lr}, "
          f"batch={args.batch_size}, lr_step={args.lr_step}")
    print(f"  area_lambda={args.area_lambda}")

    X_train, y_train, X_test, y_test, input_features, num_classes, X_train_base = load_data(args)

    # Validate that last hidden size is divisible by num_classes for GroupSum
    if args.hidden_sizes[-1] % num_classes != 0:
        closest_multiple = round(args.hidden_sizes[-1] / num_classes) * num_classes
        raise ValueError(
            f"Last hidden size ({args.hidden_sizes[-1]}) must be divisible by num_classes "
            f"({num_classes}) for GroupSum. Try --hidden-sizes ... {closest_multiple}"
        )

    model = DWNModel(
        input_features=input_features,
        num_classes=num_classes,
        num_bits=args.num_bits,
        hidden_sizes=args.hidden_sizes,
        lut_n=args.lut_n,
        mapping_first=args.mapping_first,
        mapping_rest="random",
        tau=args.tau,
        lambda_reg=args.lambda_reg,
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("Fitting thermometer encodings...")
    thermo_data = X_train_base if X_train_base is not None else X_train
    if X_train_base is not None:
        print("  (using unaugmented X_train_base for thermometer fitting — augment-refit mode)")
    model.fit_thermometer(thermo_data, verbose=True)

    model = model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test,  y_test  = X_test.to(device),  y_test.to(device)

    dataset  = TensorDataset(X_train, y_train)
    loader   = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.lr_milestones:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = _ckpt_dir()
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{args.ckpt_name}.pt")

    best_acc = 0.0
    epochs_no_improve = 0
    n_batches = len(loader)
    print(f"\n{'Epoch':>6}  {'Loss':>8}  {'Acc':>7}  {'Best':>7}  {'LR':>8}  {'AreaLUTs':>10}")
    print("-" * 62)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for i, (x_batch, y_batch) in enumerate(loader, 1):
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            if args.lambda_reg > 0:
                loss = loss + model.get_spectral_reg_loss()
            if args.area_lambda > 0:
                entropy_loss, _ = compute_area_loss(model)
                loss = loss + args.area_lambda * entropy_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"  Epoch {epoch}/{args.epochs}  batch {i}/{n_batches}  loss={epoch_loss/i:.4f}", end="\r", flush=True)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        model.eval()
        with torch.no_grad():
            correct = (model(X_test).argmax(1) == y_test).sum().item()
        acc = correct / len(y_test)
        avg_loss = epoch_loss / n_batches
        area_luts = sum(layer.output_size * (2 ** layer.n) for layer in model.lut_layers)

        if acc > best_acc:
            best_acc = acc
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "acc": acc,
                "loss": avg_loss,
                "model_config": {
                    "input_features": input_features,
                    "num_classes": num_classes,
                    "num_bits": args.num_bits,
                    "hidden_sizes": args.hidden_sizes,
                    "lut_n": args.lut_n,
                    "mapping_first": args.mapping_first,
                    "mapping_rest": "random",
                    "tau": args.tau,
                    "lambda_reg": args.lambda_reg,
                    "area_lambda": args.area_lambda,
                },
            }, ckpt_path)
        else:
            epochs_no_improve += 1

        saved = "  <-- saved" if acc >= best_acc else ""
        print(f"\r{epoch:>6}  {avg_loss:>8.4f}  {acc:>7.4f}  {best_acc:>7.4f}  {current_lr:>8.2e}  {area_luts:>10,}{saved}".ljust(70))

        if args.patience and epochs_no_improve >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs.")
            break

    print(f"\nDone. Best acc: {best_acc:.4f}  Checkpoint: {os.path.abspath(ckpt_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
