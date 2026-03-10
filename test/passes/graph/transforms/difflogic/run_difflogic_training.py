#!/usr/bin/env python3
"""
Standalone DiffLogic training runner for Pareto comparison with DWN.

DiffLogic uses 2-input differentiable logic gates (16 Boolean operations)
instead of LUTs.  No thermometer encoding — raw pixel values are used
(the relaxed gates operate on continuous [0,1] during training).

[Example usage]
MNIST (20x20 border-removed, matching the DiffLogic paper):
    python run_difflogic_training.py --dataset mnist --epochs 30 \
        --num-neurons 8000 --num-layers 6 --grad-factor 1 --tau 20 \
        --lr 0.01 --batch-size 128 --ckpt-name mnist_8k

CIFAR-10 (32x32x3 = 3072 raw features):
    python run_difflogic_training.py --dataset cifar10 --epochs 50 \
        --num-neurons 8000 --num-layers 6 --grad-factor 2 --tau 20 \
        --lr 0.01 --batch-size 128 --ckpt-name cifar10_8k

Requires the `difflogic` package:
    pip install difflogic   (needs CUDA toolkit for fast kernels)
"""
import sys
import os
import types
import argparse

# ---------------------------------------------------------------------------
# sys.path / sys.modules stubs — avoid pulling in the heavy mase __init__.py
# ---------------------------------------------------------------------------
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

try:
    from difflogic import LogicLayer, GroupSum
except ImportError:
    print("ERROR: 'difflogic' package not found.")
    print("Install with:  pip install difflogic  (requires CUDA toolkit)")
    print("See https://github.com/Felix-Petersen/difflogic")
    sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="DiffLogic training runner (Pareto comparison with DWN)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--num-neurons",  type=int,   default=8000,
                        help="Width of each hidden LogicLayer")
    parser.add_argument("--num-layers",   type=int,   default=6,
                        help="Number of LogicLayer hidden layers")
    parser.add_argument("--grad-factor",  type=float, default=1.0,
                        help="Gradient scaling factor (increase for deep nets, e.g. 2)")
    parser.add_argument("--tau",          type=float, default=20.0,
                        help="GroupSum temperature (paper uses 20 for MNIST)")
    parser.add_argument("--connections",  type=str,   default="random",
                        choices=["random", "unique"],
                        help="LogicLayer connection strategy")
    parser.add_argument("--lr",           type=float, default=0.01)
    parser.add_argument("--lr-step",      type=int,   default=14,
                        help="StepLR step size")
    parser.add_argument("--lr-gamma",     type=float, default=0.1)
    parser.add_argument("--batch-size",   type=int,   default=128)
    parser.add_argument("--dataset",      type=str,   default="mnist",
                        choices=["mnist", "cifar10", "jsc", "nid"],
                        help="Dataset to train on")
    parser.add_argument("--remove-border", action="store_true", default=True,
                        help="Remove MNIST border (28x28 -> 20x20 = 400 features, "
                             "as in the DiffLogic paper)")
    parser.add_argument("--no-remove-border", dest="remove_border", action="store_false",
                        help="Keep full 28x28 MNIST")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--patience",     type=int,   default=None,
                        help="Early stopping patience (default: disabled)")
    parser.add_argument("--eval",         action="store_true",
                        help="Load checkpoint and evaluate only")
    parser.add_argument("--ckpt",         type=str,   default=None,
                        help="Full checkpoint path (overrides --ckpt-name)")
    parser.add_argument("--ckpt-name",    type=str,   default="best",
                        help="Checkpoint filename stem (saved as <name>.pt in mase_output/difflogic/)")
    parser.add_argument("--implementation", type=str, default=None,
                        choices=["cuda", "python"],
                        help="LogicLayer implementation (default: cuda if available, else python)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(args):
    """Load dataset.  Returns (X_train, y_train, X_test, y_test, input_features, num_classes)."""
    if args.dataset == "mnist":
        return _load_mnist(args)
    elif args.dataset == "cifar10":
        return _load_cifar10(args)
    elif args.dataset == "jsc":
        return _load_jsc(args)
    elif args.dataset == "nid":
        return _load_nid(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def _load_mnist(args):
    try:
        from torchvision import datasets as tvdatasets, transforms
    except ImportError:
        raise ImportError("torchvision is required: pip install torchvision")

    if args.remove_border:
        # Use the DiffLogic paper's border-removal: 28x28 -> 20x20 = 400 features
        # We implement a simplified version here rather than importing mnist_data.py
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        cache = "~/.cache/mnist"
        train_ds = tvdatasets.MNIST(cache, train=True,  download=True, transform=transform)
        test_ds  = tvdatasets.MNIST(cache, train=False, download=True, transform=transform)

        def remove_border_tensor(img_tensor):
            """Remove 4 pixels from each border: 28x28 -> 20x20."""
            return img_tensor[:, 4:24, 4:24]

        X_train = torch.stack([remove_border_tensor(x).view(-1) for x, _ in train_ds])
        y_train = torch.tensor([y for _, y in train_ds])
        X_test  = torch.stack([remove_border_tensor(x).view(-1) for x, _ in test_ds])
        y_test  = torch.tensor([y for _, y in test_ds])
        input_features = 400
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
        cache = "~/.cache/mnist"
        train_ds = tvdatasets.MNIST(cache, train=True,  download=True, transform=transform)
        test_ds  = tvdatasets.MNIST(cache, train=False, download=True, transform=transform)
        X_train = torch.stack([x for x, _ in train_ds])
        y_train = torch.tensor([y for _, y in train_ds])
        X_test  = torch.stack([x for x, _ in test_ds])
        y_test  = torch.tensor([y for _, y in test_ds])
        input_features = 784

    print(f"mnist: {len(X_train)} train, {len(X_test)} test, features={input_features}, classes=10")
    return X_train, y_train, X_test, y_test, input_features, 10


def _load_cifar10(args):
    try:
        from torchvision import datasets as tvdatasets, transforms
    except ImportError:
        raise ImportError("torchvision is required: pip install torchvision")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    cache = "~/.cache/cifar10"
    train_ds = tvdatasets.CIFAR10(cache, train=True,  download=True, transform=transform)
    test_ds  = tvdatasets.CIFAR10(cache, train=False, download=True, transform=transform)
    X_train = torch.stack([x for x, _ in train_ds])
    y_train = torch.tensor([y for _, y in train_ds])
    X_test  = torch.stack([x for x, _ in test_ds])
    y_test  = torch.tensor([y for _, y in test_ds])
    input_features = 3072

    print(f"cifar10: {len(X_train)} train, {len(X_test)} test, features={input_features}, classes=10")
    return X_train, y_train, X_test, y_test, input_features, 10


def _load_jsc(args):
    """Load hls4ml jet substructure classification (JSC) dataset from OpenML.

    Fetches 'hls4ml_lhc_jets_hlf' (version 1) via scikit-learn's fetch_openml.
    Features are already continuous; MinMaxScaler maps them to [0, 1] as
    required by DiffLogic's continuous-relaxation gates.
    Splits 80/20 stratified.  Typical shape: ~830k samples, 16 features, 5 classes.
    """
    try:
        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, MinMaxScaler
        import numpy as np
    except ImportError:
        raise ImportError("scikit-learn is required for JSC: pip install scikit-learn")

    print("Fetching hls4ml_lhc_jets_hlf from OpenML...")
    data = fetch_openml(name="hls4ml_lhc_jets_hlf", version=1, as_frame=False, parser="auto")
    X = data.data.astype(np.float32)
    y_raw = data.target

    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(np.int64)
    num_classes = len(le.classes_)

    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    # Normalize to [0, 1] — required for DiffLogic continuous gates
    scaler = MinMaxScaler()
    X_train_np = scaler.fit_transform(X_train_np).astype(np.float32)
    X_test_np  = scaler.transform(X_test_np).astype(np.float32)

    X_train = torch.tensor(X_train_np)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    X_test  = torch.tensor(X_test_np)
    y_test  = torch.tensor(y_test_np, dtype=torch.long)

    input_features = X_train.shape[1]
    print(f"jsc: {len(X_train)} train, {len(X_test)} test, "
          f"features={input_features}, classes={num_classes} {list(le.classes_)}")
    return X_train, y_train, X_test, y_test, input_features, num_classes


def _load_nid(args):
    """Load NSL-KDD Network Intrusion Detection dataset.

    Downloads KDDTrain+.txt and KDDTest+.txt from GitHub if not cached.
    Categorical columns (protocol_type, service, flag) are one-hot encoded.
    Labels are mapped to 5 coarse classes: normal, dos, probe, r2l, u2r.
    All features are MinMax-scaled to [0, 1] for DiffLogic.
    Typical shape after encoding: ~125k train / ~22k test, ~122 features, 5 classes.
    """
    import numpy as np
    import os
    import urllib.request

    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for NID: pip install pandas")
    try:
        from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    except ImportError:
        raise ImportError("scikit-learn is required for NID: pip install scikit-learn")

    cache_dir = os.path.expanduser("~/.cache/nsl-kdd")
    os.makedirs(cache_dir, exist_ok=True)

    train_url  = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
    test_url   = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"
    train_path = os.path.join(cache_dir, "KDDTrain+.txt")
    test_path  = os.path.join(cache_dir, "KDDTest+.txt")

    for url, path in [(train_url, train_path), (test_url, test_path)]:
        if not os.path.exists(path):
            print(f"Downloading NSL-KDD from {url}...")
            urllib.request.urlretrieve(url, path)

    col_names = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
        "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "class", "difficulty",
    ]

    df_train = pd.read_csv(train_path, header=None, names=col_names)
    df_test  = pd.read_csv(test_path,  header=None, names=col_names)

    df_train = df_train.drop("difficulty", axis=1)
    df_test  = df_test.drop("difficulty", axis=1)

    # One-hot encode categorical columns jointly so both splits share the same schema
    cat_cols = ["protocol_type", "service", "flag"]
    df_all  = pd.concat([df_train, df_test], axis=0)
    df_all  = pd.get_dummies(df_all, columns=cat_cols)
    df_train2 = df_all.iloc[:len(df_train)]
    df_test2  = df_all.iloc[len(df_train):]

    attack_map = {
        "normal": "normal",
        "neptune": "dos", "back": "dos", "land": "dos", "pod": "dos",
        "smurf": "dos", "teardrop": "dos", "mailbomb": "dos", "apache2": "dos",
        "processtable": "dos", "udpstorm": "dos",
        "ipsweep": "probe", "nmap": "probe", "portsweep": "probe", "satan": "probe",
        "mscan": "probe", "saint": "probe",
        "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l", "multihop": "r2l",
        "phf": "r2l", "spy": "r2l", "warezclient": "r2l", "warezmaster": "r2l",
        "sendmail": "r2l", "named": "r2l", "snmpgetattack": "r2l", "snmpguess": "r2l",
        "xlock": "r2l", "xsnoop": "r2l", "httptunnel": "r2l",
        "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r", "rootkit": "u2r",
        "ps": "u2r", "sqlattack": "u2r", "xterm": "u2r",
    }

    y_train_raw = df_train2["class"].map(lambda x: attack_map.get(x, "other"))
    y_test_raw  = df_test2["class"].map(lambda x: attack_map.get(x, "other"))

    X_train_np = df_train2.drop("class", axis=1).values.astype(np.float32)
    X_test_np  = df_test2.drop("class", axis=1).values.astype(np.float32)

    le = LabelEncoder()
    le.fit(pd.concat([y_train_raw, y_test_raw]))
    y_train_np = le.transform(y_train_raw).astype(np.int64)
    y_test_np  = le.transform(y_test_raw).astype(np.int64)

    # Normalize to [0, 1] — required for DiffLogic continuous gates
    scaler = MinMaxScaler()
    X_train_np = scaler.fit_transform(X_train_np).astype(np.float32)
    X_test_np  = scaler.transform(X_test_np).astype(np.float32)

    X_train = torch.tensor(X_train_np)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    X_test  = torch.tensor(X_test_np)
    y_test  = torch.tensor(y_test_np, dtype=torch.long)

    num_classes = len(le.classes_)
    input_features = X_train.shape[1]
    print(f"nid (NSL-KDD): {len(X_train)} train, {len(X_test)} test, "
          f"features={input_features}, classes={num_classes} {list(le.classes_)}")
    return X_train, y_train, X_test, y_test, input_features, num_classes


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_model(input_features, num_classes, args, device):
    """Build a DiffLogic model: Flatten + N x LogicLayer + GroupSum."""
    impl = args.implementation
    if impl is None:
        impl = 'cuda' if device.type == 'cuda' else 'python'

    dev_str = 'cuda' if device.type == 'cuda' else 'cpu'
    num_neurons = args.num_neurons

    # The last LogicLayer output must be divisible by num_classes for GroupSum
    if num_neurons % num_classes != 0:
        closest = round(num_neurons / num_classes) * num_classes
        raise ValueError(
            f"num_neurons ({num_neurons}) must be divisible by num_classes "
            f"({num_classes}) for GroupSum.  Try --num-neurons {closest}"
        )

    layers = [nn.Flatten()]

    # First LogicLayer: input_features -> num_neurons
    layers.append(LogicLayer(
        input_features, num_neurons,
        device=dev_str, grad_factor=args.grad_factor,
        implementation=impl, connections=args.connections,
    ))

    # Hidden LogicLayers: num_neurons -> num_neurons
    for _ in range(args.num_layers - 1):
        layers.append(LogicLayer(
            num_neurons, num_neurons,
            device=dev_str, grad_factor=args.grad_factor,
            implementation=impl, connections=args.connections,
        ))

    # GroupSum output
    layers.append(GroupSum(k=num_classes, tau=args.tau, device=dev_str))

    model = nn.Sequential(*layers)
    return model


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def _ckpt_dir():
    return os.path.join(os.path.dirname(__file__), "../../../../../mase_output/difflogic")


def _model_config(args, input_features, num_classes):
    """Serialisable config dict for checkpoint."""
    return {
        "input_features": input_features,
        "num_classes": num_classes,
        "num_neurons": args.num_neurons,
        "num_layers": args.num_layers,
        "grad_factor": args.grad_factor,
        "tau": args.tau,
        "connections": args.connections,
        "implementation": args.implementation,
        "dataset": args.dataset,
        "remove_border": getattr(args, "remove_border", True),
    }


# ---------------------------------------------------------------------------
# Eval mode
# ---------------------------------------------------------------------------
def eval_checkpoint(args, device):
    ckpt_path = args.ckpt or os.path.join(_ckpt_dir(), f"{args.ckpt_name}.pt")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return 1

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    cfg = ckpt.get("model_config")
    if cfg is None:
        print("Checkpoint has no model_config — using CLI args")
        cfg = _model_config(args, 400 if args.dataset == "mnist" else 3072, 10)
    else:
        print(f"  Config: num_neurons={cfg['num_neurons']}, num_layers={cfg['num_layers']}, "
              f"tau={cfg['tau']}, connections={cfg['connections']}")

    # Temporarily override args from config
    for key in ["num_neurons", "num_layers", "grad_factor", "tau", "connections", "implementation"]:
        if key in cfg:
            setattr(args, key.replace("-", "_"), cfg[key])
    if "remove_border" in cfg:
        args.remove_border = cfg["remove_border"]

    X_train, y_train, X_test, y_test, input_features, num_classes = load_data(args)
    model = build_model(input_features, num_classes, args, device)
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
    return 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    if args.eval:
        return eval_checkpoint(args, device)

    print("=== DiffLogic Training ===")
    print(f"  dataset={args.dataset}, epochs={args.epochs}, num_neurons={args.num_neurons}, "
          f"num_layers={args.num_layers}, grad_factor={args.grad_factor}")
    print(f"  tau={args.tau:.2f}, lr={args.lr}, batch={args.batch_size}, "
          f"connections={args.connections}")

    X_train, y_train, X_test, y_test, input_features, num_classes = load_data(args)

    model = build_model(input_features, num_classes, args, device)
    num_params = sum(p.numel() for p in model.parameters())
    num_gates = sum(l.out_dim for l in model if isinstance(l, LogicLayer))
    print(f"Parameters: {num_params:,}  Logic gates: {num_gates:,}")

    model = model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test,  y_test  = X_test.to(device),  y_test.to(device)

    dataset  = TensorDataset(X_train, y_train)
    loader   = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = _ckpt_dir()
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{args.ckpt_name}.pt")

    best_acc = 0.0
    epochs_no_improve = 0
    n_batches = len(loader)
    print(f"\n{'Epoch':>6}  {'Loss':>8}  {'Acc':>7}  {'Best':>7}  {'LR':>8}  {'Gates':>10}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for i, (x_batch, y_batch) in enumerate(loader, 1):
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"  Epoch {epoch}/{args.epochs}  batch {i}/{n_batches}  loss={epoch_loss/i:.4f}",
                  end="\r", flush=True)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        model.eval()
        with torch.no_grad():
            correct = (model(X_test).argmax(1) == y_test).sum().item()
        acc = correct / len(y_test)
        avg_loss = epoch_loss / n_batches

        if acc > best_acc:
            best_acc = acc
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "acc": acc,
                "loss": avg_loss,
                "model_config": _model_config(args, input_features, num_classes),
            }, ckpt_path)
        else:
            epochs_no_improve += 1

        saved = "  <-- saved" if acc >= best_acc else ""
        print(f"\r{epoch:>6}  {avg_loss:>8.4f}  {acc:>7.4f}  {best_acc:>7.4f}  {current_lr:>8.2e}  "
              f"{num_gates:>10,}{saved}".ljust(70))

        if args.patience and epochs_no_improve >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs.")
            break

    print(f"\nDone. Best acc: {best_acc:.4f}  Checkpoint: {os.path.abspath(ckpt_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
