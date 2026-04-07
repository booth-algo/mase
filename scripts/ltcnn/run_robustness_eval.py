#!/usr/bin/env python3
"""
LTCNN robustness evaluation: S&P noise and rectangular occlusion.

Results are saved as JSON to ``mase_output/ltcnn/robustness_<ckpt-stem>.json``.

Example::

    python run_robustness_eval.py --ckpt mase_output/ltcnn/ltcnn-mnist.pt \
        --dataset mnist --num-trials 5
"""

import sys
import os
import argparse
import json
import math
import random

# Add src to path
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.insert(0, _src)

import torch
import numpy as np

try:
    from scipy.optimize import curve_fit
    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_AVAILABLE = False


_DATA_DIR = os.environ.get("LTCNN_DATA_DIR", os.path.expanduser("~/.cache/ltcnn"))


def _load_mnist_torchvision(dataset_name: str):
    """Return (test_images, test_labels) as float tensors in [0, 1], shape (N, C, H, W)."""
    try:
        from torchvision import datasets as tvd, transforms
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required for MNIST/FashionMNIST loading. "
            "Install it with: pip install torchvision"
        ) from exc

    transform = transforms.ToTensor()
    cache = os.path.join(_DATA_DIR, dataset_name)
    os.makedirs(cache, exist_ok=True)

    cls_map = {
        "mnist": tvd.MNIST,
        "fashion_mnist": tvd.FashionMNIST,
    }
    ds_cls = cls_map[dataset_name]
    ds = ds_cls(cache, train=False, download=True, transform=transform)

    images = torch.stack([img for img, _ in ds])          # (N, 1, 28, 28)
    labels = torch.tensor([lbl for _, lbl in ds])         # (N,)
    return images, labels


def _load_cifar10_torchvision():
    """Return (test_images, test_labels) for CIFAR-10 as (N, 3, 32, 32) float tensors."""
    try:
        from torchvision import datasets as tvd, transforms
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required for CIFAR-10 loading. "
            "Install it with: pip install torchvision"
        ) from exc

    transform = transforms.ToTensor()
    cache = os.path.join(_DATA_DIR, "cifar10")
    os.makedirs(cache, exist_ok=True)
    ds = tvd.CIFAR10(cache, train=False, download=True, transform=transform)

    images = torch.stack([img for img, _ in ds])          # (N, 3, 32, 32)
    labels = torch.tensor([lbl for _, lbl in ds])         # (N,)
    return images, labels


def load_test_data(dataset: str):
    """Load the test split for the requested dataset.

    Args:
        dataset: one of ``mnist``, ``fashion_mnist``, ``cifar10``.

    Returns:
        images: shape ``(N, C, H, W)``, dtype float32, values in ``[0, 1]``.
        labels: shape ``(N,)``, dtype long.
    """
    if dataset in ("mnist", "fashion_mnist"):
        return _load_mnist_torchvision(dataset)
    elif dataset == "cifar10":
        return _load_cifar10_torchvision()
    else:
        raise ValueError(
            f"Unsupported dataset {dataset!r}. "
            "Choose from: mnist, fashion_mnist, cifar10"
        )


def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
    """Load an LTCNN from a ``.pt`` checkpoint file.

    The checkpoint must contain ``model_config`` (LTCNN constructor kwargs)
    and ``model_state_dict``.

    Returns:
        model: restored LTCNN in ``eval()`` mode.
        ckpt: raw checkpoint dict.
    """
    from chop.nn.ltcnn.model import LTCNN  # noqa: PLC0415

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    cfg = ckpt.get("model_config")
    if cfg is None:
        raise KeyError(
            "Checkpoint does not contain 'model_config'. "
            "Re-train with the LTCNN training script which saves this key."
        )

    print(
        f"  Checkpoint info: epoch={ckpt.get('epoch', '?')}, "
        f"saved_accuracy={ckpt.get('acc', float('nan')):.4f}"
    )
    print(f"  Model config: {cfg}")

    _LTCNN_INIT_KEYS = {
        "in_channels", "num_classes", "image_size", "bit_depth", "encoding",
        "n", "conv_channels", "kernel_size", "ff_hidden_sizes", "tau",
        "learnable_mapping", "Q",
    }
    model = LTCNN(**{k: v for k, v in cfg.items() if k in _LTCNN_INIT_KEYS})
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, ckpt


def inject_salt_and_pepper(images: torch.Tensor, noise_fraction: float) -> torch.Tensor:
    """Apply salt-and-pepper noise by randomly setting pixels to 0 or 1.

    Args:
        images: shape ``(N, C, H, W)``, values in ``[0, 1]``.
        noise_fraction: fraction of pixels to corrupt, in ``[0, 1]``.

    Returns:
        Noisy images tensor, same shape and dtype as ``images``.
    """
    if noise_fraction <= 0.0:
        return images.clone()

    noisy = images.clone()
    mask = torch.rand_like(noisy) < noise_fraction
    salt = (torch.rand_like(noisy) < 0.5).float()
    noisy[mask] = salt[mask]
    return noisy


def inject_occlusion(images: torch.Tensor, rect_size: int) -> torch.Tensor:
    """Place a black rectangle at a uniformly random position in every image.

    Args:
        images: shape ``(N, C, H, W)``, values in ``[0, 1]``.
        rect_size: side length (pixels) of the square occluding rectangle.

    Returns:
        Occluded images tensor, same shape and dtype as ``images``.
    """
    if rect_size <= 0:
        return images.clone()

    N, C, H, W = images.shape
    occluded = images.clone()

    max_y = max(0, H - rect_size)
    max_x = max(0, W - rect_size)

    for i in range(N):
        top  = random.randint(0, max_y)
        left = random.randint(0, max_x)
        bottom = min(top  + rect_size, H)
        right  = min(left + rect_size, W)
        occluded[i, :, top:bottom, left:right] = 0.0

    return occluded


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 256,
    device: torch.device = None,
) -> float:
    """Evaluate model accuracy on a pre-processed batch of images.

    Args:
        model: model already on ``device`` and in ``eval()`` mode.
        images: shape ``(N, C, H, W)``.
        labels: shape ``(N,)``.
        batch_size: mini-batch size used during inference.
        device: torch device for sub-batch movement.

    Returns:
        Fraction of correctly classified samples.
    """
    correct = 0
    total   = len(labels)
    start   = 0

    while start < total:
        end   = min(start + batch_size, total)
        imgs  = images[start:end]
        lbls  = labels[start:end]
        if device is not None:
            imgs = imgs.to(device)
            lbls = lbls.to(device)

        logprobs = model(imgs)
        preds    = logprobs.argmax(dim=-1)
        correct += (preds == lbls).sum().item()
        start    = end

    return correct / total


def fit_exponential_decay(x_values: list, y_values: list):
    """Fit  y = a * exp(-β * x)  and return (a, β).

    Falls back to log-space linear regression when scipy is unavailable or
    curve_fit fails to converge.
    """
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    y_safe = np.clip(y, 1e-6, None)

    if _SCIPY_AVAILABLE:
        def _exp_model(t, a, beta):
            return a * np.exp(-beta * t)

        try:
            p0 = (y_safe[0], 1.0)
            popt, _ = curve_fit(
                _exp_model, x, y,
                p0=p0,
                bounds=([0, 0], [1.5, np.inf]),
                maxfev=10_000,
            )
            return float(popt[0]), float(popt[1])
        except RuntimeError:
            pass

    # Log-space linear regression: log(y) = log(a) - β*x
    log_y = np.log(y_safe)
    coeffs = np.polyfit(x, log_y, 1)
    beta = float(-coeffs[0])
    a    = float(math.exp(coeffs[1]))
    return a, beta


def eval_salt_and_pepper(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    noise_levels: list,
    device: torch.device,
    batch_size: int = 256,
) -> dict:
    """Sweep S&P noise fractions and return per-level accuracies.

    Args:
        model: eval-mode LTCNN on ``device``.
        images: clean test images ``(N, C, H, W)`` on CPU.
        labels: ground-truth labels ``(N,)`` on CPU.
        noise_levels: sorted list of noise fractions, e.g. ``[0.0, 0.05, ...]``.
        device: torch device for inference.
        batch_size: mini-batch size for inference.

    Returns:
        Dict with keys ``"levels"`` and ``"accuracies"``.
    """
    accuracies = []
    labels_dev = labels.to(device)

    for frac in noise_levels:
        noisy    = inject_salt_and_pepper(images, frac).to(device)
        acc      = evaluate(model, noisy, labels_dev, batch_size=batch_size)
        accuracies.append(acc)
        print(f"  noise={frac:.2f}  acc={acc:.4f}")

    return {"levels": list(noise_levels), "accuracies": accuracies}


def eval_occlusion(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    rect_sizes: list,
    num_trials: int,
    device: torch.device,
    batch_size: int = 256,
) -> dict:
    """Sweep occlusion rectangle sizes and return per-size averaged accuracies.

    Args:
        model: eval-mode LTCNN on ``device``.
        images: clean test images ``(N, C, H, W)`` on CPU.
        labels: ground-truth labels ``(N,)`` on CPU.
        rect_sizes: list of square rectangle side lengths in pixels.
        num_trials: number of random placements to average per size.
        device: torch device for inference.
        batch_size: mini-batch size for inference.

    Returns:
        Dict with keys ``"sizes"`` and ``"accuracies"``.
    """
    accuracies = []
    labels_dev = labels.to(device)

    for sz in rect_sizes:
        trial_accs = []
        for _ in range(num_trials):
            occ = inject_occlusion(images, sz).to(device)
            acc = evaluate(model, occ, labels_dev, batch_size=batch_size)
            trial_accs.append(acc)

        mean_acc = float(np.mean(trial_accs))
        accuracies.append(mean_acc)
        print(f"  size={sz}x{sz:<3}  acc={mean_acc:.4f}  "
              f"(trials={num_trials}, std={float(np.std(trial_accs)):.4f})")

    return {"sizes": list(rect_sizes), "accuracies": accuracies}


def parse_args():
    parser = argparse.ArgumentParser(
        description="LTCNN robustness evaluation — S&P noise and rectangular occlusion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="Path to the LTCNN checkpoint (.pt file).",
    )
    parser.add_argument(
        "--dataset", type=str, default="mnist",
        choices=["mnist", "fashion_mnist", "cifar10"],
        help="Dataset used for evaluation (must match the checkpoint's training dataset).",
    )
    parser.add_argument(
        "--num-trials", type=int, default=5,
        help="Number of random occlusion placements to average per rectangle size.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Mini-batch size used during inference.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible noise injection.",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(
            os.path.dirname(__file__), "../../mase_output/ltcnn"
        ),
        help="Directory to write the JSON results file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    ckpt_path = os.path.abspath(args.ckpt)
    model, ckpt = load_model_from_checkpoint(ckpt_path, device)

    print(f"\nLoading test data for: {args.dataset}")
    images, labels = load_test_data(args.dataset)
    print(f"  Test set: {len(images)} samples, shape {tuple(images.shape[1:])}")

    SP_LEVELS   = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    RECT_SIZES  = [2, 4, 6, 8, 10, 12]

    results = {
        "checkpoint": ckpt_path,
        "dataset":    args.dataset,
        "epoch":      ckpt.get("epoch"),
        "saved_accuracy": ckpt.get("acc"),
        "seed":       args.seed,
        "num_trials": args.num_trials,
    }

    print("\n# Salt & Pepper Robustness")
    sp_data = eval_salt_and_pepper(
        model, images, labels, SP_LEVELS, device, batch_size=args.batch_size
    )

    # Fit decay for non-zero noise levels only (avoids degenerate β=0)
    sp_x_fit = [lvl for lvl in sp_data["levels"] if lvl > 0]
    sp_y_fit = [acc for lvl, acc in zip(sp_data["levels"], sp_data["accuracies"]) if lvl > 0]

    sp_a, sp_beta = fit_exponential_decay(sp_x_fit, sp_y_fit)
    print(f"  β_s&p = {sp_beta:.4f}  (a={sp_a:.4f})")

    results["salt_and_pepper"] = {
        **sp_data,
        "fit_a":    sp_a,
        "fit_beta": sp_beta,
    }

    print(f"\n# Occlusion Robustness (trials={args.num_trials})")
    occ_data = eval_occlusion(
        model, images, labels, RECT_SIZES, args.num_trials,
        device, batch_size=args.batch_size
    )

    # Use rectangle area (size²) as x-axis so β has units of "per px²"
    occ_areas = [sz * sz for sz in occ_data["sizes"]]
    occ_a, occ_beta = fit_exponential_decay(occ_areas, occ_data["accuracies"])
    print(f"  β_occ = {occ_beta:.6f}  (a={occ_a:.4f},  x-axis = rect area in px²)")

    results["occlusion"] = {
        **occ_data,
        "areas":    occ_areas,
        "fit_a":    occ_a,
        "fit_beta": occ_beta,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_stem   = os.path.splitext(os.path.basename(ckpt_path))[0]
    output_path = os.path.join(args.output_dir, f"robustness_{ckpt_stem}.json")

    with open(output_path, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\nResults saved to: {output_path}")

    print(f"\n# Summary")
    print(f"  dataset    : {args.dataset}")
    print(f"  checkpoint : {ckpt_stem}")
    print(f"  clean acc  : {sp_data['accuracies'][0]:.4f}  (noise=0)")
    print(f"  β_s&p      : {sp_beta:.4f}")
    print(f"  β_occ      : {occ_beta:.6f}  (per px²)")


if __name__ == "__main__":
    main()
