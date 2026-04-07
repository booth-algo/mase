#!/usr/bin/env python3
"""
LTCNN NAS entry point using the MASE search framework.

Uses LTCNNSearchSpace, RunnerLTCNNTrain, and RunnerLTCNNArea registered
in the MASE search infrastructure. Requires: torch, torchvision, optuna,
toml, dill.

Usage::

    python scripts/ltcnn/run_ltcnn_nas.py \
        --config configs/ltcnn/search_ltcnn_mnist.toml \
        --save-dir mase_output/ltcnn/nas
"""

import sys
import os
import argparse
from pathlib import Path

_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.insert(0, _src)

import torch
from torch.utils.data import DataLoader


class LTCNNDataModule:
    """Lightweight data-module wrapper compatible with MASE runner expectations."""

    def __init__(self, train_ds, val_ds, batch_size: int):
        self.batch_size = batch_size
        self._train_ds = train_ds
        self._val_ds = val_ds

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=2, pin_memory=True, persistent_workers=True,
        )


class DatasetInfo:
    """Minimal dataset metadata container expected by MASE runners."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes


def _download_with_retries(cls, cache, train, transform, max_retries=3):
    """Attempt to download a torchvision dataset with retries on transient failures."""
    import shutil

    for attempt in range(1, max_retries + 1):
        try:
            return cls(cache, train=train, download=True, transform=transform)
        except RuntimeError:
            if attempt == max_retries:
                raise
            raw_dir = os.path.join(cache, cls.__name__, "raw")
            if os.path.isdir(raw_dir):
                shutil.rmtree(raw_dir)
            print(f"Download attempt {attempt} failed, retrying...")


def _load_mnist(dataset_name: str):
    from torchvision import datasets as tvds, transforms

    transform = transforms.Compose([transforms.ToTensor()])
    cache = os.path.expanduser(f"~/.cache/{dataset_name}")
    cls = tvds.MNIST if dataset_name == "mnist" else tvds.FashionMNIST
    cls.mirrors = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "http://yann.lecun.com/exdb/mnist/",
    ]
    train_ds = _download_with_retries(cls, cache, True, transform)
    test_ds = _download_with_retries(cls, cache, False, transform)
    return train_ds, test_ds, 10


def _load_cifar10():
    from torchvision import datasets as tvds, transforms

    transform = transforms.Compose([transforms.ToTensor()])
    cache = os.path.expanduser("~/.cache/cifar10")
    train_ds = tvds.CIFAR10(cache, train=True, download=True, transform=transform)
    test_ds = tvds.CIFAR10(cache, train=False, download=True, transform=transform)
    return train_ds, test_ds, 10


def load_dataset(dataset_name: str):
    if dataset_name in ("mnist", "fashion_mnist"):
        return _load_mnist(dataset_name)
    elif dataset_name == "cifar10":
        return _load_cifar10()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LTCNN NAS via MASE search framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a TOML search config (e.g. configs/ltcnn/search_ltcnn_mnist.toml)",
    )
    parser.add_argument(
        "--save-dir",
        default="mase_output/ltcnn/nas",
        help="Directory where search results are written",
    )
    args = parser.parse_args()

    try:
        import toml
        config = toml.load(args.config)
    except ModuleNotFoundError:
        import tomllib
        with open(args.config, "rb") as f:
            config = tomllib.load(f)

    ss_setup = config["search"]["search_space"]["setup"]
    dataset_name = config.get("dataset", "mnist")
    batch_size = config.get("batch_size", 64)
    seed = config.get("seed", 42)

    torch.manual_seed(seed)

    train_ds, val_ds, num_classes = load_dataset(dataset_name)
    data_module = LTCNNDataModule(train_ds, val_ds, batch_size)

    from chop.nn.ltcnn.model import LTCNN

    n_values = ss_setup["n_values"]
    conv_channels_options = ss_setup["conv_channels_options"]
    kernel_size_options = ss_setup["kernel_size_options"]
    ff_hidden_sizes_options = ss_setup["ff_hidden_sizes_options"]
    bit_depth_options = ss_setup["bit_depth_options"]

    first_conv = (
        list(conv_channels_options[0])
        if isinstance(conv_channels_options[0], list)
        else list(conv_channels_options)
    )

    template_model = LTCNN(
        in_channels=ss_setup.get("in_channels", 1),
        num_classes=num_classes,
        image_size=ss_setup.get("image_size", 28),
        bit_depth=bit_depth_options[0],
        encoding=ss_setup.get("encoding", "quantization"),
        n=n_values[0],
        conv_channels=first_conv,
        kernel_size=kernel_size_options[0],
        ff_hidden_sizes=list(ff_hidden_sizes_options[0]),
        tau=ss_setup.get("tau", 10.0),
        learnable_mapping=ss_setup.get("learnable_mapping", False),
        Q=None,
    )

    accel = config.get("accelerator", "auto")
    if accel not in ("auto", "cpu", "gpu"):
        accel = "gpu" if accel in ("cuda",) else "cpu"

    from chop.actions.search.search import search

    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    class _NoopVisualizer:
        def log_metrics(self, **kwargs):
            pass

    search(
        model=template_model,
        model_info=None,
        task=config.get("task", "cls"),
        dataset_info=DatasetInfo(num_classes),
        data_module=data_module,
        search_config=config,
        save_path=save_path,
        accelerator=accel,
        visualizer=_NoopVisualizer(),
    )


if __name__ == "__main__":
    main()
