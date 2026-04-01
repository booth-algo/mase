#!/usr/bin/env python3
import sys
import os
import argparse
from pathlib import Path

_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.insert(0, _src)

import torch
from torch.utils.data import DataLoader


class DWNDataModule:
    def __init__(self, train_ds, val_ds, batch_size):
        self.batch_size = batch_size
        self._train_ds = train_ds
        self._val_ds = val_ds

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self._train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self._val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)


class DatasetInfo:
    def __init__(self, num_classes):
        self.num_classes = num_classes


def _load_mnist_vision(dataset_name):
    from torchvision import datasets as tvdatasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    cache = os.path.expanduser(f"~/.cache/{dataset_name}")
    cls = tvdatasets.MNIST if dataset_name == "mnist" else tvdatasets.FashionMNIST
    train_ds = cls(cache, train=True, download=True, transform=transform)
    test_ds = cls(cache, train=False, download=True, transform=transform)
    return train_ds, test_ds, 784, 10


def _load_cifar10():
    from torchvision import datasets as tvdatasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    cache = os.path.expanduser("~/.cache/cifar10")
    train_ds = tvdatasets.CIFAR10(cache, train=True, download=True, transform=transform)
    test_ds = tvdatasets.CIFAR10(cache, train=False, download=True, transform=transform)
    return train_ds, test_ds, 3072, 10


def load_dataset(dataset_name):
    if dataset_name in ("mnist", "fashion_mnist"):
        return _load_mnist_vision(dataset_name)
    elif dataset_name == "cifar10":
        return _load_cifar10()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name!r}. Choose from: mnist, fashion_mnist, cifar10")


def main():
    parser = argparse.ArgumentParser(description="DWN NAS via MASE search framework")
    parser.add_argument("--config", required=True, help="Path to TOML search config")
    parser.add_argument("--save-dir", default="mase_output/dwn/nas", help="Output directory")
    args = parser.parse_args()

    import toml
    config = toml.load(args.config)

    ss_setup = config["search"]["search_space"]["setup"]
    dataset_name = config.get("dataset", "mnist")
    batch_size = config.get("batch_size", 32)
    seed = config.get("seed", 42)

    torch.manual_seed(seed)

    train_ds, val_ds, input_features, num_classes = load_dataset(dataset_name)
    data_module = DWNDataModule(train_ds, val_ds, batch_size)

    from chop.nn.dwn.model import DWNModel

    hidden_sizes = ss_setup["hidden_sizes"]
    if isinstance(hidden_sizes[0], list):
        hidden_sizes = [hs[0] for hs in hidden_sizes]

    template_model = DWNModel(
        input_features=input_features,
        num_classes=num_classes,
        num_bits=ss_setup.get("num_bits", 3),
        hidden_sizes=hidden_sizes,
        lut_n=ss_setup["n_values"][0],
    )

    accel = config.get("accelerator", "auto")
    if accel == "auto":
        accel = "cuda" if torch.cuda.is_available() else "cpu"

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
