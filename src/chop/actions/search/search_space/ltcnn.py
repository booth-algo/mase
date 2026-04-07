import math
from typing import Any

from .base import SearchSpaceBase


class LTCNNSearchSpace(SearchSpaceBase):
    """Search space for LTCNN architecture NAS.

    Searches over n, conv_channels, kernel_size, ff_hidden_sizes, bit_depth, Q.
    See configs/ltcnn/search_ltcnn_mnist.toml for expected config keys.
    """

    def _post_init_setup(self) -> None:
        setup: dict[str, Any] = self.config["setup"]

        self.n_values: list[int] = setup["n_values"]
        if not self.n_values:
            raise ValueError("n_values must be non-empty in search space config")

        self.conv_channels_options: list = setup["conv_channels_options"]
        if not self.conv_channels_options:
            raise ValueError(
                "conv_channels_options must be non-empty in search space config"
            )
        self.search_conv_channels: bool = isinstance(
            self.conv_channels_options[0], list
        )

        self.kernel_size_options: list[int] = setup["kernel_size_options"]
        if not self.kernel_size_options:
            raise ValueError(
                "kernel_size_options must be non-empty in search space config"
            )

        self.ff_hidden_sizes_options: list[list[int]] = setup["ff_hidden_sizes_options"]
        if not self.ff_hidden_sizes_options:
            raise ValueError(
                "ff_hidden_sizes_options must be non-empty in search space config"
            )

        self.bit_depth_options: list[int] = setup["bit_depth_options"]
        if not self.bit_depth_options:
            raise ValueError(
                "bit_depth_options must be non-empty in search space config"
            )

        # Q == 0 in config means "no subsampling" (maps to None inside LTCNN)
        self.Q_options: list[int] = setup["Q_options"]
        if not self.Q_options:
            raise ValueError("Q_options must be non-empty in search space config")

        self.encoding: str = setup.get("encoding", "quantization")
        self.in_channels: int = int(setup.get("in_channels", 1))
        self.num_classes: int = int(setup.get("num_classes", 10))
        self.image_size: int = int(setup.get("image_size", 28))
        self.tau: float = float(setup.get("tau", 10.0))
        self.learnable_mapping: bool = bool(setup.get("learnable_mapping", False))

    def build_search_space(self) -> None:
        """Populate ``choices_flattened`` and ``choice_lengths_flattened``."""
        self.choices_flattened["global/n"] = list(self.n_values)
        self.choice_lengths_flattened["global/n"] = len(self.n_values)

        if self.search_conv_channels:
            self.choices_flattened["global/conv_channels"] = [
                list(opt) for opt in self.conv_channels_options
            ]
            self.choice_lengths_flattened["global/conv_channels"] = len(
                self.conv_channels_options
            )

        self.choices_flattened["global/kernel_size"] = list(self.kernel_size_options)
        self.choice_lengths_flattened["global/kernel_size"] = len(
            self.kernel_size_options
        )

        self.choices_flattened["global/ff_hidden_sizes"] = [
            list(opt) for opt in self.ff_hidden_sizes_options
        ]
        self.choice_lengths_flattened["global/ff_hidden_sizes"] = len(
            self.ff_hidden_sizes_options
        )

        self.choices_flattened["global/bit_depth"] = list(self.bit_depth_options)
        self.choice_lengths_flattened["global/bit_depth"] = len(self.bit_depth_options)

        self.choices_flattened["global/Q"] = list(self.Q_options)
        self.choice_lengths_flattened["global/Q"] = len(self.Q_options)

    def optuna_sampler(self, trial) -> dict[str, Any]:
        """Sample a configuration using an Optuna trial object.

        Args:
            trial: an ``optuna.trial.Trial`` instance.

        Returns:
            A configuration dict ready to be passed to :meth:`rebuild_model`.
        """
        n = trial.suggest_categorical("global_n", self.n_values)

        if self.search_conv_channels:
            idx = trial.suggest_int("global_conv_channels", 0, len(self.conv_channels_options) - 1)
            conv_channels = list(self.conv_channels_options[idx])
        else:
            conv_channels = list(self.conv_channels_options)

        kernel_size = trial.suggest_categorical(
            "global_kernel_size", self.kernel_size_options
        )

        idx_ff = trial.suggest_int(
            "global_ff_hidden_sizes", 0, len(self.ff_hidden_sizes_options) - 1
        )
        ff_hidden_sizes = list(self.ff_hidden_sizes_options[idx_ff])

        bit_depth = trial.suggest_categorical(
            "global_bit_depth", self.bit_depth_options
        )

        Q_raw = trial.suggest_categorical("global_Q", self.Q_options)

        return self._make_config(n, conv_channels, kernel_size, ff_hidden_sizes, bit_depth, Q_raw)

    def flattened_indexes_to_config(self, indexes: dict[str, int]) -> dict[str, Any]:
        """Convert flat index dict to a model config dict.

        Args:
            indexes: mapping from choice key to integer index within that key's
                     choices list (as produced by the search strategy).

        Returns:
            A configuration dict ready to be passed to :meth:`rebuild_model`.
        """
        n = self.n_values[indexes["global/n"]]

        if self.search_conv_channels:
            conv_channels = list(
                self.conv_channels_options[indexes["global/conv_channels"]]
            )
        else:
            conv_channels = list(self.conv_channels_options)

        kernel_size = self.kernel_size_options[indexes["global/kernel_size"]]
        ff_hidden_sizes = list(
            self.ff_hidden_sizes_options[indexes["global/ff_hidden_sizes"]]
        )
        bit_depth = self.bit_depth_options[indexes["global/bit_depth"]]
        Q_raw = self.Q_options[indexes["global/Q"]]

        return self._make_config(n, conv_channels, kernel_size, ff_hidden_sizes, bit_depth, Q_raw)

    def rebuild_model(self, sampled_config: dict[str, Any], is_eval_mode: bool):
        """Instantiate an LTCNN from the given sampled configuration.

        Args:
            sampled_config: dict containing a ``"model_config"`` sub-dict as
                returned by :meth:`_make_config`.
            is_eval_mode: when ``True`` the returned model is in eval mode.

        Returns:
            An ``LTCNN`` instance placed on ``self.accelerator``.
        """
        from chop.nn.ltcnn.model import LTCNN

        cfg = sampled_config["model_config"]

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
            learnable_mapping=cfg["learnable_mapping"],
            Q=cfg["Q"],
        )

        model = model.to(self.accelerator)
        model.eval() if is_eval_mode else model.train()
        return model

    def _make_config(
        self,
        n: int,
        conv_channels: list[int],
        kernel_size: int,
        ff_hidden_sizes: list[int],
        bit_depth: int,
        Q_raw: int,
    ) -> dict[str, Any]:
        Q = None if Q_raw == 0 else int(Q_raw)
        # Paper: tau = sqrt(nf / C) where nf = final layer output nodes
        last_hidden = ff_hidden_sizes[-1]
        group_size = max(1, last_hidden // self.num_classes)
        nf = self.num_classes * group_size
        tau = math.sqrt(nf / self.num_classes)
        return {
            "model_config": {
                "in_channels": self.in_channels,
                "num_classes": self.num_classes,
                "image_size": self.image_size,
                "encoding": self.encoding,
                "tau": tau,
                "learnable_mapping": self.learnable_mapping,
                "n": n,
                "conv_channels": conv_channels,
                "kernel_size": kernel_size,
                "ff_hidden_sizes": ff_hidden_sizes,
                "bit_depth": bit_depth,
                "Q": Q,
            }
        }
