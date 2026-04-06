import torch

from .base import SearchSpaceBase


class DWNSearchSpace(SearchSpaceBase):

    def _post_init_setup(self) -> None:
        config = self.config
        setup = config["setup"]

        self.n_values = setup["n_values"]
        self.hidden_sizes_options = setup["hidden_sizes"]
        self.search_hidden_sizes = isinstance(self.hidden_sizes_options[0], list)
        self.num_layers = len(self.hidden_sizes_options)

        self.num_bits = setup["num_bits"]
        self.mapping_first = setup["mapping_first"]
        self.mapping_rest = setup["mapping_rest"]
        self.tau = setup["tau"]
        self.lambda_reg = setup.get("lambda_reg", 0.0)

        self.input_features = self.model.input_features
        self.num_classes = self.model.num_classes

        normalize_input = setup.get("normalize_input", True)
        max_samples = setup.get("max_thermo_samples", 50000)
        batches = []
        n = 0
        for batch in self.data_module.train_dataloader():
            x = batch[0]
            batches.append(x)
            n += x.size(0)
            if n >= max_samples:
                break
        self._thermo_data = torch.cat(batches, dim=0)[:max_samples].float()
        if normalize_input:
            self._thermo_data = self._thermo_data / 255.0

    def build_search_space(self) -> None:
        for i in range(self.num_layers):
            key = f"layer_{i}/lut_n"
            self.choices_flattened[key] = list(self.n_values)
            self.choice_lengths_flattened[key] = len(self.n_values)
            if self.search_hidden_sizes:
                key = f"layer_{i}/hidden_size"
                self.choices_flattened[key] = list(self.hidden_sizes_options[i])
                self.choice_lengths_flattened[key] = len(self.hidden_sizes_options[i])

    def optuna_sampler(self, trial):
        lut_n_list = []
        hidden_sizes = []
        for i in range(self.num_layers):
            n = trial.suggest_categorical(f"layer_{i}_lut_n", self.n_values)
            lut_n_list.append(n)
            if self.search_hidden_sizes:
                hs = trial.suggest_categorical(
                    f"layer_{i}_hidden_size", self.hidden_sizes_options[i]
                )
                hidden_sizes.append(hs)
        if not self.search_hidden_sizes:
            hidden_sizes = list(self.hidden_sizes_options)
        return self._make_config(lut_n_list, hidden_sizes)

    def flattened_indexes_to_config(self, indexes):
        lut_n_list = [
            self.n_values[indexes[f"layer_{i}/lut_n"]]
            for i in range(self.num_layers)
        ]
        if self.search_hidden_sizes:
            hidden_sizes = [
                self.hidden_sizes_options[i][indexes[f"layer_{i}/hidden_size"]]
                for i in range(self.num_layers)
            ]
        else:
            hidden_sizes = list(self.hidden_sizes_options)
        return self._make_config(lut_n_list, hidden_sizes)

    def _make_config(self, lut_n_list, hidden_sizes):
        return {
            "model_config": {
                "input_features": self.input_features,
                "num_classes": self.num_classes,
                "num_bits": self.num_bits,
                "hidden_sizes": hidden_sizes,
                "lut_n": lut_n_list,
                "mapping_first": self.mapping_first,
                "mapping_rest": self.mapping_rest,
                "tau": self.tau,
                "lambda_reg": self.lambda_reg,
            }
        }

    def rebuild_model(self, sampled_config, is_eval_mode):
        from chop.nn.dwn.model import DWNModel

        cfg = sampled_config["model_config"]
        if cfg["hidden_sizes"][-1] % cfg["num_classes"] != 0:
            raise ValueError(
                f"Last hidden_size ({cfg['hidden_sizes'][-1]}) must be "
                f"divisible by num_classes ({cfg['num_classes']})"
            )

        model = DWNModel(
            input_features=cfg["input_features"],
            num_classes=cfg["num_classes"],
            num_bits=cfg["num_bits"],
            hidden_sizes=cfg["hidden_sizes"],
            lut_n=cfg["lut_n"],
            mapping_first=cfg["mapping_first"],
            mapping_rest=cfg.get("mapping_rest", "random"),
            tau=cfg["tau"],
            lambda_reg=cfg.get("lambda_reg", 0.0),
        )
        model.fit_thermometer(self._thermo_data)
        model = model.to(self.accelerator)
        if is_eval_mode:
            model.eval()
        else:
            model.train()
        return model
