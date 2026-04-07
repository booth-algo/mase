import math

from .base import HWRunnerBase


class RunnerLTCNNArea(HWRunnerBase):
    """Compute total LUT area for LTCNN models."""

    available_metrics = ("area_luts",)

    def _post_init_setup(self) -> None:
        pass

    def __call__(
        self, data_module, model, sampled_config, num_batches: int = 0
    ) -> dict[str, float]:
        if not hasattr(model, "conv_layers"):
            model = getattr(model, "model", model)

        if hasattr(model, "conv_layers") and hasattr(model, "ff_layers"):
            from chop.nn.ltcnn.metrics import compute_area_luts

            area = compute_area_luts(model)
        else:
            area = self._compute_area_from_config(sampled_config)

        return {"area_luts": float(area)}

    @staticmethod
    def _compute_area_from_config(sampled_config: dict) -> int:
        """Estimate LUT area analytically from config when model is unavailable."""
        cfg = sampled_config.get("model_config", sampled_config)

        n = cfg["n"]
        in_channels = cfg["in_channels"]
        bit_depth = cfg["bit_depth"]
        conv_channels = cfg["conv_channels"]
        kernel_size = cfg["kernel_size"]
        ff_hidden_sizes = cfg["ff_hidden_sizes"]
        num_classes = cfg["num_classes"]
        Q_raw = cfg.get("Q", 0)
        Q = None if (Q_raw == 0 or Q_raw is None) else int(Q_raw)

        area = 0

        encoded_channels = in_channels * bit_depth
        ch_in = encoded_channels
        for ch_out in conv_channels:
            effective_ch = Q if (Q is not None and Q < ch_in) else ch_in
            n_leaf_inputs = kernel_size * kernel_size * effective_ch
            current_size = n_leaf_inputs
            while current_size > 1:
                num_nodes = math.ceil(current_size / n)
                area += ch_out * num_nodes * (2 ** n)
                current_size = num_nodes
            ch_in = ch_out

        for out_size in ff_hidden_sizes:
            area += out_size * (2 ** n)

        group_size = max(1, ff_hidden_sizes[-1] // num_classes)
        final_out = num_classes * group_size
        area += final_out * (2 ** n)

        return area
