from .base import HWRunnerBase


class RunnerDWNArea(HWRunnerBase):
    available_metrics = ("area_luts",)

    def _post_init_setup(self) -> None:
        pass

    def __call__(self, data_module, model, sampled_config) -> dict[str, float]:
        if hasattr(model, "lut_layers"):
            area = sum(
                layer.output_size * (2 ** layer.n) for layer in model.lut_layers
            )
        else:
            cfg = sampled_config.get("model_config", sampled_config)
            area = sum(
                hs * (2 ** n)
                for hs, n in zip(cfg["hidden_sizes"], cfg["lut_n"])
            )
        return {"area_luts": float(area)}
