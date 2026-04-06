import torch
import torch.nn as nn

from chop.nn.dwn.metrics import compute_entropy_loss

from .base import SWRunnerBase


class RunnerDWNTrain(SWRunnerBase):
    available_metrics = ("loss", "accuracy")

    def _post_init_setup(self) -> None:
        cfg = self.config or {}
        assert "max_epochs" in cfg, "max_epochs is required in runner config"
        self.max_epochs = cfg["max_epochs"]
        self.learning_rate = cfg.get("learning_rate", 0.01)
        self.optimizer_name = cfg.get("optimizer_name", "adam")
        self.area_lambda = cfg.get("area_lambda", 0.0)
        self.lr_step = cfg.get("lr_step", None)
        self.lr_gamma = cfg.get("lr_gamma", 0.1)
        self.eval_split = cfg.get("eval_split", "val_dataloader")
        self.num_eval_samples = cfg.get("num_eval_samples", -1)

    def __call__(self, data_module, model, sampled_config) -> dict[str, float]:
        if not isinstance(model, torch.nn.Module):
            model = model.model

        device = self.accelerator
        model = model.to(device)

        match self.optimizer_name:
            case "adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            case "adamw":
                optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
            case "sgd":
                optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
            case _:
                raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        scheduler = None
        if self.lr_step is not None:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.lr_step, gamma=self.lr_gamma
            )

        criterion = nn.CrossEntropyLoss()

        for _ in range(self.max_epochs):
            model.train()
            for batch in data_module.train_dataloader():
                x, y = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                if self.area_lambda > 0.0:
                    loss = loss + self.area_lambda * compute_entropy_loss(model)
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

        model.eval()
        eval_loader = getattr(data_module, self.eval_split)()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in eval_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                logits = model(x)
                total_loss += criterion(logits, y).item() * x.size(0)
                total_correct += (logits.argmax(1) == y).sum().item()
                total_samples += x.size(0)
                if self.num_eval_samples > 0 and total_samples >= self.num_eval_samples:
                    break

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        return {"loss": float(avg_loss), "accuracy": float(accuracy)}
