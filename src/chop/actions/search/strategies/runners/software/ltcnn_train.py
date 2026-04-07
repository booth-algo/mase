import torch
import torch.nn as nn

from .base import SWRunnerBase


class RunnerLTCNNTrain(SWRunnerBase):
    """Train an LTCNN during NAS and return accuracy/loss metrics."""

    available_metrics = ("loss", "accuracy")

    def _post_init_setup(self) -> None:
        cfg = self.config or {}
        if "max_epochs" not in cfg:
            raise AssertionError("max_epochs is required in runner config")

        self.max_epochs: int = int(cfg["max_epochs"])
        self.learning_rate: float = float(cfg.get("learning_rate", 0.02))
        self.optimizer_name: str = cfg.get("optimizer_name", "adam")
        self.lr_gamma: float = float(cfg.get("lr_gamma", 0.95))
        self.patience: int = int(cfg.get("patience", 10))
        self.eval_split: str = cfg.get("eval_split", "val_dataloader")
        self.num_eval_samples: int = int(cfg.get("num_eval_samples", -1))

    def __call__(self, data_module, model, sampled_config) -> dict[str, float]:
        if not isinstance(model, torch.nn.Module):
            model = model.model

        device = self.accelerator
        model = model.to(device)
        use_amp = device != "cpu" and device != torch.device("cpu")

        optimizer = self._build_optimizer(model)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_gamma)
        criterion = nn.NLLLoss()

        best_accuracy = 0.0
        epochs_no_improve = 0

        train_loader = data_module.train_dataloader()
        for _ in range(self.max_epochs):
            model.train()
            for batch in train_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    log_probs = model(x)
                    loss = criterion(log_probs, y)
                loss.backward()
                optimizer.step()
            scheduler.step()

            if self.patience > 0:
                acc = self._evaluate_accuracy(model, data_module, device, use_amp)
                if acc > best_accuracy:
                    best_accuracy = acc
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        break

        model.eval()
        avg_loss, accuracy = self._evaluate(model, data_module, device, criterion, use_amp)
        return {"loss": float(avg_loss), "accuracy": float(accuracy)}

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        lr = self.learning_rate
        match self.optimizer_name:
            case "adam":
                return torch.optim.Adam(model.parameters(), lr=lr)
            case "adamw":
                return torch.optim.AdamW(model.parameters(), lr=lr)
            case "sgd":
                return torch.optim.SGD(model.parameters(), lr=lr)
            case _:
                raise ValueError(f"Unsupported optimizer: {self.optimizer_name!r}")

    def _evaluate_accuracy(self, model, data_module, device, use_amp) -> float:
        model.eval()
        correct = total = 0
        loader = getattr(data_module, self.eval_split)()
        with torch.no_grad():
            for batch in loader:
                x, y = batch[0].to(device), batch[1].to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += x.size(0)
                if self.num_eval_samples > 0 and total >= self.num_eval_samples:
                    break
        model.train()
        return correct / total if total > 0 else 0.0

    def _evaluate(self, model, data_module, device, criterion, use_amp):
        loader = getattr(data_module, self.eval_split)()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in loader:
                x, y = batch[0].to(device), batch[1].to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    log_probs = model(x)
                total_loss += criterion(log_probs, y).item() * x.size(0)
                total_correct += (log_probs.argmax(dim=1) == y).sum().item()
                total_samples += x.size(0)
                if self.num_eval_samples > 0 and total_samples >= self.num_eval_samples:
                    break

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        return avg_loss, accuracy
