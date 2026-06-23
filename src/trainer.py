"""Training and evaluation loop for the FFPA-Net baseline."""
import csv
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .loss import DeepSupervisionLoss
from .metrics import MetricsAccumulator, compute_dice_both


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, model: nn.Module, config: Dict, output_dir: Path, device: str = "cuda"):
        self.model = model.to(device)
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        self.num_classes = config["model"].get(
            "actual_num_classes", config["model"].get("num_classes", 1)
        )
        self.is_binary = self.num_classes == 1
        self.use_deep_supervision = config.get("ffpa", {}).get("use_deep_supervision", True)

        train_cfg = config["training"]
        self.epochs = train_cfg["epochs"]
        self.lr = train_cfg["learning_rate"]
        self.weight_decay = train_cfg["weight_decay"]
        self.val_interval = train_cfg.get("val_interval", 1)
        self.save_interval = train_cfg.get("save_interval", 10)

        es_cfg = train_cfg.get("early_stopping", {})
        self.early_stopping = es_cfg.get("enabled", True)
        self.patience = es_cfg.get("patience", 15)
        self.min_delta = es_cfg.get("min_delta", 0.001)

        loss_cfg = config.get("loss", {})
        self.criterion = DeepSupervisionLoss(
            num_classes=self.num_classes,
            weight_dice=loss_cfg.get("weight_dice", 0.5),
            weight_ce=loss_cfg.get("weight_ce", 0.5),
        )

        self.optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        self.best_dice_fg = 0.0
        self.best_dice = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.training_log = []

        with open(self.output_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        print(f"\nStarting training for {self.epochs} epochs...")
        print(f"  Classes: {self.num_classes}")
        print(f"  Deep supervision: {self.use_deep_supervision}")

        for epoch in range(self.epochs):
            train_loss, train_dice, train_dice_fg = self._train_epoch(train_loader, epoch)

            if (epoch + 1) % self.val_interval == 0:
                val_loss, val_dice, val_dice_fg = self._validate(val_loader)
                self.training_log.append(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "train_dice": train_dice,
                        "train_dice_fg": train_dice_fg,
                        "val_loss": val_loss,
                        "val_dice": val_dice,
                        "val_dice_fg": val_dice_fg,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                )
                print(
                    f"Epoch {epoch + 1}/{self.epochs} | "
                    f"Train Loss: {train_loss:.4f}, Dice_FG: {train_dice_fg:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Dice_FG: {val_dice_fg:.4f}"
                )

                if val_dice_fg > self.best_dice_fg + self.min_delta:
                    self.best_dice = val_dice
                    self.best_dice_fg = val_dice_fg
                    self.best_epoch = epoch + 1
                    self.patience_counter = 0
                    self._save_checkpoint("best_model.pth", epoch, val_dice, val_dice_fg)
                else:
                    self.patience_counter += 1

                if self.early_stopping and self.patience_counter >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth", epoch)

            self.scheduler.step()

        self._save_training_log()
        print(
            f"\nTraining complete. Best Dice_FG: {self.best_dice_fg:.4f} "
            f"at epoch {self.best_epoch}"
        )
        return {
            "best_dice": self.best_dice,
            "best_dice_fg": self.best_dice_fg,
            "best_epoch": self.best_epoch,
            "training_log": self.training_log,
        }

    def evaluate(self, data_loader: DataLoader, save_predictions: bool = False) -> Dict:
        self.model.eval()
        accumulator = MetricsAccumulator(self.num_classes)
        pred_dir = self.output_dir / "predictions"
        if save_predictions:
            pred_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                outputs = self.model(images, return_aux=False)

                for i in range(images.size(0)):
                    pred_i = outputs[i].detach().cpu()
                    target_i = masks[i].detach().cpu()
                    patient_id = batch["patient_id"][i]
                    slice_name = batch["slice_name"][i]
                    accumulator.update(pred_i, target_i, patient_id, slice_name)

                    if save_predictions:
                        self._save_prediction(
                            pred_i, patient_id, slice_name, pred_dir
                        )

        summary = accumulator.get_summary()
        self._save_slice_metrics(accumulator.slice_metrics)
        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    def _train_epoch(self, train_loader, epoch) -> Tuple[float, float, float]:
        self.model.train()
        total_loss = total_dice = total_dice_fg = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images, return_aux=self.use_deep_supervision)
            loss = self.criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            with torch.no_grad():
                main_output = outputs[0] if isinstance(outputs, tuple) else outputs
                dice, dice_fg = self._batch_metrics(main_output, masks)

            total_loss += loss.item()
            total_dice += dice
            total_dice_fg += dice_fg

        n = len(train_loader)
        return total_loss / n, total_dice / n, total_dice_fg / n

    def _validate(self, val_loader) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = total_dice = total_dice_fg = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                outputs = self.model(images, return_aux=False)
                loss = self.criterion(outputs, masks)
                dice, dice_fg = self._batch_metrics(outputs, masks)
                total_loss += loss.item()
                total_dice += dice
                total_dice_fg += dice_fg

        n = len(val_loader)
        return total_loss / n, total_dice / n, total_dice_fg / n

    def _batch_metrics(self, pred, target) -> Tuple[float, float]:
        if self.is_binary:
            pred_binary = (torch.sigmoid(pred) > 0.5).float()
            if target.dim() == 3:
                target = target.unsqueeze(1)
            intersection = (pred_binary * target).sum(dim=(1, 2, 3))
            union = pred_binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
            dice = ((2 * intersection + 1e-5) / (union + 1e-5)).mean().item()
            return dice, dice

        dice_scores = []
        dice_fg_scores = []
        for b in range(pred.size(0)):
            dice_bg, dice_fg = compute_dice_both(
                pred[b].detach().cpu(),
                target[b].detach().cpu(),
                self.num_classes,
            )
            dice_scores.append(dice_bg)
            dice_fg_scores.append(dice_fg)
        return float(np.mean(dice_scores)), float(np.mean(dice_fg_scores))

    def _save_checkpoint(self, filename, epoch, val_dice=None, val_dice_fg=None):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_dice": self.best_dice,
            "best_dice_fg": self.best_dice_fg,
            "num_classes": self.num_classes,
            "config": self.config,
        }
        if val_dice is not None:
            checkpoint["val_dice"] = val_dice
        if val_dice_fg is not None:
            checkpoint["val_dice_fg"] = val_dice_fg
        torch.save(checkpoint, self.output_dir / filename)

    def _save_training_log(self):
        if not self.training_log:
            return
        fieldnames = list(self.training_log[0].keys())
        with open(self.output_dir / "training_log.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.training_log)

    def _save_slice_metrics(self, slice_metrics):
        if not slice_metrics:
            return
        fieldnames = list(slice_metrics[0].keys())
        with open(self.output_dir / "slice_metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(slice_metrics)

    def _save_prediction(self, pred, patient_id, slice_name, pred_dir):
        import cv2

        patient_dir = pred_dir / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)

        if self.is_binary:
            mask = (torch.sigmoid(pred).squeeze() > 0.5).numpy().astype(np.uint8) * 255
        else:
            class_info = self.config["model"].get("class_info", {})
            inverse_mapping = class_info.get("inverse_mapping", {})
            pred_class = torch.argmax(pred, dim=0).numpy()
            mask = np.zeros_like(pred_class, dtype=np.uint8)
            for remapped_id, original_id in inverse_mapping.items():
                mask[pred_class == remapped_id] = original_id

        cv2.imwrite(str(patient_dir / slice_name), mask)
