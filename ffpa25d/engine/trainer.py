"""
Training + evaluation engine for the 2.5D FFPA-Net.

  * training uses return_aux = use_deep_supervision (deep supervision on)
  * validation / evaluation use return_aux = False (single-output, = inference)
  * best checkpoint is selected by validation Dice_FG

The Evaluator writes summary.json with the keys analysis/make_tables.py expects
(dice / dice_fg / hd95 / num_slices / training).
"""
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from ..losses.loss import DeepSupervisionLoss
from ..metrics import MetricsAccumulator, compute_dice_both, get_model_efficiency


def _num_classes(config: Dict) -> int:
    return config["model"].get("actual_num_classes",
                               config["model"].get("num_classes", 1))


def _input_size(model, config) -> tuple:
    ch = 3 if getattr(model, "three_slice", False) else 1
    s = config["data"]["image_size"]
    return (1, ch, s, s)


def _quick_dice_fg(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """Cheap batch Dice_FG for progress display, via metrics.compute_dice_both."""
    with torch.no_grad():
        if target.dim() == 4:
            target = target.squeeze(1)
        vals = [compute_dice_both(pred[b], target[b], num_classes)[1]
                for b in range(pred.size(0))]
        return float(np.mean(vals))


class Trainer:
    def __init__(self, model, config: Dict, output_dir, device: str = "cuda"):
        self.model = model.to(device)
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        self.num_classes = _num_classes(config)
        self.is_binary = self.num_classes == 1
        self.use_deep_supervision = config.get("ffpa", {}).get("use_deep_supervision", True)

        tcfg = config["training"]
        self.epochs = tcfg["epochs"]
        self.val_interval = tcfg.get("val_interval", 1)
        self.save_interval = tcfg.get("save_interval", 10)
        es = tcfg.get("early_stopping", {})
        self.early_stopping = es.get("enabled", True)
        self.patience = es.get("patience", 15)
        self.min_delta = es.get("min_delta", 0.001)

        lcfg = config.get("loss", {})
        self.criterion = DeepSupervisionLoss(
            num_classes=self.num_classes,
            weight_dice=lcfg.get("weight_dice", 0.5),
            weight_ce=lcfg.get("weight_ce", lcfg.get("weight_bce", 0.5)),
        )
        self.optimizer = optim.AdamW(model.parameters(), lr=tcfg["learning_rate"],
                                     weight_decay=tcfg["weight_decay"])
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6)

        self.best_dice_fg = 0.0
        self.best_dice = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.training_log = []

    def train(self, train_loader, val_loader) -> Dict:
        print(f"\nStarting training for {self.epochs} epochs "
              f"({'binary' if self.is_binary else f'{self.num_classes}-class'}, "
              f"DS={self.use_deep_supervision})...")
        last_epoch = 0
        for epoch in range(self.epochs):
            last_epoch = epoch
            tr_loss, tr_dice = self._run_epoch(train_loader, epoch, train=True)

            if (epoch + 1) % self.val_interval == 0:
                va_loss, va_dice = self._run_epoch(val_loader, epoch, train=False)
                self.training_log.append({
                    "epoch": epoch + 1, "train_loss": tr_loss, "train_dice_fg": tr_dice,
                    "val_loss": va_loss, "val_dice_fg": va_dice,
                    "lr": self.optimizer.param_groups[0]["lr"]})
                print(f"Epoch {epoch+1}/{self.epochs} | "
                      f"train loss {tr_loss:.4f} dice_fg {tr_dice:.4f} | "
                      f"val loss {va_loss:.4f} dice_fg {va_dice:.4f}")

                if va_dice > self.best_dice_fg + self.min_delta:
                    self.best_dice_fg = va_dice
                    self.best_epoch = epoch + 1
                    self.patience_counter = 0
                    self._save_checkpoint("best_model.pth", epoch, va_dice)
                else:
                    self.patience_counter += 1
                if self.early_stopping and self.patience_counter >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth", epoch)
            self.scheduler.step()

        self._save_log()
        print(f"\nTraining complete. Best Dice_FG {self.best_dice_fg:.4f} @ epoch {self.best_epoch}")
        return {"best_epoch": self.best_epoch, "final_epoch": last_epoch + 1,
                "best_val_dice_fg": self.best_dice_fg}

    def _run_epoch(self, loader, epoch, train: bool):
        self.model.train() if train else self.model.eval()
        total_loss, total_dice, n = 0.0, 0.0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        desc = f"Epoch {epoch+1}" if train else "  val"
        with ctx:
            for batch in tqdm(loader, desc=desc, leave=False):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                if train:
                    self.optimizer.zero_grad()
                    outputs = self.model(images, return_aux=self.use_deep_supervision)
                    loss = self.criterion(outputs, masks)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    main = outputs[0] if isinstance(outputs, tuple) else outputs
                else:
                    outputs = self.model(images, return_aux=False)
                    loss = self.criterion(outputs, masks)
                    main = outputs
                total_loss += loss.item()
                total_dice += _quick_dice_fg(main, masks, self.num_classes)
                n += 1
        n = max(n, 1)
        return total_loss / n, total_dice / n

    def _save_checkpoint(self, name, epoch, val_dice_fg=None):
        ckpt = {"epoch": epoch, "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "num_classes": self.num_classes, "config": self.config}
        if val_dice_fg is not None:
            ckpt["val_dice_fg"] = val_dice_fg
        torch.save(ckpt, self.output_dir / name)

    def _save_log(self):
        if not self.training_log:
            return
        with open(self.output_dir / "training_log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.training_log[0].keys()))
            w.writeheader()
            w.writerows(self.training_log)


class Evaluator:
    def __init__(self, model, config: Dict, output_dir, device: str = "cuda"):
        self.model = model.to(device)
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device
        self.num_classes = _num_classes(config)
        self.is_binary = self.num_classes == 1

        ecfg = config.get("evaluation", {})
        self.threshold = ecfg.get("threshold", 0.5)
        self.save_predictions = ecfg.get("save_predictions", True)
        self.pred_dir = self.output_dir / "predictions"
        if self.save_predictions:
            self.pred_dir.mkdir(parents=True, exist_ok=True)

        class_info = config["model"].get("class_info") or {}
        self.inverse_mapping = class_info.get("inverse_mapping")  # {idx: original_id}

    def evaluate(self, test_loader) -> Dict:
        print("\nEvaluating on test set...")
        self.model.eval()
        acc = MetricsAccumulator(num_classes=self.num_classes)
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                pids, snames = batch["patient_id"], batch["slice_name"]
                outputs = self.model(images, return_aux=False)
                for i in range(images.size(0)):
                    pid = pids[i] if isinstance(pids, list) else pids
                    sname = snames[i] if isinstance(snames, list) else snames
                    acc.update(outputs[i], masks[i], pid, sname, self.threshold)
                    if self.save_predictions:
                        self._save_pred(outputs[i], pid, sname)

        summary_metrics = acc.get_summary()        # {dice:{mean,std},...,num_slices,num_patients}
        self._save_slice_metrics(acc.get_slice_metrics())
        efficiency = get_model_efficiency(
            self.model, input_size=_input_size(self.model, self.config), device=self.device)

        summary = {
            "variant": self.config.get("experiment", {}).get("name", "unknown"),
            "num_classes": self.num_classes,
            "model": efficiency,
            "timestamp": datetime.now().isoformat(),
            **summary_metrics,                       # dice / dice_fg / iou / iou_fg / hd95 / num_slices
        }
        self._print(summary, efficiency, summary_metrics)
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    def _save_pred(self, pred: torch.Tensor, pid: str, sname: str):
        import cv2
        d = self.pred_dir / pid
        d.mkdir(parents=True, exist_ok=True)
        if self.is_binary:
            arr = (torch.sigmoid(pred).squeeze().cpu().numpy() > self.threshold).astype(np.uint8) * 255
        else:
            classes = torch.argmax(pred, dim=0).cpu().numpy()
            if self.inverse_mapping:
                arr = np.zeros_like(classes, dtype=np.uint8)
                for remapped, original in self.inverse_mapping.items():
                    arr[classes == int(remapped)] = int(original)
            else:
                arr = classes.astype(np.uint8)
        cv2.imwrite(str(d / sname), arr)

    def _save_slice_metrics(self, slices):
        if not slices:
            return
        # NEW columns (detect_tp/fn/fp/tn) appended AFTER the existing ones.
        # csv.DictReader (used by every existing analysis script -- e.g.
        # compute_regional_metrics.py) reads by column NAME, not position, so
        # this is safe: old scripts that only look up dice_fg/hd95/etc. are
        # completely unaffected by the new columns being present.
        fields = ["patient_id", "slice_name", "dice", "dice_fg", "iou", "iou_fg", "hd95",
                  "detect_tp", "detect_fn", "detect_fp", "detect_tn"]
        with open(self.output_dir / "slice_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for m in slices:
                w.writerow({
                    "patient_id": m["patient_id"], "slice_name": m["slice_name"],
                    "dice": f"{m['dice']:.6f}", "dice_fg": f"{m['dice_fg']:.6f}",
                    "iou": f"{m['iou']:.6f}", "iou_fg": f"{m['iou_fg']:.6f}",
                    "hd95": f"{m['hd95']:.6f}" if np.isfinite(m["hd95"]) else "inf",
                    "detect_tp": m.get("detect_tp", 0), "detect_fn": m.get("detect_fn", 0),
                    "detect_fp": m.get("detect_fp", 0), "detect_tn": m.get("detect_tn", 0),
                })
    @staticmethod
    def _print(summary, eff, m):
        print("\n" + "=" * 64)
        print(f"Variant: {summary['variant']}  |  classes: {summary['num_classes']}")
        print(f"Params: {eff['params_readable']}  FLOPs: {eff['flops_readable']}  "
              f"Infer: {eff['inference_time_ms']:.2f} ms")
        print("-" * 64)
        for k in ("dice", "dice_fg", "iou", "iou_fg", "hd95"):
            print(f"{k:<8}: {m[k]['mean']:.4f} ± {m[k]['std']:.4f}")
        print("=" * 64)