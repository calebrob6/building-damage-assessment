# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Train DINOv3 + a fresh UPerNet head on xView2, with uniform random crops.

Original-only train/val membership and automatic test scoring remain the CLI
default. Add --include-tier3 --evaluation-split val for validation-only trials.
Use --eval-only --checkpoint PATH --evaluation-split test for selected winners.
--resume restores the optimizer, scheduler, scaler, epoch and step from last.ckpt.
Targets are checked, never clipped; supplied unclassified encoding is unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset, Sampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bda.trainers import CustomSemanticSegmentationTask
from bda.xview2 import (
    CLASS_NAMES, GROUPINGS, build_manifest, check_manifest, class_weight_estimate,
    digest, load_image, load_mask, post_images as _post_images, read_json,
    split_paths, target_for as _target_for, validate_manifest, write_json,
)
from bda.experiment_runner import durable_json, sync_file

NUM_CLASSES = 3
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0


class XView2SegDataset(Dataset):
    """Uniform random training windows; full-image validation and test scoring."""

    def __init__(self, image_fns, grouping, crop_size=512, crops_per_image=1,
                 train=True, seed=0):
        self.image_fns = image_fns
        self.lut = np.asarray(GROUPINGS[grouping], dtype=np.uint8)
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image if train else 1
        self.train = train
        self.seed = seed
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.image_fns) * self.crops_per_image

    def _load(self, idx):
        image_fn = self.image_fns[idx % len(self.image_fns)]
        return load_image(image_fn), self.lut[load_mask(_target_for(image_fn))].astype(np.int64)

    def _random_crop(self, image, mask):
        h, w = mask.shape
        cs = self.crop_size
        if h <= cs or w <= cs:
            return image, mask
        # The original loop always returned attempt 0: preserve uniform sampling.
        y, x = self.rng.randint(0, h - cs), self.rng.randint(0, w - cs)
        return image[y:y + cs, x:x + cs], mask[y:y + cs, x:x + cs]

    def __getitem__(self, idx):
        epoch = None
        if isinstance(idx, tuple):
            epoch, idx = idx
        image, mask = self._load(idx)
        if self.train and self.crop_size is not None:
            if epoch is None:
                image, mask = self._random_crop(image, mask)
            else:
                previous = self.rng
                self.rng = random.Random(f"{self.seed}:{epoch}:{idx}")
                try:
                    image, mask = self._random_crop(image, mask)
                finally:
                    self.rng = previous
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "image": torch.from_numpy(image.transpose(2, 0, 1).copy()).float(),
            "mask": torch.from_numpy(mask.copy()).long(),
        }


class EpochSampler(Sampler):
    """Epoch-indexed permutations/crops do not depend on worker prefetch timing."""

    def __init__(self, dataset, seed=0):
        self.dataset, self.seed, self.epoch = dataset, seed, 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        epoch = self.epoch
        generator = torch.Generator().manual_seed(self.seed + epoch)
        for index in torch.randperm(len(self), generator=generator).tolist():
            yield epoch, index


class EpochDataLoader(DataLoader):
    def __iter__(self):
        self.generator.manual_seed(self.sampler.seed + self.sampler.epoch)
        return super().__iter__()

    def state_dict(self):
        return {"next_epoch": self.sampler.epoch + 1,
                "generator": self.generator.get_state(),
                "private_rng": self.dataset.rng.getstate()}

    def load_state_dict(self, state):
        self.sampler.set_epoch(state["next_epoch"])
        self.generator.set_state(state["generator"])
        self.dataset.rng.setstate(state["private_rng"])


class ReproducibilityState(Callback):
    """Plain/Tensor state only, compatible with torch.load(weights_only=True)."""

    def __init__(self, train_loader, val_loader, audit_path=None):
        self.train_loader, self.val_loader = train_loader, val_loader
        self.pending = None
        self.audit_path = Path(audit_path) if audit_path else None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx or self.audit_path is None:
            return
        def tensor_hash(tensor):
            return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        numpy_state = np.random.get_state()
        row = {
            "epoch": int(trainer.current_epoch), "global_step": int(trainer.global_step),
            "first_images_sha256": tensor_hash(batch["image"]),
            "first_targets_sha256": tensor_hash(batch["mask"]),
            "python_rng": digest(random.getstate()), "torch_rng": tensor_hash(torch.get_rng_state()),
            "numpy_rng": digest([numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]]),
            "cuda_rng": tensor_hash(torch.cuda.get_rng_state(trainer.strategy.root_device))
            if trainer.strategy.root_device.type == "cuda" else None,
            "optimizer_step_counts": sorted({
                int(value["step"]) for value in trainer.optimizers[0].state.values() if "step" in value
            }),
            "learning_rates": [float(group["lr"]) for group in trainer.optimizers[0].param_groups],
            "scheduler_last_epoch": trainer.lr_scheduler_configs[0].scheduler.last_epoch,
            "scaler": trainer.precision_plugin.scaler.state_dict()
            if getattr(trainer.precision_plugin, "scaler", None) is not None else None,
        }
        rows = read_json(self.audit_path) if self.audit_path.exists() else []
        existing = next((entry for entry in rows if entry["epoch"] == row["epoch"]), None)
        if existing is not None and existing != row:
            raise ValueError("Resumed epoch input/augmentation RNG audit changed")
        if existing is None:
            rows.append(row)
            durable_json(self.audit_path, rows)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if "optimizer_states" not in checkpoint:
            return
        numpy_state = np.random.get_state()
        state = {
            "python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": [numpy_state[0], torch.tensor(numpy_state[1].astype(np.int64)),
                      int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])],
            "train_loader": self.train_loader.state_dict(),
            "val_generator": self.val_loader.generator.get_state(),
            "cuda": None,
        }
        if trainer.strategy.root_device.type == "cuda":
            state["cuda"] = torch.cuda.get_rng_state(trainer.strategy.root_device)
        checkpoint["xview2_reproducibility"] = state

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if "xview2_reproducibility" not in checkpoint:
            raise ValueError("Study continuation requires complete RNG/data-loader state")
        self.pending = checkpoint["xview2_reproducibility"]

    def _restore_global(self, trainer):
        state = self.pending
        random.setstate(state["python"])
        torch.set_rng_state(state["torch"].cpu())
        npstate = state["numpy"]
        np.random.set_state((npstate[0], npstate[1].cpu().numpy().astype(np.uint32),
                             npstate[2], npstate[3], npstate[4]))
        if state["cuda"] is not None:
            if trainer.strategy.root_device.type != "cuda":
                raise ValueError("Cannot restore CUDA study RNG onto CPU")
            torch.cuda.set_rng_state(state["cuda"].cpu(), trainer.strategy.root_device)

    def on_fit_start(self, trainer, pl_module):
        if self.pending:
            self._restore_global(trainer)
            self.train_loader.load_state_dict(self.pending["train_loader"])
            self.val_loader.generator.set_state(self.pending["val_generator"])

    def on_train_start(self, trainer, pl_module):
        if self.pending:
            self._restore_global(trainer)


def seed_worker(worker_id):
    # Avoid duplicating the same private Random stream in every forked worker.
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.utils.data.get_worker_info().dataset.rng.seed(seed)


def compute_class_weights(image_fns, grouping, sample=300):
    weights, counts = class_weight_estimate(image_fns, grouping, sample)
    return torch.tensor(weights, dtype=torch.float32), counts


def confusion_metrics(conf):
    """Rows are true labels; undefined metrics are JSON null, not nonstandard NaN."""
    metrics = {"confusion_matrix": conf.tolist(), "per_class": {}}
    for c, name in enumerate(CLASS_NAMES):
        tp = int(conf[c, c])
        fp = int(conf[:, c].sum()) - tp
        fn = int(conf[c, :].sum()) - tp
        ratio = lambda numerator, denominator: numerator / denominator if denominator else None
        metrics["per_class"][name] = {
            "iou": ratio(tp, tp + fp + fn),
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn), "support": int(conf[c, :].sum()),
        }
    ious = [m["iou"] for m in metrics["per_class"].values() if m["iou"] is not None]
    metrics["mean_iou"] = sum(ious) / len(ious) if ious else None
    metrics["damaged_f1"] = metrics["per_class"]["damaged"]["f1"]
    metrics["overall_accuracy"] = float(np.trace(conf) / conf.sum()) if conf.sum() else None
    return metrics


@torch.inference_mode()
def evaluate(task, image_fns, grouping, device, batch_size=2, num_workers=4,
             geometry_metrics=False):
    ds = XView2SegDataset(image_fns, grouping, crop_size=None, train=False)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    model = task.model.eval().to(device)
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    tracker = None
    if geometry_metrics:
        from bda.boundary_metrics import SegmentationGeometryMetrics
        tracker = SegmentationGeometryMetrics().to(device)
    for batch in loader:
        logits = model(batch["image"].to(device))
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite evaluation logits")
        y = batch["mask"].numpy().reshape(-1)
        pred = logits.argmax(1).cpu().numpy().reshape(-1)
        conf += np.bincount(y * NUM_CLASSES + pred, minlength=NUM_CLASSES**2).reshape(3, 3)
        if tracker is not None:
            tracker.update(logits, batch["mask"].to(device))
    metrics = confusion_metrics(conf)
    if tracker is not None:
        metrics["geometry"] = tracker.to_dict()
        if not np.array_equal(conf, tracker.confusion.cpu().numpy()):
            raise RuntimeError("Standalone geometry/confusion accumulators disagree")
    return metrics


class EpochHistory(Callback):
    def __init__(self, path):
        self.path = Path(path)
        self.rows = read_json(self.path) if self.path.exists() else []
        self.start = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.start = time.monotonic()

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or self.start is None:
            return
        loss = float(trainer.callback_metrics["val_loss"].detach().cpu())
        if not math.isfinite(loss):
            raise ValueError("Nonfinite validation loss")
        epoch = int(trainer.current_epoch)
        self.rows = [row for row in self.rows if row["epoch"] < epoch]
        self.rows.append({
            "epoch": epoch, "epochs_completed": epoch + 1,
            "global_step": int(trainer.global_step), "val_loss": loss,
            "learning_rates": [float(g["lr"]) for g in trainer.optimizers[0].param_groups],
            "seconds": time.monotonic() - self.start,
        })
        if hasattr(trainer, "strategy"):
            from bda.sweep_resources import process_tree_pss
            self.rows[-1]["process_tree_pss_bytes"] = process_tree_pss(os.getpid())
        write_json(self.path, self.rows)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["xview2_epoch_history"] = self.rows
        if ("optimizer_states" in checkpoint and "lr_schedulers" in checkpoint
                and trainer.validating and self.rows and self.rows[-1]["epoch"] == trainer.current_epoch):
            # Lightning saves at validation end, before the epoch-level plateau
            # update, and skips that update when resuming this completed epoch.
            # Snapshot the pending update without changing the live optimizer.
            for index, config in enumerate(trainer.lr_scheduler_configs):
                if isinstance(config.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler = copy.copy(config.scheduler)
                    scheduler.optimizer = copy.copy(config.scheduler.optimizer)
                    scheduler.optimizer.param_groups = [
                        dict(group) for group in config.scheduler.optimizer.param_groups
                    ]
                    scheduler.step(self.rows[-1]["val_loss"])
                    checkpoint["lr_schedulers"][index] = scheduler.state_dict()
                    optimizer_index = next(
                        i for i, optimizer in enumerate(trainer.optimizers)
                        if optimizer is config.scheduler.optimizer
                    )
                    for saved, updated in zip(
                        checkpoint["optimizer_states"][optimizer_index]["param_groups"],
                        scheduler.optimizer.param_groups,
                    ):
                        saved["lr"] = updated["lr"]
            checkpoint["xview2_pending_plateau_step_included"] = True


def checkpoint_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class StudyCheckpoints(Callback):
    """Publish immutable validation candidates and frozen budget evidence."""

    def __init__(self, output, config, milestones, n_val_images):
        self.output, self.config = Path(output), config
        self.milestones, self.n_val_images = set(milestones), n_val_images
        self.path = self.output / "candidates.json"
        self.rows = read_json(self.path)["candidates"] if self.path.exists() else []
        self.pending_row = None

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["xview2_study"] = {
            "config_hash": digest(self.config), "horizon": self.config["max_epochs"],
            "epochs_completed": (self.pending_row["epochs_completed"] if self.pending_row
                                 else self.rows[-1]["epochs_completed"] if self.rows else 0),
            "pending_candidate": self.pending_row,
        }
        checkpoint["xview2_candidates"] = self.rows

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        study_state = checkpoint["xview2_study"]
        if study_state["config_hash"] != digest(self.config):
            raise ValueError("Study checkpoint configuration mismatch")
        if any(row["epochs_completed"] > study_state["epochs_completed"] for row in self.rows):
            raise ValueError("Refusing a stale continuation behind published candidate evidence")
        for row in checkpoint.get("xview2_candidates", []):
            existing = next((item for item in self.rows if item["epoch"] == row["epoch"]), None)
            if existing is not None and existing != row:
                raise ValueError("Checkpoint and published candidate catalog disagree")
            if not Path(row["checkpoint"]).is_file() or checkpoint_hash(row["checkpoint"]) != row["sha256"]:
                raise ValueError("A previously published candidate checkpoint is missing or changed")
            if existing is None:
                self.rows.append(row)
        pending = study_state.get("pending_candidate")
        if pending:
            self.commit_candidate(pending, saved=checkpoint)
        elif self.rows:
            durable_json(self.path, {"config_hash": digest(self.config), "candidates": self.rows})
        completed = study_state["epochs_completed"]
        if completed in self.milestones and self.rows:
            self.publish_stage(completed, self.output / "checkpoints" / "last.ckpt")

    def commit_candidate(self, row, trainer=None, pl_module=None, saved=None):
        path = Path(row["checkpoint"])
        existing = next((item for item in self.rows if item["epoch"] == row["epoch"]), None)
        if existing:
            if {k: v for k, v in existing.items() if k != "sha256"} != row:
                raise ValueError(f"Immutable candidate changed: epoch {row['epoch']}")
            if checkpoint_hash(path) != existing["sha256"]:
                raise ValueError("Candidate checkpoint hash changed")
            return
        if path.exists():
            candidate = torch.load(path, map_location="cpu", weights_only=True)
            current = saved["state_dict"] if saved is not None else pl_module.state_dict()
            if (candidate.get("epoch") != row["epoch"] or candidate.get("global_step") != row["global_step"]
                    or set(candidate["state_dict"]) != set(current)
                    or any(not torch.equal(value.cpu(), candidate["state_dict"][key].cpu())
                           for key, value in current.items())):
                raise ValueError(f"Uncatalogued candidate conflicts with durable training state: {path}")
        else:
            path.parent.mkdir(exist_ok=True)
            if saved is None:
                trainer.save_checkpoint(path, weights_only=True)
            else:
                payload = {key: saved[key] for key in (
                    "epoch", "global_step", "pytorch-lightning_version", "state_dict",
                    "loops", "hyper_parameters", "hparams_name", "hparams_type",
                ) if key in saved}
                partial = path.with_suffix(".partial")
                torch.save(payload, partial)
                sync_file(partial)
                partial.replace(path)
        sync_file(path)
        self.rows.append({**row, "sha256": checkpoint_hash(path)})
        self.rows.sort(key=lambda item: item["epoch"])
        durable_json(self.path, {"config_hash": digest(self.config), "candidates": self.rows})

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        summary = pl_module.last_validation_summary
        if not summary:
            raise RuntimeError("Geometry-enabled validation did not publish its summary")
        epoch = int(trainer.current_epoch)
        metrics = confusion_metrics(np.asarray(summary["confusion_matrix"], dtype=np.int64))
        metrics["geometry"] = summary["geometry"]
        loss = float(trainer.callback_metrics["val_loss"].detach().cpu())
        if not math.isfinite(loss):
            raise ValueError("Nonfinite candidate validation loss")
        path = self.output / "candidates" / f"epoch_{epoch + 1:03d}.ckpt"
        row = {
            "epoch": epoch, "epochs_completed": epoch + 1, "global_step": int(trainer.global_step),
            "checkpoint": str(path.resolve()), "val_loss": loss, "metrics": metrics,
            "n_val_images": self.n_val_images,
        }
        # Commit a full recovery point first. If interrupted during candidate
        # publication, its exact weights and validation summary can be recovered
        # without replaying a numerically nondeterministic CUDA epoch.
        self.pending_row = row
        last = self.output / "checkpoints" / "last.ckpt"
        last.parent.mkdir(exist_ok=True)
        trainer.save_checkpoint(last, weights_only=False)
        sync_file(last)
        self.commit_candidate(row, trainer=trainer, pl_module=pl_module)
        self.pending_row = None
        if epoch + 1 in self.milestones:
            self.publish_stage(epoch + 1, last)

    def publish_stage(self, completed, last):
        stage = self.output / "stages" / f"epoch_{completed:03d}"
        ready = stage / "ready.json"
        if ready.exists():
            marker = read_json(ready)
            if (marker["config_hash"] != digest(self.config)
                    or checkpoint_hash(marker["continuation_checkpoint"]) != marker["continuation_sha256"]):
                raise ValueError("Immutable stage snapshot changed")
            return
        stage.mkdir(parents=True, exist_ok=True)
        frozen = [row for row in self.rows if row["epochs_completed"] <= completed]
        if [row["epochs_completed"] for row in frozen] != list(range(1, completed + 1)):
            raise ValueError("A stage cannot omit candidate epochs")
        continuation = stage / "last.ckpt"
        if continuation.exists():
            saved = torch.load(continuation, map_location="cpu", weights_only=True)
            if (saved.get("xview2_study", {}).get("config_hash") != digest(self.config)
                    or saved["xview2_study"].get("epochs_completed") != completed
                    or saved.get("global_step") != frozen[-1]["global_step"]
                    or not saved.get("optimizer_states") or not saved.get("lr_schedulers")
                    or "xview2_reproducibility" not in saved):
                raise ValueError(f"Unpublished continuation conflicts with this stage: {continuation}")
            del saved
        else:
            partial = stage / "last.ckpt.partial"
            shutil.copyfile(last, partial)
            sync_file(partial)
            partial.replace(continuation)
            sync_file(continuation)
        catalog = {"config_hash": digest(self.config), "budget": completed, "candidates": frozen}
        durable_json(stage / "catalog.json", catalog, immutable=True)
        durable_json(ready, {
            "schema_version": 1, "status": "ready", "epochs_completed": completed,
            "global_step": frozen[-1]["global_step"], "config_hash": digest(self.config),
            "manifest_hash": self.config["manifest_hash"], "horizon": self.config["max_epochs"],
            "catalog": str((stage / "catalog.json").resolve()), "catalog_hash": digest(catalog),
            "continuation_checkpoint": str(continuation.resolve()),
            "continuation_sha256": checkpoint_hash(continuation),
        }, immutable=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xview2-root", default=os.path.expanduser("~/data/XView2"))
    p.add_argument("--include-tier3", action="store_true")
    p.add_argument("--grouping", choices=list(GROUPINGS), required=True)
    p.add_argument("--backbone", default="dinov3_vits16")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--require-gpu-uuid", help="Require exactly this isolated visible GPU.")
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--crops-per-image", type=int, default=4)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--eval-batch-size", type=int, default=2)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, help="Cap images per split (smoke tests only).")
    p.add_argument("--limit-train-batches", type=int, help="Smoke tests only.")
    p.add_argument("--limit-val-batches", type=int, help="Smoke tests only.")
    p.add_argument("--evaluation-split", choices=["val", "test", "both", "none"], default="both")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--checkpoint", help="Explicit checkpoint for --eval-only.")
    p.add_argument("--resume", action="store_true", help="Restore this output's genuine last.ckpt.")
    p.add_argument("--manifest", help="Shared manifest; must match current dataset metadata.")
    p.add_argument("--preflight-report", help="Previously validated report for this manifest.")
    p.add_argument("--class-weights-json", help="Shared training-only class weights.")
    p.add_argument("--loss", default="ce", choices=[
        "ce", "dice", "ce_dice", "boundary_ce", "boundary_ce_dice", "focal_dice", "ce_tversky",
    ])
    p.add_argument("--loss-options", default="{}", help="Plain JSON object of criterion options.")
    p.add_argument("--geometry-metrics", action="store_true")
    p.add_argument("--validation-precision", choices=["mixed", "fp32"], default="mixed")
    p.add_argument("--epoch-candidates", action="store_true",
                   help="Save immutable full-validation candidates and exact-continuation state.")
    p.add_argument("--milestones", type=int, nargs="+", default=[])
    p.add_argument("--stage-epochs", type=int, help="Active stop budget; --max-epochs remains the total horizon.")
    p.add_argument("--promote-stage", action="store_true",
                   help="Extend a successfully completed stage from its frozen continuation checkpoint.")
    return p


def training_config(args, manifest, weights):
    keys = ("grouping", "backbone", "crop_size", "batch_size", "crops_per_image",
            "max_epochs", "lr", "val_fraction", "num_workers", "eval_batch_size",
            "seed", "include_tier3", "limit", "limit_train_batches", "limit_val_batches",
            "evaluation_split")
    config = {key: getattr(args, key) for key in keys}
    config.update({
        "schema_version": 1, "xview2_root": str(Path(args.xview2_root).expanduser().resolve()),
        "manifest_hash": manifest["manifest_hash"], "class_weights": weights,
        "initialization": "pretrained backbone; fresh UPerNet head",
        "freeze_backbone": False, "crop_sampling": "uniform",
        "precision": "16-mixed", "optimizer": "AdamW",
        "scheduler": {"name": "ReduceLROnPlateau", "patience": 5, "monitor": "val_loss"},
        "loss": "class-weighted cross entropy", "ignore_index": 255,
        "normalization": {"mean": IMAGENET_MEAN.tolist(), "std": IMAGENET_STD.tolist()},
        "augmentation": "RandomRotation(p=0.5,degrees=90), horizontal/vertical flips(p=0.5)",
        "versions": {name: version(name) for name in (
            "torch", "lightning", "torchgeo", "transformers", "kornia", "numpy", "Pillow",
        )},
    })
    if (args.loss != "ce" or args.loss_options or args.geometry_metrics
            or args.validation_precision != "mixed" or args.epoch_candidates):
        config.update({
            "loss": args.loss, "loss_options": args.loss_options,
            "geometry_metrics": args.geometry_metrics, "validation_precision": args.validation_precision,
            "epoch_candidates": args.epoch_candidates, "milestones": sorted(args.milestones),
            "reproducibility": "epoch-indexed sampler/crops; explicit loader and Python/NumPy/Torch/CUDA state",
            "checkpoint_protocol_version": 1,
            "scheduler": {"name": "ReduceLROnPlateau", "factor": 0.1, "patience": 5, "monitor": "val_loss"},
        })
    return config


def score_checkpoint(args, checkpoint, partitions, manifest, device):
    # No pretrained download/reinitialization is needed when restoring all weights.
    task = CustomSemanticSegmentationTask.load_from_checkpoint(
        checkpoint, map_location="cpu", weights=False, weights_only=True,
    )
    if task.hparams["model"] != "upernet" or task.hparams["backbone"] != args.backbone:
        raise ValueError("Evaluation checkpoint architecture/backbone mismatch")
    splits = ["val", "test"] if args.evaluation_split == "both" else [args.evaluation_split]
    results = {}
    for split in splits:
        if split == "none":
            continue
        started = time.monotonic()
        metrics = evaluate(task, partitions[split], args.grouping, device,
                           args.eval_batch_size, args.num_workers, args.geometry_metrics)
        metrics.update({
            "grouping": args.grouping, "backbone": args.backbone, "split": split,
            "checkpoint": str(Path(checkpoint).resolve()),
            f"n_{split}_images": len(partitions[split]),
            "manifest_hash": manifest["manifest_hash"],
            "seconds": time.monotonic() - started, "smoke_test": args.limit is not None,
        })
        write_json(Path(args.output_dir) / f"{split}_metrics.json", metrics)
        results[split] = metrics
        print(f"{split}: damaged_F1={metrics['damaged_f1']} mIoU={metrics['mean_iou']}", flush=True)
    return results


def main():
    args = parser().parse_args()
    args.loss_options = json.loads(args.loss_options)
    if not isinstance(args.loss_options, dict):
        raise ValueError("--loss-options must be a JSON object")
    # Also reject NaN/Infinity and values that cannot round-trip as plain data.
    json.dumps(args.loss_options, allow_nan=False)
    stage_epochs = args.max_epochs if args.stage_epochs is None else args.stage_epochs
    if args.epoch_candidates:
        if (not args.geometry_metrics or args.validation_precision != "fp32"
                or args.evaluation_split not in ("val", "none")
                or not args.milestones or len(set(args.milestones)) != len(args.milestones)
                or any(value < 1 or value > args.max_epochs for value in args.milestones)
                or stage_epochs not in args.milestones):
            raise ValueError("Candidates require fp32 geometry validation, milestones and validation-only scoring")
    elif args.stage_epochs is not None or args.promote_stage or args.milestones:
        raise ValueError("Stage controls require --epoch-candidates")
    if not 1 <= stage_epochs <= args.max_epochs or (args.promote_stage and not args.resume):
        raise ValueError("Invalid stage transition")
    if (args.lr <= 0 or not math.isfinite(args.lr) or args.crop_size <= 0
            or args.crop_size > 1024 or args.crop_size % 32
            or min(args.batch_size, args.eval_batch_size, args.crops_per_image, args.max_epochs) < 1
            or args.num_workers < 0):
        raise ValueError("Invalid training sizes, workers or learning rate")
    for value in (args.limit, args.limit_train_batches, args.limit_val_batches):
        if value is not None and value < 1:
            raise ValueError("Smoke-test limits must be positive")
    if args.eval_only != bool(args.checkpoint) or (args.eval_only and args.resume):
        raise ValueError("--eval-only requires --checkpoint and cannot be combined with --resume")
    if args.require_gpu_uuid:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != args.require_gpu_uuid or args.gpu != 0:
            raise ValueError("GPU isolation mismatch: require one UUID and logical --gpu 0")
    if not torch.cuda.is_available() or args.gpu >= torch.cuda.device_count() or args.gpu < 0:
        raise RuntimeError("Requested CUDA device unavailable; CPU fallback is forbidden")
    if args.require_gpu_uuid and torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA GPU")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    pl.seed_everything(args.seed, workers=True)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # The controller may create the directory and its log/config before starting us.
    if not args.resume and not args.eval_only and (
        (output / "config.json").exists() or (output / "checkpoints").exists()
    ):
        raise FileExistsError("Training artifacts already exist; use --resume or a new output")
    manifest = build_manifest(args.xview2_root, args.include_tier3, args.seed, args.val_fraction)
    if args.manifest:
        check_manifest(read_json(args.manifest), manifest)
    if args.preflight_report:
        report = read_json(args.preflight_report)
        if (report.get("status") != "validated" or report.get("image_size") != [1024, 1024]
                or report.get("target_codes") != [0, 1, 2, 3, 4]
                or report.get("manifest_hash") != manifest["manifest_hash"]
                or report.get("images_validated") != sum(
                    manifest["counts"][s]["total"] for s in ("train", "val", "test"))):
            raise ValueError("Preflight report does not validate this complete manifest")
    else:
        report = validate_manifest(args.xview2_root, manifest)
    write_json(output / "manifest.json", manifest)
    write_json(output / "preflight.json", report)
    partitions = split_paths(args.xview2_root, manifest)
    if args.class_weights_json:
        shared = read_json(args.class_weights_json)
        if shared["manifest_hash"] != manifest["manifest_hash"]:
            raise ValueError("Class weights manifest mismatch")
        weight_record = shared["groupings"][args.grouping]
    else:
        weights, counts = compute_class_weights(partitions["train"], args.grouping)
        weight_record = {"weights": weights.tolist(), "pixel_counts": counts.astype(int).tolist()}
    weights = torch.tensor(weight_record["weights"], dtype=torch.float32)
    if weights.shape != (3,) or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Expected three finite positive class weights")
    if args.limit:
        partitions = {
            "train": partitions["train"][:args.limit],
            "val": partitions["val"][:max(2, args.limit // 5)],
            "test": partitions["test"][:args.limit],
        }
    print(f"grouping={args.grouping} manifest={manifest['manifest_hash']} "
          f"counts={ {k: len(v) for k, v in partitions.items()} } weights={weights.tolist()}", flush=True)
    if args.eval_only:
        if args.evaluation_split == "none":
            raise ValueError("--eval-only requires a scoring split")
        for split in (["val", "test"] if args.evaluation_split == "both" else [args.evaluation_split]):
            if (output / f"{split}_metrics.json").exists():
                raise FileExistsError("Evaluation metrics already exist; use a new output directory")
        write_json(output / "evaluation_config.json", {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "manifest_hash": manifest["manifest_hash"], "grouping": args.grouping,
            "backbone": args.backbone, "evaluation_split": args.evaluation_split,
            "num_workers": args.num_workers, "eval_batch_size": args.eval_batch_size,
            "counts": {key: len(value) for key, value in partitions.items()},
            "gpu_uuid": args.require_gpu_uuid, "limit": args.limit,
            "geometry_metrics": args.geometry_metrics, "validation_precision": args.validation_precision,
        })
        score_checkpoint(args, args.checkpoint, partitions, manifest, device)
        return
    config = training_config(args, manifest, weight_record)
    if args.resume and read_json(output / "config.json") != config:
        raise ValueError("Cannot resume: training configuration mismatch")
    write_json(output / "config.json", config)
    metadata_path = output / "training_state.json"
    previous = read_json(metadata_path) if args.resume and metadata_path.exists() else {}
    promoted_checkpoint = None
    if args.epoch_candidates and args.resume:
        previous_stage = previous.get("stage_epochs")
        if args.promote_stage:
            if (previous.get("status") != "stage_complete" or previous_stage is None
                    or previous.get("epochs_completed") != previous_stage
                    or previous_stage >= stage_epochs or stage_epochs != args.max_epochs):
                raise ValueError("Promotion requires a successfully completed smaller stage")
            marker = read_json(output / "stages" / f"epoch_{previous_stage:03d}" / "ready.json")
            if marker["config_hash"] != digest(config) or marker["epochs_completed"] != previous_stage:
                raise ValueError("Promotion marker/configuration mismatch")
            promoted_checkpoint = marker["continuation_checkpoint"]
            if checkpoint_hash(promoted_checkpoint) != marker["continuation_sha256"]:
                raise ValueError("Frozen continuation checkpoint changed")
        elif previous_stage != stage_epochs:
            raise ValueError("Stage change requires explicit --promote-stage")
    if previous.get("attempts"):
        previous["attempts"][-1].setdefault("status", previous["status"])
        if previous.get("error"):
            previous["attempts"][-1].setdefault("error", previous["error"])
    metadata = {
        "status": "running", "config_hash": digest(config), "manifest_hash": manifest["manifest_hash"],
        "counts": {k: len(v) for k, v in partitions.items()},
        "initial_lr": args.lr, "epochs_requested": args.max_epochs,
        "stage_epochs": stage_epochs,
        "gpu_uuid": args.require_gpu_uuid, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "class_weights": weight_record, "attempts": previous.get("attempts", []),
    }
    attempt = {"started_at": time.time(), "resumed": args.resume}
    metadata["attempts"].append(attempt)
    write_json(metadata_path, metadata)
    started = time.monotonic()
    trainer = None
    try:
        train_ds = XView2SegDataset(partitions["train"], args.grouping, args.crop_size,
                                   args.crops_per_image, train=True, seed=args.seed)
        if len(train_ds) < args.batch_size:
            raise ValueError("drop_last=True would produce zero training batches")
        val_ds = XView2SegDataset(partitions["val"], args.grouping, train=False)
        loader_options = {"batch_size": args.batch_size, "num_workers": args.num_workers,
                          "drop_last": True, "pin_memory": True, "worker_init_fn": seed_worker}
        if args.epoch_candidates:
            train_loader = EpochDataLoader(
                train_ds, sampler=EpochSampler(train_ds, args.seed),
                generator=torch.Generator().manual_seed(args.seed), **loader_options,
            )
        else:
            train_loader = DataLoader(train_ds, shuffle=True, **loader_options)
        val_loader = DataLoader(
            val_ds, batch_size=args.eval_batch_size, num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.seed + 100000) if args.epoch_candidates else None,
        )
        task = CustomSemanticSegmentationTask(
            model="upernet", backbone=args.backbone, weights=True, in_channels=3,
            num_classes=NUM_CLASSES, loss=args.loss, loss_options=args.loss_options, class_weights=weights,
            ignore_index=255, lr=args.lr, patience=5, freeze_backbone=False,
            geometry_metrics=args.geometry_metrics, validation_precision=args.validation_precision,
        )
        task.train()
        metadata["initialization"] = {
            "pretrained_backbone": True,
            "fresh_head": True,
            "restored_last_checkpoint": args.resume and (output / "checkpoints/last.ckpt").is_file(),
            "trainable_backbone_parameters": sum(p.numel() for p in task.model.backbone.parameters()
                                                 if p.requires_grad),
            "trainable_head_parameters": sum(p.numel() for p in task.model.decode_head.parameters()
                                             if p.requires_grad),
        }
        checkpoint_cb = ModelCheckpoint(
            monitor="val_loss", mode="min", dirpath=output / "checkpoints", save_top_k=1,
            save_last=not args.epoch_candidates, save_weights_only=False, save_on_train_epoch_end=False,
            filename="best-{epoch:02d}-{val_loss:.4f}",
        )
        history = EpochHistory(output / "epochs.json")
        callbacks = [history]
        study_checkpoints = None
        if args.epoch_candidates:
            scored_images = min(len(partitions["val"]),
                                args.limit_val_batches * args.eval_batch_size
                                if args.limit_val_batches else len(partitions["val"]))
            study_checkpoints = StudyCheckpoints(output, config, args.milestones, scored_images)
            callbacks += [ReproducibilityState(train_loader, val_loader, output / "rng_epochs.json"),
                          study_checkpoints]
        callbacks.append(checkpoint_cb)
        trainer = pl.Trainer(
            max_epochs=stage_epochs, accelerator="gpu", devices=[args.gpu],
            precision="16-mixed", callbacks=callbacks,
            logger=pl.loggers.CSVLogger(str(output), name="logs"), log_every_n_steps=20,
            limit_train_batches=args.limit_train_batches or 1.0,
            limit_val_batches=args.limit_val_batches or 1.0,
            num_sanity_val_steps=0 if args.epoch_candidates else 2,
        )
        last_path = output / "checkpoints" / "last.ckpt"
        resume_checkpoint = args.resume and last_path.is_file()
        if args.epoch_candidates and args.resume and not resume_checkpoint:
            raise FileNotFoundError("Study resume requires its full last checkpoint")
        resume_path = promoted_checkpoint or (str(last_path) if resume_checkpoint else None)
        if resume_path and args.epoch_candidates:
            saved = torch.load(resume_path, map_location="cpu", weights_only=True)
            if (not saved.get("optimizer_states") or not saved.get("lr_schedulers")
                    or saved.get("xview2_study", {}).get("config_hash") != digest(config)
                    or "xview2_reproducibility" not in saved):
                raise ValueError("Invalid full study continuation checkpoint")
            del saved
        attempt["resume_mode"] = "last_checkpoint" if resume_checkpoint else "pretrained_initialization"
        attempt["resume_checkpoint"] = str(resume_path) if resume_path else None
        if args.promote_stage:
            attempt["promoted_from_stage"] = previous.get("stage_epochs")
        if args.resume and not resume_checkpoint:
            print("No last.ckpt exists: explicitly restarting the failed pre-checkpoint attempt "
                  "from pretrained initialization; previous logs are retained.", flush=True)
        fit_started = time.monotonic()
        try:
            trainer.fit(task, train_dataloaders=train_loader, val_dataloaders=val_loader,
                        ckpt_path=resume_path)
        finally:
            attempt["fit_seconds"] = time.monotonic() - fit_started
        # ModelCheckpoint can leave last.ckpt at validation-end loop progress
        # when the final epoch also improves the best score. Save the actual
        # post-fit optimizer/scheduler/scaler and completed-epoch loop state.
        trainer.save_checkpoint(str(last_path), weights_only=False)
        if args.epoch_candidates:
            sync_file(last_path)
        if not last_path.is_file():
            raise RuntimeError("Training did not produce a resumable last checkpoint")
        last = torch.load(last_path, map_location="cpu", weights_only=True)
        if not last.get("optimizer_states") or not last.get("lr_schedulers"):
            raise RuntimeError("last.ckpt is not a full resumable training checkpoint")
        # Lightning's final on_train_end checkpoint uses epoch=max_epochs,
        # whereas validation checkpoints use the zero-based selected epoch.
        epochs_completed = int(last["loops"]["fit_loop"]["epoch_progress"]["total"]["completed"])
        metadata.update({
            "epochs_completed": epochs_completed, "global_step": int(last["global_step"]),
            "optimizer_step_counts": sorted({
                int(value["step"]) for value in last["optimizer_states"][0]["state"].values()
                if "step" in value
            }),
            "training_batches_per_epoch": int(trainer.num_training_batches),
            "last_checkpoint": str(last_path), "epoch_history": history.rows,
            "final_learning_rates": [float(group["lr"]) for group in trainer.optimizers[0].param_groups],
            "scheduler_state": {
                key: str(value) if isinstance(value, float) and not math.isfinite(value) else value
                for key, value in trainer.lr_scheduler_configs[0].scheduler.state_dict().items()
            },
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        })
        del last
        if epochs_completed != stage_epochs:
            raise RuntimeError(f"Incomplete training stage: {epochs_completed}/{stage_epochs} epochs")
        best = checkpoint_cb.best_model_path
        if study_checkpoints is not None:
            study_checkpoints.publish_stage(stage_epochs, last_path)
            candidate = min(study_checkpoints.rows, key=lambda row: (row["val_loss"], row["epoch"]))
            best = candidate["checkpoint"]
        if not best or not Path(best).is_file():
            raise RuntimeError("No best-validation-loss checkpoint; refusing last-checkpoint fallback")
        best_state = torch.load(best, map_location="cpu", weights_only=True)
        metadata.update({
            "selected_checkpoint": str(Path(best).resolve()),
            "selected_epoch": int(best_state["epoch"]),
            "selected_global_step": int(best_state["global_step"]),
            "best_val_loss": candidate["val_loss"] if study_checkpoints else float(checkpoint_cb.best_model_score.cpu()),
            "selected_checkpoint_policy": (
                "Diagnostic within-recipe best loss; guarded study selection is separate"
                if study_checkpoints else "Lowest validation loss"
            ),
        })
        del best_state
        task.cpu()
        for optimizer in trainer.optimizers:
            optimizer.state.clear()
        torch.cuda.empty_cache()
        results = score_checkpoint(args, best, partitions, manifest, device)
        metadata["evaluation_seconds"] = {k: v["seconds"] for k, v in results.items()}
        metadata["status"] = "stage_complete" if stage_epochs < args.max_epochs else "complete"
    finally:
        error = sys.exc_info()[1]
        if error is not None:
            metadata["status"] = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
            metadata["error"] = f"{type(error).__name__}: {error}"
            if trainer is not None:
                metadata["observed_global_step"] = int(trainer.global_step)
        attempt.update({"finished_at": time.time(), "seconds": time.monotonic() - started,
                        "status": metadata["status"]})
        if trainer is not None:
            attempt["observed_global_step_at_exit"] = int(trainer.global_step)
        if metadata.get("error"):
            attempt["error"] = metadata["error"]
        metadata["total_seconds"] = sum(a["seconds"] for a in metadata["attempts"] if "seconds" in a)
        write_json(metadata_path, metadata)


if __name__ == "__main__":
    main()
