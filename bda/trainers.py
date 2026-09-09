# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Custom torchgeo trainers."""

from contextlib import nullcontext
from typing import Any
import torch
from torch import Tensor
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from torchgeo.trainers import SemanticSegmentationTask
import kornia.augmentation as K

from .boundary_metrics import SegmentationGeometryMetrics
from .losses import build_loss


class CustomSemanticSegmentationTask(SemanticSegmentationTask):
    """A custom trainer for semantic segmentation tasks."""

    def __init__(
        self, *args, use_constraint_loss=False, loss_options=None,
        geometry_metrics=False, validation_precision="mixed", **kwargs,
    ):
        if loss_options is not None and not isinstance(loss_options, dict):
            raise ValueError("loss_options must be a dictionary")
        if type(geometry_metrics) is not bool or validation_precision not in ("mixed", "fp32"):
            raise ValueError("Invalid geometry or validation-precision configuration")
        if use_constraint_loss and (
            kwargs.get("loss", "ce") not in ("ce", "dice") or loss_options or geometry_metrics
        ):
            raise ValueError("The new loss/geometry options cannot be combined with constraint loss")
        self.loss_options = dict(loss_options or {})
        self.track_geometry = geometry_metrics
        self.validation_precision = validation_precision
        self.last_validation_summary = None
        if "ignore" in kwargs:
            del kwargs[
                "ignore"
            ]  # workaround for https://github.com/microsoft/torchgeo/pull/2314, can be removed with torchgeo 0.7
        super().__init__(*args, **kwargs)

        self.use_constraint_loss = use_constraint_loss
        self.save_hyperparameters({
            "loss_options": self.loss_options,
            "geometry_metrics": geometry_metrics,
            "validation_precision": validation_precision,
        })
        self.geometry_tracker = (
            SegmentationGeometryMetrics(
                num_classes=self.hparams["num_classes"], class_index=1,
                tolerance=2, ignore_index=self.hparams["ignore_index"],
            ) if geometry_metrics else None
        )

        self.train_augs = K.AugmentationSequential(
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=None,
            keepdim=True,
        )

    def configure_callbacks(self) -> list[Callback]:
        """Configures the callbacks for the trainer.

        Returns:
            an empty list to override the default callbacks, we set these in the Trainer
        """
        return []

    def configure_models(self) -> None:
        """Initialize the model.

        Adds a ``"upernet"`` option (DINOv3 ViT backbone + UPerNet decode head)
        on top of the ``segmentation_models_pytorch`` models that torchgeo's
        :class:`~torchgeo.trainers.SemanticSegmentationTask` supports. Because the
        architecture is rebuilt from the saved hyperparameters, ``upernet``
        checkpoints load transparently in ``inference.py``.
        """
        if self.hparams["model"] == "upernet":
            from .dinov3_upernet import DINOv3UPerNet

            self.model = DINOv3UPerNet(
                backbone=self.hparams["backbone"],
                in_channels=self.hparams["in_channels"],
                num_classes=self.hparams["num_classes"],
                pretrained=self.weights is True,
            )
            if self.hparams["freeze_backbone"]:
                for param in self.model.backbone.parameters():
                    param.requires_grad = False
        else:
            super().configure_models()

    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        self.criterion = build_loss(
            self.hparams["loss"], class_weights=self.hparams["class_weights"],
            ignore_index=self.hparams["ignore_index"],
            num_classes=self.hparams["num_classes"], options=self.loss_options,
        )

    def _log_components(self, prefix, batch_size, loss=None):
        components = getattr(self.criterion, "last_components", {})
        if self.track_geometry and self.hparams["loss"] == "ce" and loss is not None:
            components = {"ce": loss.detach()}
        if components:
            self.log_dict(
                {f"{prefix}_{name}": value for name, value in components.items()},
                batch_size=batch_size,
            )

    def on_validation_epoch_start(self):
        if self.track_geometry:
            self.geometry_tracker.reset()
            self.last_validation_summary = None

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch["image"], batch["mask"]
        context = (
            torch.autocast(device_type=x.device.type, enabled=False)
            if self.validation_precision == "fp32" else nullcontext()
        )
        with context:
            if not self.track_geometry:
                if self.validation_precision == "fp32":
                    batch = {**batch, "image": x.float()}
                result = super().validation_step(batch, batch_idx, dataloader_idx)
                self._log_components("val", x.shape[0])
                return result
            logits = self(x.float() if self.validation_precision == "fp32" else x)
            loss = self.criterion(logits, y)
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise ValueError("Nonfinite validation logits or loss")
        self.log("val_loss", loss, batch_size=x.shape[0])
        predictions = logits.argmax(1)
        self.val_metrics(predictions, y)
        self.log_dict(self.val_metrics, batch_size=x.shape[0])
        self.geometry_tracker.update(predictions, y)
        self._log_components("val", x.shape[0], loss)

    def on_validation_epoch_end(self):
        if self.track_geometry:
            scores = self.geometry_tracker.compute()
            self.last_validation_summary = {
                "confusion_matrix": self.geometry_tracker.confusion.detach().cpu().tolist(),
                "geometry": self.geometry_tracker.to_dict(),
            }
            self.log_dict(
                {f"val_{name}": value for name, value in scores.items()},
                on_step=False, on_epoch=True, batch_size=1,
            )

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the training loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss tensor.
        """
        batch = self.train_augs(batch)
        x = batch["image"]
        y = batch["mask"]

        batch_size = x.shape[0]
        y_hat = self(x)

        if self.use_constraint_loss:
            ce_loss = F.cross_entropy(y_hat, y, ignore_index=0, reduction="none")
            standard_mask = (y > 0) & (y != 5)
            loss = ce_loss[standard_mask].mean()

            constraint_mask = y == 5
            if constraint_mask.any():
                probs = F.softmax(y_hat, dim=1)
                penalty = probs[:, 3, :, :][constraint_mask]
                constraint_loss = penalty.mean()
                loss = loss + constraint_loss
        else:
            loss = self.criterion(y_hat, y)

        if (self.track_geometry or self.hparams["loss"] not in ("ce", "dice")) and not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        self.log("train_loss", loss, batch_size=batch_size)
        self._log_components("train", batch_size, loss)
        self.train_metrics(y_hat, y)
        self.log_dict(self.train_metrics, batch_size=batch_size)
        return loss
