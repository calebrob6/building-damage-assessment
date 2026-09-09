# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Explicit, AMP-safe segmentation criteria for the any-damage shape study."""

import math

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

from .boundaries import boundary_band

LOSS_NAMES = (
    "ce", "dice", "ce_dice", "boundary_ce", "boundary_ce_dice",
    "focal_dice", "ce_tversky",
)


def _number(options, name, default, minimum=0):
    value = options.get(name, default)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < minimum):
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return float(value)


def _class_weights(weights, num_classes):
    if weights is not None:
        if (not isinstance(weights, torch.Tensor) or weights.shape != (num_classes,)
                or not weights.is_floating_point() or not torch.isfinite(weights).all()
                or (weights < 0).any() or weights.sum() <= 0):
            raise ValueError("Class weights must be a finite nonnegative vector with positive sum")
    return weights


class CompoundSegmentationLoss(nn.Module):
    def __init__(self, name, class_weights, ignore_index, num_classes, options):
        super().__init__()
        self.name = name
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.register_buffer("class_weights", None if class_weights is None else class_weights.detach().clone())
        self.last_components = {}
        has_boundary = name.startswith("boundary_")
        has_region = name.endswith("_dice") or name == "ce_tversky"
        allowed = set()
        if has_boundary:
            allowed.update(("boundary_class", "boundary_radius", "boundary_multiplier"))
        if has_region:
            allowed.update(("foreground_classes", "region_weight"))
        if name == "focal_dice":
            allowed.add("focal_gamma")
        if name == "ce_tversky":
            allowed.update(("tversky_alpha", "tversky_beta"))
        if set(options) - allowed:
            raise ValueError(f"Unsupported options for {name}: {sorted(set(options) - allowed)}")
        self.boundary_class = options.get("boundary_class", 1)
        self.boundary_radius = options.get("boundary_radius", 3)
        self.boundary_multiplier = _number(options, "boundary_multiplier", 4, minimum=1)
        if has_boundary:
            if (type(self.boundary_class) is not int or not 0 <= self.boundary_class < num_classes
                    or self.boundary_class == ignore_index):
                raise ValueError("Boundary class must be a valid nonignored class")
            if type(self.boundary_radius) is not int or self.boundary_radius < 0:
                raise ValueError("Boundary radius must be a nonnegative integer")
        self.region_weight = _number(options, "region_weight", 1)
        self.focal_gamma = _number(options, "focal_gamma", 2)
        self.region = None
        if has_region:
            classes = options.get("foreground_classes", [1, 2])
            if (not isinstance(classes, (list, tuple)) or not classes
                    or any(type(c) is not int or not 0 <= c < num_classes or c == ignore_index for c in classes)
                    or len(set(classes)) != len(classes)):
                raise ValueError("Foreground classes must be distinct valid nonignored class indices")
            parameters = {
                "mode": "multiclass", "classes": list(classes),
                "from_logits": True, "ignore_index": ignore_index, "eps": 1e-7,
            }
            if name == "ce_tversky":
                alpha = _number(options, "tversky_alpha", 0.7)
                beta = _number(options, "tversky_beta", 0.3)
                if not math.isclose(alpha + beta, 1, abs_tol=1e-7):
                    raise ValueError("Tversky false-positive/false-negative weights must sum to one")
                self.region = smp.losses.TverskyLoss(**parameters, alpha=alpha, beta=beta, gamma=1)
            else:
                self.region = smp.losses.DiceLoss(**parameters)

    def forward(self, logits, target):
        if (logits.ndim != 4 or logits.shape[1] != self.num_classes
                or target.shape != (logits.shape[0], *logits.shape[2:])
                or target.is_floating_point()):
            raise ValueError("Expected (N,C,H,W) logits and matching integer (N,H,W) labels")
        valid = target != self.ignore_index if self.ignore_index is not None else torch.ones_like(target, dtype=torch.bool)
        if ((target[valid] < 0) | (target[valid] >= self.num_classes)).any():
            raise ValueError("Nonignored target labels are outside the class range")
        if not torch.isfinite(logits).all():
            raise ValueError("Loss requires finite logits")
        with torch.autocast(device_type=logits.device.type, enabled=False):
            values = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
            safe_target = target.masked_fill(~valid, 0).long()
            nll = -F.log_softmax(values, dim=1).gather(1, safe_target.unsqueeze(1)).squeeze(1)
            weights = (torch.ones_like(nll) if self.class_weights is None
                       else self.class_weights.to(values.dtype)[safe_target])
            weights = weights * valid
            denominator = weights.sum().clamp_min(torch.finfo(values.dtype).eps)
            ce = (nll * weights).sum() / denominator
            components = {"ce": ce}
            if self.name.startswith("boundary_"):
                band = boundary_band(target, self.boundary_class, self.boundary_radius, self.ignore_index)
                spatial = 1 + (self.boundary_multiplier - 1) * band.to(values.dtype)
                combined_weights = weights * spatial
                primary = (nll * combined_weights).sum() / combined_weights.sum().clamp_min(torch.finfo(values.dtype).eps)
                components["boundary_ce"] = primary
            elif self.name == "focal_dice":
                primary = (nll * (1 - (-nll).exp()).pow(self.focal_gamma) * weights).sum() / denominator
                components["focal"] = primary
            else:
                primary = ce
            if self.region is not None:
                region = self.region(values, target.long()) if self.region_weight else ce * 0
                components["tversky" if self.name == "ce_tversky" else "dice"] = region
                total = primary + self.region_weight * region
            else:
                total = primary
        if not torch.isfinite(total):
            raise ValueError("Nonfinite segmentation loss")
        self.last_components = {key: value.detach() for key, value in components.items()}
        return total


def build_loss(name, *, class_weights=None, ignore_index=None, num_classes=3, options=None):
    if name not in LOSS_NAMES:
        raise ValueError(f"Unknown loss {name!r}; choose one of {LOSS_NAMES}")
    if options is not None and not isinstance(options, dict):
        raise ValueError("Loss options must be a dictionary")
    options = dict(options or {})
    weights = _class_weights(class_weights, num_classes)
    if name in ("ce", "dice"):
        if options:
            raise ValueError(f"Legacy {name} does not accept additional loss options")
        if name == "ce":
            return nn.CrossEntropyLoss(
                ignore_index=-1000 if ignore_index is None else ignore_index, weight=weights,
            )
        return smp.losses.DiceLoss(mode="multiclass", ignore_index=ignore_index)
    return CompoundSegmentationLoss(name, weights, ignore_index, num_classes, options)
