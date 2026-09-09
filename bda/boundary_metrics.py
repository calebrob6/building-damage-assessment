# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Split-level geometry metrics for three-class xView2 segmentation."""

import math

import torch
from torchmetrics import Metric

from .boundaries import dilate, erode, inner_boundary


def _ratio(numerator, denominator):
    numerator, denominator = numerator.double(), denominator.double()
    return torch.where(
        denominator > 0, numerator / denominator,
        torch.full_like(numerator, float("nan")),
    )


class SegmentationGeometryMetrics(Metric):
    """Aggregate contours with a Chebyshev matching radius; empty folds are undefined."""

    full_state_update = False

    def __init__(self, num_classes=3, class_index=1, tolerance=2, ignore_index=255):
        super().__init__()
        if num_classes != 3:
            raise ValueError("xView2 geometry metrics require exactly three classes")
        if class_index != 1:
            raise ValueError("This study measures undamaged class 1 boundaries")
        if type(tolerance) is not int or tolerance < 0:
            raise ValueError("Boundary tolerance must be a nonnegative integer")
        if ignore_index in (0, 1, 2):
            raise ValueError("All three xView2 classes must remain valid")
        self.num_classes = num_classes
        self.class_index = class_index
        self.tolerance = tolerance
        self.ignore_index = ignore_index
        self.add_state("confusion", default=torch.zeros(3, 3, dtype=torch.long), dist_reduce_fx="sum")
        for name in ("pred_boundary_count", "true_boundary_count", "pred_matches", "true_matches"):
            self.add_state(name, default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, predictions, target):
        if predictions.ndim == 4:
            if predictions.shape[1] != 3 or not torch.isfinite(predictions).all():
                raise ValueError("Expected finite three-class logits")
            predictions = predictions.argmax(1)
        if (predictions.shape != target.shape or target.ndim != 3
                or predictions.is_floating_point() or target.is_floating_point()):
            raise ValueError("Expected matching integer (batch, height, width) labels")
        valid = (target != self.ignore_index if self.ignore_index is not None
                 else torch.ones_like(target, dtype=torch.bool))
        truth, prediction = target[valid].long(), predictions[valid].long()
        if ((truth < 0) | (truth >= 3) | (prediction < 0) | (prediction >= 3)).any():
            raise ValueError("Nonignored labels must be in [0, 2]")
        self.confusion += torch.bincount(
            truth * 3 + prediction, minlength=9,
        ).reshape(3, 3)
        # Exclude unknown neighborhoods so matches cannot cross an ignored strip.
        safe = erode(valid, self.tolerance + 1)
        predicted_edge = inner_boundary((predictions == 1) & valid, valid) & safe
        true_edge = inner_boundary((target == 1) & valid, valid) & safe
        self.pred_boundary_count += predicted_edge.sum()
        self.true_boundary_count += true_edge.sum()
        self.pred_matches += (predicted_edge & dilate(true_edge, self.tolerance)).sum()
        self.true_matches += (true_edge & dilate(predicted_edge, self.tolerance)).sum()

    def compute(self):
        conf = self.confusion
        pred1, true1, tp1 = conf[:, 1].sum(), conf[1].sum(), conf[1, 1]
        pred2, true2, tp2 = conf[:, 2].sum(), conf[2].sum(), conf[2, 2]
        precision = _ratio(self.pred_matches, self.pred_boundary_count)
        recall = _ratio(self.true_matches, self.true_boundary_count)
        p, r = torch.nan_to_num(precision), torch.nan_to_num(recall)
        f1 = 2 * p * r / (p + r).clamp_min(torch.finfo(torch.float64).eps)
        f1 = torch.where(
            self.pred_boundary_count + self.true_boundary_count > 0,
            f1, torch.full_like(f1, float("nan")),
        )
        union_tp = conf[1:, 1:].sum()
        return {
            "undamaged_boundary_precision": precision,
            "undamaged_boundary_recall": recall,
            "undamaged_boundary_f1": f1,
            "undamaged_iou": _ratio(tp1, pred1 + true1 - tp1),
            "undamaged_precision": _ratio(tp1, pred1),
            "undamaged_recall": _ratio(tp1, true1),
            "undamaged_area_ratio": _ratio(pred1, true1),
            "building_union_iou": _ratio(union_tp, conf[1:].sum() + conf[:, 1:].sum() - union_tp),
            "damaged_f1": _ratio(2 * tp2, pred2 + true2),
        }

    def to_dict(self):
        result = {}
        for key, tensor in self.compute().items():
            value = float(tensor.detach().cpu())
            result[key] = value if math.isfinite(value) else None
        return result
