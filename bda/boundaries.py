# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Binary contour helpers that do not invent background outside a crop."""

import torch
import torch.nn.functional as F


def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if mask.ndim != 3 or mask.dtype != torch.bool:
        raise ValueError("Expected a boolean (batch, height, width) mask")
    if type(radius) is not int or radius < 0:
        raise ValueError("Radius must be a nonnegative integer")
    if radius == 0:
        return mask
    return F.max_pool2d(
        mask.unsqueeze(1).float(), 2 * radius + 1, stride=1, padding=radius,
    ).squeeze(1) > 0


def erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    # Pooling ignores positions outside the image rather than treating them as zeros.
    return ~dilate(~mask, radius)


def inner_boundary(mask: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if valid.shape != mask.shape or valid.dtype != torch.bool:
        raise ValueError("Validity mask must match the boolean class mask")
    return mask & ~erode(mask, 1) & erode(valid, 1)


def boundary_band(
    target: torch.Tensor, class_index: int = 1, radius: int = 3,
    ignore_index: int | None = 255,
) -> torch.Tensor:
    """Band on both sides of the selected class's observed ground-truth contour."""
    if target.ndim != 3 or target.is_floating_point():
        raise ValueError("Expected integer (batch, height, width) target labels")
    if type(class_index) is not int or class_index < 0 or class_index == ignore_index:
        raise ValueError("Boundary class must be a nonnegative, nonignored class")
    valid = target != ignore_index if ignore_index is not None else torch.ones_like(target, dtype=torch.bool)
    selected = (target == class_index) & valid
    return (dilate(selected, radius) ^ erode(selected, radius)) & erode(valid, radius)
