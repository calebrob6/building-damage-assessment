# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Render a checkpoint on the frozen validation-panel list, without selecting images."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bda.xview2 import (
    GROUPINGS, build_manifest, load_image, load_mask, read_json, target_for, write_json,
)
from inference_dinov3 import IMAGENET_MEAN, IMAGENET_STD, load_model
from scripts.prepare_any_reference import render_panel


@torch.inference_mode()
def render_checkpoint(checkpoint, panel_manifest, output_dir, device="cuda:0", label=""):
    panel_manifest, checkpoint, output_dir = Path(panel_manifest), Path(checkpoint), Path(output_dir)
    specification = read_json(panel_manifest)
    data_root = Path(specification["data_root"]).resolve()
    current = build_manifest(data_root, include_tier3=True)
    if current["manifest_hash"] != specification["manifest_hash"]:
        raise ValueError("Dataset changed since the visual panel list was frozen")
    validation = {row["image"] for row in current["splits"]["val"]}
    panels = specification["panels"]
    if (not panels or len({row["image"] for row in panels}) != len(panels)
            or any(row["image"] not in validation for row in panels)):
        raise ValueError("Expected unique validation-only panels")
    for panel in panels:
        x, y, width, height = panel["window"]
        if min(x, y) < 0 or min(width, height) <= 0 or x + width > 1024 or y + height > 1024:
            raise ValueError("Invalid fixed display window")
    output_dir.mkdir(parents=True, exist_ok=False)
    target_device = torch.device(device)
    model = load_model(checkpoint, target_device)
    lut = np.asarray(GROUPINGS["any"], dtype=np.uint8)
    outputs = []
    for offset in range(0, len(panels), 2):
        batch_panels = panels[offset:offset + 2]
        images = [load_image(data_root / panel["image"]) for panel in batch_panels]
        targets = [lut[load_mask(target_for(data_root / panel["image"]))] for panel in batch_panels]
        batch = np.stack([
            ((image - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy()
            for image in images
        ])
        logits = model(torch.from_numpy(batch).to(target_device))
        if not logits.isfinite().all():
            raise ValueError("Nonfinite qualitative-rendering logits")
        predictions = logits.argmax(1).cpu().numpy()
        for panel, image, target, prediction in zip(batch_panels, images, targets, predictions):
            output = output_dir / (Path(panel["image"]).stem + ".png")
            render_panel(
                image, target, prediction, panel["window"],
                f"{label or checkpoint.parent.parent.name} | {panel['category']} | {Path(panel['image']).name}",
                output,
            )
            outputs.append(str(output))
    with checkpoint.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    write_json(output_dir / "rendering.json", {
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": checksum,
        "panel_manifest": str(panel_manifest.resolve()), "manifest_hash": current["manifest_hash"],
        "inference": "full 1024-pixel image, float32; crop only for display",
        "label": label, "outputs": outputs,
    })
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    outputs = render_checkpoint(
        args.checkpoint, args.panel_manifest, args.output_dir, args.device, args.label,
    )
    print(f"Saved {len(outputs)} fixed validation panels to {args.output_dir}")


if __name__ == "__main__":
    main()
