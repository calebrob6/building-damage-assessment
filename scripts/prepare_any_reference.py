# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Measure the archived any-model geometry and freeze GT-selected visual panels."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bda.boundaries import inner_boundary
from bda.boundary_metrics import SegmentationGeometryMetrics
from bda.xview2 import (
    GROUPINGS, build_manifest, check_manifest, load_image, load_mask,
    read_json, split_paths, target_for, write_json,
)
from inference_dinov3 import IMAGENET_MEAN, IMAGENET_STD, load_model


def select_panels(data_root, filenames, per_category=3):
    """Use only GT connected components; these are not instance annotations."""
    data_root = Path(data_root).resolve()
    records = []
    for filename in filenames:
        mask = load_mask(target_for(filename)) == 1
        labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        sizes = np.bincount(labels.ravel())[1:]
        identifiers = np.flatnonzero(sizes >= 4) + 1
        if not len(identifiers):
            continue
        areas = sizes[identifiers - 1]
        centers = np.asarray(ndimage.center_of_mass(mask, labels, identifiers))[:, ::-1]
        records.append({
            "image": Path(filename).relative_to(data_root).as_posix(),
            "components": len(areas), "median_area": float(np.median(areas)),
            "max_area": int(areas.max()), "areas": areas, "centers": centers,
        })
    categories = {
        "small": sorted(records, key=lambda row: (row["median_area"], row["image"])),
        "large": sorted(records, key=lambda row: (-row["max_area"], row["image"])),
        "crowded": sorted(records, key=lambda row: (-row["components"], row["image"])),
        "isolated": sorted(
            [row for row in records if row["components"] <= 5],
            key=lambda row: (row["components"], -row["max_area"], row["image"]),
        ),
    }
    chosen, used = [], set()
    for category, rows in categories.items():
        count = 0
        for row in rows:
            if row["image"] in used:
                continue
            if category == "crowded":
                center = np.median(row["centers"], axis=0)
            else:
                index = (int(np.argmax(row["areas"])) if category in ("large", "isolated")
                         else int(np.argmin(np.abs(row["areas"] - row["median_area"]))))
                center = row["centers"][index]
            x, y = (int(np.clip(round(value) - 256, 0, 512)) for value in center)
            chosen.append({
                "image": row["image"], "category": category, "window": [x, y, 512, 512],
                "gt_class1_components": row["components"],
                "gt_median_component_area": row["median_area"],
            })
            used.add(row["image"])
            count += 1
            if count == per_category:
                break
        if count != per_category:
            raise ValueError(f"Only {count} suitable {category} panels; expected {per_category}")
    return chosen


def render_panel(image_rgb, target, prediction, window, label, output_path):
    """Full-image predictions are cropped only for display, never before inference."""
    x, y, width, height = window
    image = np.asarray(image_rgb, dtype=np.uint8)[y:y + height, x:x + width]
    truth = np.asarray(target)[y:y + height, x:x + width]
    predicted = np.asarray(prediction)[y:y + height, x:x + width]
    palette = np.array([[0, 0, 0], [0, 210, 0], [240, 0, 0]], dtype=np.uint8)

    def overlay(mask):
        output = image.copy()
        foreground = (mask == 1) | (mask == 2)
        output[foreground] = (
            0.6 * image[foreground] + 0.4 * palette[mask[foreground]]
        ).astype(np.uint8)
        return output

    valid = torch.from_numpy((truth != 255)[None])
    gt_edge = inner_boundary(torch.from_numpy((truth == 1)[None]), valid)[0].numpy()
    pred_edge = inner_boundary(torch.from_numpy((predicted == 1)[None]), valid)[0].numpy()
    contours = image.copy()
    contours[gt_edge] = (0, 255, 255)
    contours[pred_edge] = (255, 0, 255)
    contours[gt_edge & pred_edge] = (255, 255, 255)
    canvas = Image.new("RGB", (4 * width, height + 44), "black")
    for index, panel in enumerate((image, overlay(truth), overlay(predicted), contours)):
        canvas.paste(Image.fromarray(panel), (index * width, 44))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), label, fill="white")
    for index, title in enumerate(("RGB", "GT: green=undamaged, red=damaged",
                                   "Prediction", "Class 1 contours: GT cyan, prediction magenta")):
        draw.text((index * width + 8, 24), title, fill="white")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


@torch.inference_mode()
def main():
    old_run = ROOT / "outputs/xview2_dinov3_upernet_tier3_lr_sweep/run_001"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xview2-root", type=Path, default=Path.home() / "data/xview2")
    parser.add_argument("--manifest", type=Path, default=old_run / "manifest.json")
    parser.add_argument("--reference-metrics", type=Path, default=old_run / "any/lr_3e-05/val_metrics.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    check_manifest(manifest, build_manifest(args.xview2_root, include_tier3=True))
    reference = read_json(args.reference_metrics)
    if (reference["grouping"] != "any" or reference["split"] != "val"
            or reference["n_val_images"] != 279 or reference["manifest_hash"] != manifest["manifest_hash"]):
        raise ValueError("Expected the archived any-model validation reference")
    filenames = split_paths(args.xview2_root, manifest)["val"]
    panels = select_panels(args.xview2_root, filenames)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    panel_manifest = {
        "schema_version": 1, "manifest_hash": manifest["manifest_hash"],
        "selection": "ground-truth class-1 connected components; predictions not used",
        "data_root": str(args.xview2_root.resolve()), "panels": panels,
    }
    write_json(args.output_dir / "panels.json", panel_manifest)
    selected = {row["image"]: row for row in panels}
    device = torch.device(args.device)
    model = load_model(Path(reference["checkpoint"]), device)
    metric = SegmentationGeometryMetrics().to(device)
    lut = np.asarray(GROUPINGS["any"], dtype=np.uint8)
    started = time.monotonic()
    for offset in tqdm(range(0, len(filenames), 2), desc="Reference validation"):
        batch_names = filenames[offset:offset + 2]
        raw_images = [load_image(name) for name in batch_names]
        targets = np.stack([lut[load_mask(target_for(name))] for name in batch_names])
        images = np.stack([
            ((image - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy()
            for image in raw_images
        ])
        logits = model(torch.from_numpy(images).to(device))
        if not logits.isfinite().all():
            raise ValueError("Nonfinite reference logits")
        predicted = logits.argmax(1)
        metric.update(predicted, torch.from_numpy(targets).to(device))
        masks = predicted.cpu().numpy()
        for name, image, target, prediction in zip(batch_names, raw_images, targets, masks):
            relative = Path(name).relative_to(args.xview2_root.resolve()).as_posix()
            if relative in selected:
                panel = selected[relative]
                render_panel(
                    image, target, prediction, panel["window"],
                    f"Archived any 15-epoch reference | {panel['category']} | {Path(name).name}",
                    args.output_dir / "panels" / f"{Path(name).stem}.png",
                )
    confusion = metric.confusion.cpu().tolist()
    if confusion != reference["confusion_matrix"]:
        raise RuntimeError("Reference pixel confusion matrix changed; investigate before training")
    with Path(reference["checkpoint"]).open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {
        **reference, "geometry": metric.to_dict(), "checkpoint_sha256": checksum,
        "reference_metrics_source": str(args.reference_metrics.resolve()),
        "panel_manifest": str((args.output_dir / "panels.json").resolve()),
        "geometry_class": 1, "boundary_tolerance": 2,
        "boundary_distance": "chebyshev", "precision": "float32",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "geometry_evaluation_seconds": time.monotonic() - started,
    }
    write_json(args.output_dir / "reference_metrics.json", result)
    print("Geometry:", result["geometry"])
    print("Panels:", len(panels))
    print("Saved:", args.output_dir / "reference_metrics.json")


if __name__ == "__main__":
    main()
