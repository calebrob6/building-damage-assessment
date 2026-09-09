# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Reproducible xView2 splits, strict target validation and training-only weights.

Supplied target encodings (including already-encoded unclassified annotations)
are preserved. Tier3 is appended *after* the historical original-train shuffle.
No labels are regenerated and hold is never used for training or evaluation.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image

GROUPINGS = {
    "any": [0, 1, 2, 2, 2],
    "major": [0, 1, 1, 2, 2],
    "destroyed": [0, 1, 1, 1, 2],
}
CLASS_NAMES = ["background", "undamaged", "damaged"]
SPLITS = ("train", "val", "test")


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def write_json(path, value) -> None:
    """Atomically publish JSON in its destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    partial.replace(path)


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def post_images(root, source: str) -> list[str]:
    files = sorted((Path(root) / source / "images").glob("*_post_disaster.png"))
    if not files:
        raise ValueError(f"No post-disaster images under {root}/{source}/images")
    return [str(path) for path in files]


def target_for(image_fn) -> str:
    path = Path(image_fn)
    return str(path.parent.parent / "targets" / (path.stem + "_target.png"))


def disaster(image_fn) -> str:
    return Path(image_fn).name.split("_")[0]


def _fingerprint(path: Path) -> dict:
    stat = path.stat()
    if not path.is_file():
        raise ValueError(f"Not a regular file: {path}")
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def build_manifest(root, include_tier3=False, seed=0, val_fraction=0.1) -> dict:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    root = Path(root).expanduser().resolve()
    original = post_images(root, "train")
    random.Random(seed).shuffle(original)
    n_val = int(len(original) * val_fraction)
    if n_val == 0 or n_val == len(original):
        raise ValueError("Train and validation partitions must both be nonempty")
    partitions = {"train": original[n_val:], "val": original[:n_val],
                  "test": post_images(root, "test")}
    if include_tier3:
        partitions["train"] += post_images(root, "tier3")
    entries, counts, seen, seen_paths = {}, {}, {}, {}
    for split in SPLITS:
        entries[split] = []
        for filename in partitions[split]:
            image = Path(filename)
            target = Path(target_for(image))
            if image.name in seen:
                raise ValueError(f"Sample overlap: {image.name} in {seen[image.name]} and {split}")
            seen[image.name] = split
            for path in (image, target):
                resolved = path.resolve()
                if resolved in seen_paths:
                    raise ValueError(f"Asset overlap: {path} and {seen_paths[resolved]}")
                seen_paths[resolved] = str(path)
            entries[split].append({
                "image": image.relative_to(root).as_posix(),
                "target": target.relative_to(root).as_posix(),
                "source": image.parent.parent.name, "disaster": disaster(image),
                "image_stat": _fingerprint(image), "target_stat": _fingerprint(target),
            })
        counts[split] = {
            "total": len(entries[split]),
            "by_source": dict(sorted(Counter(e["source"] for e in entries[split]).items())),
            "by_disaster": dict(sorted(Counter(e["disaster"] for e in entries[split]).items())),
        }
    # Check hold membership without opening or scoring its imagery or annotations.
    for image in (root / "hold" / "images").glob("*_post_disaster.png"):
        if image.name in seen:
            raise ValueError(f"Unused hold sample overlaps {seen[image.name]}: {image.name}")
    manifest = {
        "schema_version": 1, "seed": seed, "val_fraction": val_fraction,
        "include_tier3": include_tier3, "source_folders": ["train", "tier3", "test"]
        if include_tier3 else ["train", "test"],
        "unused_source_folders": ["hold"], "counts": counts, "splits": entries,
        "fingerprint_policy": "relative paths, byte sizes and nanosecond mtimes",
    }
    manifest["manifest_hash"] = digest(manifest)
    return manifest


def check_manifest(manifest: dict, current: dict) -> None:
    payload = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    if digest(payload) != manifest.get("manifest_hash"):
        raise ValueError("Stored manifest hash is invalid")
    if manifest != current:
        raise ValueError("Dataset/configuration changed: manifest mismatch")


def assert_sweep_counts(manifest: dict) -> None:
    if (not manifest["include_tier3"] or manifest["seed"] != 0
            or manifest["val_fraction"] != 0.1):
        raise ValueError("Sweep requires tier3, seed 0 and original 10% validation")
    expected = {"train": 8889, "val": 279, "test": 933}
    actual = {key: manifest["counts"][key]["total"] for key in SPLITS}
    if actual != expected or manifest["counts"]["train"]["by_source"] != {
        "train": 2520, "tier3": 6369
    }:
        raise ValueError(f"Unexpected sweep split sizes: {manifest['counts']}")


def split_paths(root, manifest: dict) -> dict[str, list[str]]:
    root = Path(root).expanduser().resolve()
    return {split: [str(root / e["image"]) for e in manifest["splits"][split]]
            for split in SPLITS}


def load_mask(path, expected_size=(1024, 1024)) -> np.ndarray:
    with Image.open(path) as mask:
        if mask.format != "PNG" or mask.mode != "L" or mask.size != expected_size:
            raise ValueError(
                f"Invalid target {path}: expected grayscale L PNG {expected_size}, "
                f"got {mask.format}/{mask.mode}/{mask.size}"
            )
        values = np.asarray(mask).copy()
    if values.min() < 0 or values.max() > 4:
        raise ValueError(f"Invalid target codes in {path}: {np.unique(values).tolist()}")
    return values


def load_image(path, expected_size=(1024, 1024)) -> np.ndarray:
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != expected_size:
            raise ValueError(
                f"Invalid image {path}: expected RGB PNG {expected_size}, "
                f"got {image.format}/{image.mode}/{image.size}"
            )
        return np.asarray(image, dtype=np.float32).copy()


def validate_manifest(root, manifest: dict, expected_size=(1024, 1024)) -> dict:
    """Sequential preflight: PNG integrity, RGB dimensions and every target pixel."""
    start = time.monotonic()
    histograms = {}
    root = Path(root).expanduser().resolve()
    for split in SPLITS:
        counts = np.zeros(5, dtype=np.int64)
        for entry in manifest["splits"][split]:
            image_path, target_path = root / entry["image"], root / entry["target"]
            for path, key in ((image_path, "image_stat"), (target_path, "target_stat")):
                if _fingerprint(path) != entry[key]:
                    raise ValueError(f"Asset changed since manifest construction: {path}")
            with Image.open(image_path) as image:
                if (image.format != "PNG" or image.mode != "RGB"
                        or image.size != expected_size):
                    raise ValueError(f"Invalid RGB PNG/dimensions: {image_path}")
                image.verify()
            mask = load_mask(target_path, expected_size)
            counts += np.bincount(mask.ravel(), minlength=5)
        histograms[split] = counts.tolist()
    return {
        "schema_version": 1, "status": "validated",
        "manifest_hash": manifest["manifest_hash"],
        "image_size": list(expected_size), "target_codes": [0, 1, 2, 3, 4],
        "images_validated": sum(manifest["counts"][s]["total"] for s in SPLITS),
        "pixel_counts_by_split": histograms, "seconds": time.monotonic() - start,
    }


def class_weight_estimate(image_fns, grouping, sample=300, expected_size=(1024, 1024)):
    """Preserve the historical Random(0), 300-mask inverse-square-root estimator."""
    if not image_fns or sample <= 0:
        raise ValueError("Class weights require nonempty training data and a positive sample")
    filenames = (image_fns if len(image_fns) <= sample
                 else random.Random(0).sample(image_fns, sample))
    lut = np.asarray(GROUPINGS[grouping], dtype=np.uint8)
    counts = np.zeros(3, dtype=np.float64)
    for filename in filenames:
        mapped = lut[load_mask(target_for(filename), expected_size)]
        counts += np.bincount(mapped.ravel(), minlength=3)
    weights = 1.0 / np.sqrt(counts / counts.sum() + 1e-6)
    weights /= weights.mean()
    return weights.astype(np.float32), counts


def shared_class_weights(root, manifest) -> dict:
    filenames = split_paths(root, manifest)["train"]
    result = {"manifest_hash": manifest["manifest_hash"],
              "estimator": {"seed": 0, "sample": 300, "formula": "mean-normalized 1/sqrt(freq+1e-6)"},
              "groupings": {}}
    for grouping in GROUPINGS:
        weights, counts = class_weight_estimate(filenames, grouping)
        result["groupings"][grouping] = {
            "weights": weights.tolist(), "pixel_counts": counts.astype(np.int64).tolist(),
        }
    return result
