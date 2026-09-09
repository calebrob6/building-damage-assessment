# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Summarize the xView2 (xBD) train/val/test split used by the DINOv3+UPerNet
example (``scripts/train_eval_xview2_dinov3_upernet.py``).

Reproduces the exact split -- the ``train/`` folder is shuffled with ``--seed``
and the first ``--val-fraction`` becomes ``val`` (the rest ``train``); the
``test/`` folder is ``test`` -- and writes a CSV of the number of **patches**
(1024x1024 post-disaster image tiles) and **footprints** (building polygons in
the post-disaster label JSONs) per disaster per split, plus a TOTAL row.
With ``--include-tier3``, append all tier3 images to training only, using the
shared manifest helper without changing original validation membership.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bda.xview2 import (
    build_manifest, disaster as _disaster, split_paths, validate_manifest, write_json,
)

SPLITS = ["train", "val", "test"]


def _num_footprints(image_fn: str) -> int:
    folder = os.path.dirname(os.path.dirname(image_fn))
    label_fn = os.path.join(
        folder, "labels", os.path.basename(image_fn).replace(".png", ".json")
    )
    with open(label_fn) as stream:
        feats = json.load(stream)["features"]["xy"]
    return sum(1 for f in feats if f.get("properties", {}).get("feature_type") == "building")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xview2-root", default=os.path.expanduser("~/data/XView2"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--output", default="scripts/xview2_split_sizes.csv")
    p.add_argument("--include-tier3", action="store_true",
                   help="Append all tier3 to training, preserving original validation.")
    p.add_argument("--manifest-output", help="Also write the reproducible split manifest.")
    p.add_argument("--validate", action="store_true",
                   help="Sequentially validate every RGB PNG and target before reporting.")
    args = p.parse_args()

    manifest = build_manifest(
        args.xview2_root, args.include_tier3, args.seed, args.val_fraction,
    )
    if args.validate:
        report = validate_manifest(args.xview2_root, manifest)
        print(json.dumps(report, indent=2))
    if args.manifest_output:
        write_json(args.manifest_output, manifest)
    split_fns = split_paths(args.xview2_root, manifest)

    # disaster -> split -> (patches, footprints)
    patches: dict[str, dict[str, int]] = {}
    footprints: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        for fn in split_fns[split]:
            d = _disaster(fn)
            patches.setdefault(d, {s: 0 for s in SPLITS})
            footprints.setdefault(d, {s: 0 for s in SPLITS})
            patches[d][split] += 1
            footprints[d][split] += _num_footprints(fn)

    header = ["disaster"]
    for s in SPLITS:
        header += [f"{s}_patches", f"{s}_footprints"]
    header += ["total_patches", "total_footprints"]

    rows = []
    totals = {f"{s}_patches": 0 for s in SPLITS}
    totals.update({f"{s}_footprints": 0 for s in SPLITS})
    for d in sorted(patches):
        row = {"disaster": d}
        tp = tf = 0
        for s in SPLITS:
            row[f"{s}_patches"] = patches[d][s]
            row[f"{s}_footprints"] = footprints[d][s]
            totals[f"{s}_patches"] += patches[d][s]
            totals[f"{s}_footprints"] += footprints[d][s]
            tp += patches[d][s]
            tf += footprints[d][s]
        row["total_patches"] = tp
        row["total_footprints"] = tf
        rows.append(row)

    total_row = {"disaster": "TOTAL"}
    total_row.update(totals)
    total_row["total_patches"] = sum(totals[f"{s}_patches"] for s in SPLITS)
    total_row["total_footprints"] = sum(totals[f"{s}_footprints"] for s in SPLITS)
    rows.append(total_row)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

    widths = {h: max(len(h), *(len(str(r[h])) for r in rows)) for h in header}
    print(" | ".join(h.ljust(widths[h]) for h in header))
    for r in rows:
        print(" | ".join(str(r[h]).ljust(widths[h]) for h in header))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
