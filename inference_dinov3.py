# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Run an xView2 DINOv3 + UPerNet checkpoint on an 8-bit RGB GeoTIFF.

Output classes are 0 (background), 1 (undamaged), and 2 (damaged), with
255 reserved for NoData. The meaning of "damaged" depends on the checkpoint's
training grouping: any damage, major damage or worse, or destroyed only.
"""

from __future__ import annotations

import argparse
from itertools import islice, product
import math
from pathlib import Path
import tempfile

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.windows import Window
import torch
from tqdm import tqdm

from bda.dinov3_upernet import DINOv3UPerNet


# Match scripts/train_eval_xview2_dinov3_upernet.py, without clipping.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0


def load_model(checkpoint_fn: Path, device: torch.device) -> DINOv3UPerNet:
    """Load model weights without constructing the Lightning training task."""
    checkpoint = torch.load(checkpoint_fn, map_location="cpu", weights_only=True)
    hparams = checkpoint["hyper_parameters"]
    expected = {"model": "upernet", "in_channels": 3, "num_classes": 3}
    if any(hparams.get(key) != value for key, value in expected.items()):
        raise ValueError("Expected a three-class, three-channel UPerNet checkpoint")
    model = DINOv3UPerNet(backbone=hparams["backbone"], pretrained=False)
    state = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False).to(device)


def _reflect_indices(start: int, size: int, limit: int) -> np.ndarray:
    if limit == 1:
        return np.zeros(size, dtype=np.int64)
    period = 2 * (limit - 1)
    indices = np.arange(start, start + size) % period
    return np.minimum(indices, period - indices)


def read_patch(src, y: int, x: int, patch_size: int, padding: int):
    """Read an overlapping patch, reflecting context beyond the image edges."""
    rows = _reflect_indices(y - padding, patch_size, src.height)
    cols = _reflect_indices(x - padding, patch_size, src.width)
    y0, x0 = int(rows.min()), int(cols.min())
    y1, x1 = int(rows.max()) + 1, int(cols.max()) + 1
    image = src.read(window=Window(x0, y0, x1 - x0, y1 - y0), masked=True)
    image = image[:, rows[:, None] - y0, cols[None, :] - x0]
    valid = ~np.ma.getmaskarray(image).any(axis=0)
    return image.filled(0), valid


@torch.inference_mode()
def run_inference(
    input_fn: Path,
    checkpoint_fn: Path,
    output_fn: Path,
    device: str = "cuda:0",
    patch_size: int = 512,
    padding: int = 64,
    batch_size: int = 8,
    overwrite: bool = False,
) -> None:
    """Stream predictions to a GeoTIFF on the source grid with bounded memory."""
    if patch_size < 32 or patch_size % 16:
        raise ValueError("patch_size must be a multiple of 16 and at least 32")
    if padding < 0 or 2 * padding >= patch_size:
        raise ValueError("padding must be nonnegative and less than half patch_size")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if output_fn.resolve() in (input_fn.resolve(), checkpoint_fn.resolve()):
        raise ValueError("Output must not overwrite the input image or checkpoint")
    if output_fn.exists() and not overwrite:
        raise FileExistsError(f"{output_fn} already exists; use --overwrite to replace it")
    target_device = torch.device(device)
    if target_device.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda:<index>")
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for CPU inference")

    with rasterio.open(input_fn) as src:
        if src.count != 3 or src.dtypes != ("uint8",) * 3:
            raise ValueError("Expected an 8-bit, three-band RGB raster")
        if src.colorinterp != (ColorInterp.red, ColorInterp.green, ColorInterp.blue):
            raise ValueError("Expected bands in red, green, blue order")
        model = load_model(checkpoint_fn, target_device)
        stride = patch_size - 2 * padding
        coordinates = product(range(0, src.height, stride), range(0, src.width, stride))
        num_patches = math.ceil(src.height / stride) * math.ceil(src.width / stride)
        profile = {
            "driver": "GTiff",
            "width": src.width,
            "height": src.height,
            "crs": src.crs,
            "transform": src.transform,
            "count": 1,
            "dtype": "uint8",
            "nodata": 255,
            "compress": "lzw",
            "predictor": 2,
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "BIGTIFF": "IF_SAFER",
        }
        output_fn.parent.mkdir(parents=True, exist_ok=True)
        print(f"Running {checkpoint_fn} on {device}: {num_patches} patches", flush=True)
        with tempfile.TemporaryDirectory(prefix=".dinov3-", dir=output_fn.parent) as tmp:
            temporary_fn = Path(tmp) / "predictions.tif"
            with rasterio.open(temporary_fn, "w", **profile) as dst:
                dst.set_band_description(1, "building_damage_class")
                dst.write_colormap(1, {
                    0: (0, 0, 0, 255),
                    1: (0, 200, 0, 255),
                    2: (255, 0, 0, 255),
                    255: (0, 0, 0, 0),
                })
                dst.update_tags(
                    MODEL=f"{model.__class__.__name__}",
                    CHECKPOINT=str(checkpoint_fn.resolve()),
                    SOURCE_IMAGE=str(input_fn.resolve()),
                    CLASS_0="background",
                    CLASS_1="undamaged",
                    CLASS_2="damaged",
                    NORMALIZATION="ImageNet mean/std on 8-bit RGB; no clipping",
                )
                with tqdm(total=num_patches, unit="patch") as progress:
                    while batch_coordinates := list(islice(coordinates, batch_size)):
                        patches = [
                            read_patch(src, y, x, patch_size, padding)
                            for y, x in batch_coordinates
                        ]
                        images = np.stack([image for image, _ in patches]).astype(np.float32)
                        images -= IMAGENET_MEAN[None, :, None, None]
                        images /= IMAGENET_STD[None, :, None, None]
                        inputs = torch.from_numpy(images).to(target_device)
                        with torch.autocast(
                            device_type=target_device.type,
                            dtype=torch.float16,
                            enabled=target_device.type == "cuda",
                        ):
                            logits = model(inputs)
                        if not torch.isfinite(logits).all().item():
                            raise RuntimeError("Model produced non-finite predictions")
                        predictions = logits.argmax(1).to(torch.uint8).cpu().numpy()
                        for prediction, (_, valid), (y, x) in zip(
                            predictions, patches, batch_coordinates
                        ):
                            h, w = min(stride, src.height - y), min(stride, src.width - x)
                            core = np.s_[padding : padding + h, padding : padding + w]
                            output = np.where(valid[core], prediction[core], 255).astype(np.uint8)
                            dst.write(output, 1, window=Window(x, y, w, h))
                        progress.update(len(batch_coordinates))
            if output_fn.exists() and not overwrite:
                raise FileExistsError(f"{output_fn} was created while inference was running")
            temporary_fn.replace(output_fn)
    print(f"Saved {output_fn}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input 8-bit RGB GeoTIFF")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained .ckpt file")
    parser.add_argument("--output", type=Path, required=True, help="Output class GeoTIFF")
    parser.add_argument("--device", default="cuda:0", help="Torch device (default: cuda:0); CPU: cpu")
    parser.add_argument("--patch-size", type=int, default=512, help="Patch size (default: 512)")
    parser.add_argument("--padding", type=int, default=64, help="Discarded context per side (default: 64)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output")
    args = parser.parse_args()
    run_inference(
        args.input, args.checkpoint, args.output, device=args.device,
        patch_size=args.patch_size, padding=args.padding,
        batch_size=args.batch_size, overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
