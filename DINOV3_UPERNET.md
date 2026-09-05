# DINOv3 + UPerNet segmentation model

This repo can train and run a [DINOv3](https://ai.meta.com/dinov3/) ViT backbone with a [UPerNet](https://arxiv.org/abs/1807.10221) decode head as a drop-in alternative to the `segmentation_models_pytorch` U-Net / DeepLabV3+ models. It is wired into `CustomSemanticSegmentationTask`. Use `inference_dinov3.py` for the three-class xView2 checkpoints described below.

## Architecture

`bda/dinov3_upernet.py` implements `DINOv3UPerNet`:

* **Backbone** — a pretrained DINOv3 ViT (loaded from the gated `facebook/dinov3-*`
  Hugging Face repos). Four equally spaced transformer blocks are read out; the
  class + register tokens are dropped and the patch tokens are reshaped back to
  feature maps.
* **Neck** — the four same-resolution (stride-16) maps are turned into a
  `{stride 4, 8, 16, 32}` pyramid with transposed-conv / identity / max-pool
  branches (the BEiT/MAE ViT-UPerNet recipe).
* **Head** — a UPerNet head (Pyramid Pooling Module + FPN fusion, GroupNorm) that
  produces per-pixel logits, upsampled to the input resolution.

It maps a `(B, in_channels, H, W)` ImageNet-normalized tensor to `(B, num_classes,
H, W)` logits, exactly like the other segmentation models.

Available backbones (`backbone=...`): `dinov3_vits16` (21 M), `dinov3_vitb16`,
`dinov3_vitl16`, `dinov3_vitl16_sat` (satellite-pretrained SAT-493M).

## Requirements

The DINOv3 backbone needs the optional `transformers` dependency (already added to
`environment.yml`) and access to the **gated** DINOv3 weights:

```bash
pip install "transformers>=4.56"
huggingface-cli login          # after requesting access on the model page
```

Model input patch dimensions must be multiples of 16 and at least 32 pixels. `inference_dinov3.py` handles arbitrary raster dimensions by reflecting context at image edges.

## Running xView2 checkpoints with `inference_dinov3.py`

The standalone CLI accepts an 8-bit, three-band RGB GeoTIFF and one trained checkpoint. It rebuilds the architecture from the checkpoint, loads all model weights strictly, and needs only the Hugging Face backbone configuration (downloaded or already cached), not a second copy of the pretrained backbone weights. No experiment YAML is required.

```bash
python inference_dinov3.py \
    --input path/to/image.tif \
    --checkpoint outputs/xview2_dinov3_upernet_any/checkpoints/best-epoch=12-val_loss=0.1613.ckpt \
    --output outputs/image_any_predictions.tif \
    --device cuda:0
```

To run all three locally trained models:

```bash
for grouping in any major destroyed; do
    python inference_dinov3.py \
        --input B160001101B07110_clip.tif \
        --checkpoint outputs/xview2_dinov3_upernet_"$grouping"/checkpoints/best-*.ckpt \
        --output outputs/dinov3_predictions/B160001101B07110_clip_"$grouping"_predictions.tif \
        --device cuda:0 || break
done
```

Inference uses the training script's ImageNet normalization without clipping, 512-pixel patches, 64 pixels of discarded context per side, and batches of 8. Override these with `--patch-size`, `--padding`, and `--batch-size`. Reads and writes are windowed, so memory does not scale with raster area. Edge pixels are predicted too, including for images smaller than one patch.

Outputs retain the input dimensions, CRS, and transform. Pixel values are **0 = background**, **1 = undamaged**, and **2 = damaged**, where the checkpoint determines the damage grouping. Input pixels masked in any band become **255 = NoData**; background remains valid data. Outputs are compressed, tiled GeoTIFFs with a class palette and checkpoint metadata. Existing outputs are protected unless `--overwrite` is supplied, and failed runs do not replace them.

The default device is `cuda:0`; use `--device cpu` for explicit CPU inference. CUDA uses float16 autocast, while CPU inference uses float32. Device indices respect `CUDA_VISIBLE_DEVICES`, so `CUDA_VISIBLE_DEVICES=0 python inference_dinov3.py ... --device cuda:0` restricts execution to physical GPU 0.

Do not use the generic `inference.py` for these xView2 checkpoints: its preprocessing clips normalized values and its output class conventions differ. `inference_dinov3.py` specifically expects checkpoints trained with the xView2 script's RGB normalization and three-class labels, not arbitrary custom `fine_tune.py` label schemes.

## Training through `fine_tune.py`

`fine_tune.py` now reads `training.model`, `training.backbone`, and
`training.weights` from the config (defaulting to the previous
`unet` / `resnext50_32x4d`). To fine-tune DINOv3 + UPerNet, add:

```yaml
training:
  model: upernet
  backbone: dinov3_vits16
  weights: true        # load the pretrained DINOv3 backbone
  # ...existing training keys...
```

## Example: xView2 (xBD) damage segmentation

`scripts/train_eval_xview2_dinov3_upernet.py` trains and evaluates the model on
the [xView2 / xBD](https://xview2.org) dataset, collapsing the four damage grades
into a single `damaged` class in one of three ways
(`{0: background, 1: undamaged, 2: damaged}`):

| grouping | undamaged | damaged |
| --- | --- | --- |
| `any` | no-damage | minor + major + destroyed |
| `major` | no-damage + minor | major + destroyed |
| `destroyed` | no-damage + minor + major | destroyed |

```bash
python scripts/train_eval_xview2_dinov3_upernet.py \
    --xview2-root ~/data/XView2 --grouping any --gpu 0 \
    --output-dir outputs/xview2_dinov3_upernet_any
```

It trains on the `train/` folder (post-disaster RGB, 512-px crops, class-weighted
cross-entropy for the heavy background imbalance) and reports pixel IoU /
precision / recall / F1 per class on the **full** `test/` folder. Results are in
[`RESULTS.md`](scripts/xview2_dinov3_upernet_RESULTS.md).
