# xView2 DINOv3 + UPerNet results

Fully-supervised DINOv3 ViT-S/16 + UPerNet (`scripts/train_eval_xview2_dinov3_upernet.py`),
trained on the xView2 (xBD) `train/` folder and evaluated on the **full**
`test/` folder (933 post-disaster images, per-pixel metrics accumulated over a
3×3 confusion matrix). Each row is a separate model that collapses the four xBD
damage grades into one `damaged` class differently; every model is a 3-class
`{background, undamaged, damaged}` segmentation.

**Training config:** backbone `dinov3_vits16` (LVD-1689M, pretrained, fine-tuned end-to-end), 512-pixel uniform random crops, batch 16, class-weighted cross-entropy (inverse-sqrt-frequency), AdamW lr 1e-4, 15 epochs, mixed precision, ImageNet normalization. Best-`val_loss` checkpoint. The original sampler returned its first random crop, so the previously documented building-biased sampling was not actually active.

## Test-set metrics (per-pixel)

| Grouping (damaged =) | Class | IoU | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| **any** (minor+major+destroyed) | background | 0.954 | 0.994 | 0.960 | 0.976 |
| | undamaged | 0.542 | 0.601 | 0.847 | 0.703 |
| | damaged | 0.354 | 0.391 | 0.791 | 0.523 |
| **major** (major+destroyed) | background | 0.958 | 0.994 | 0.963 | 0.978 |
| | undamaged | 0.563 | 0.610 | 0.879 | 0.720 |
| | damaged | 0.377 | 0.424 | 0.774 | 0.548 |
| **destroyed** (destroyed only) | background | 0.958 | 0.994 | 0.964 | 0.979 |
| | undamaged | 0.587 | 0.629 | 0.898 | 0.739 |
| | damaged | 0.271 | 0.298 | 0.749 | 0.426 |

| Grouping | mean IoU | damaged F1 | overall accuracy |
| --- | --- | --- | --- |
| any | 0.617 | 0.523 | 0.952 |
| major | 0.632 | 0.548 | 0.957 |
| destroyed | 0.605 | 0.426 | 0.959 |

**Notes.** The damaged class is rare (≈0.3–2% of pixels) and gets rarer from
`any` → `destroyed`, which is why its IoU drops accordingly. The class-weighted
loss trades precision for recall (damaged recall 0.75–0.79), i.e. the models find
most damaged pixels at the cost of some false positives — a reasonable operating
point for triage. A larger backbone (`dinov3_vitb16` / `dinov3_vitl16_sat`),
longer training, or post-hoc thresholding would push these further.

Reproduce:

```bash
for g in any major destroyed; do
  python scripts/train_eval_xview2_dinov3_upernet.py \
      --xview2-root ~/data/XView2 --grouping $g --gpu 0 \
      --output-dir outputs/xview2_dinov3_upernet_$g
done
```

## Original + tier3 learning-rate sweep

**Outcome:** adding tier3 did not provide a clear overall improvement on the original held-out test set. The selected `any` model was effectively tied with the original model in damaged F1, while `major` and `destroyed` regressed. All three selected models increased damaged-class recall, but precision fell, especially for `major`.

All **12 trials completed 15 epochs and 33,330 logged training steps**. Training used **8,889 images**: the same 2,520 original training images plus all 6,369 tier3 images. The original 279-image validation split was unchanged, and its images and targets were byte-identical between the two local dataset copies. `hold/` was not used. Each trial started from the pretrained DINOv3 ViT-S/16 backbone and a fresh UPerNet head, with seed 0, four uniform 512-pixel crops per image, batch size 16, mixed precision, and the existing seeded 300-mask inverse-square-root class-weight estimator.

The sweep varied the initial AdamW learning rate; ReduceLROnPlateau remained active. Within each trial, the lowest-validation-loss checkpoint was retained. Learning rates were then ranked by damaged-class F1 on the full validation split, with lower validation loss and lower initial LR as deterministic tie breakers. Only the three selected winners were evaluated on the full **933-image original test set**.

### Complete validation sweep

Epoch numbers below are zero-based checkpoint epochs, not the total number of epochs trained. Bold rows identify the selected learning rate for each grouping.

| Grouping | Initial LR | Selected epoch | Best validation loss | Validation damaged F1 | Validation mean IoU |
| --- | --- | ---: | ---: | ---: | ---: |
| any | `1e-5` | 14 | 0.1773 | 0.5234 | 0.6073 |
| **any** | **`3e-5`** | **13** | **0.1773** | **0.5688** | **0.6282** |
| any | `1e-4` | 9 | 0.1842 | 0.5046 | 0.5980 |
| any | `3e-4` | 9 | 0.2128 | 0.4667 | 0.5712 |
| major | `1e-5` | 7 | 0.1979 | 0.4198 | 0.5717 |
| major | `3e-5` | 13 | 0.1976 | 0.4060 | 0.5681 |
| **major** | **`1e-4`** | **9** | **0.2016** | **0.4526** | **0.5811** |
| major | `3e-4` | 13 | 0.2187 | 0.3795 | 0.5427 |
| destroyed | `1e-5` | 11 | 0.1429 | 0.3662 | 0.5716 |
| **destroyed** | **`3e-5`** | **12** | **0.1417** | **0.4334** | **0.6020** |
| destroyed | `1e-4` | 14 | 0.1443 | 0.3774 | 0.5846 |
| destroyed | `3e-4` | 14 | 0.1627 | 0.3688 | 0.5648 |

### Winner-only test comparison

The original-only columns are the archived baseline metrics reported above. F1 differences are absolute score differences, not relative percentages.

| Grouping | Selected LR | Original damaged F1 | Tier3 damaged F1 | F1 difference | Original mean IoU | Tier3 mean IoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| any | `3e-5` | 0.5234 | 0.5254 | +0.0020 | 0.6169 | 0.6218 |
| major | `1e-4` | 0.5476 | 0.4675 | -0.0801 | 0.6324 | 0.5942 |
| destroyed | `3e-5` | 0.4264 | 0.4072 | -0.0193 | 0.6052 | 0.5971 |

| Grouping | Class | Test IoU | Test precision | Test recall | Test F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| any | background | 0.9522 | 0.9954 | 0.9564 | 0.9755 |
| any | undamaged | 0.5569 | 0.5970 | 0.8925 | 0.7154 |
| any | damaged | 0.3563 | 0.3895 | 0.8069 | 0.5254 |
| major | background | 0.9465 | 0.9953 | 0.9507 | 0.9725 |
| major | undamaged | 0.5312 | 0.5671 | 0.8934 | 0.6938 |
| major | damaged | 0.3050 | 0.3256 | 0.8285 | 0.4675 |
| destroyed | background | 0.9548 | 0.9951 | 0.9593 | 0.9769 |
| destroyed | undamaged | 0.5808 | 0.6115 | 0.9204 | 0.7348 |
| destroyed | damaged | 0.2556 | 0.2743 | 0.7901 | 0.4072 |

These are single-seed results, and the validation/test sets contain original disasters rather than held-out tier3 disasters. The larger dataset also increases training steps per epoch, and the sweep changes learning rate and worker seeding relative to the archived runs. This is not an equal-compute or data-only ablation, and the small `any` difference is not evidence of a statistically established improvement. On this benchmark, the original `major` and `destroyed` models remain the better balanced-F1 choices.

### Artifacts and inference-ready checkpoints

Run directory: `outputs/xview2_dinov3_upernet_tier3_lr_sweep/run_001/`.

| Selected model | Checkpoint relative to the run directory |
| --- | --- |
| any | `any/lr_3e-05/checkpoints/best-epoch=13-val_loss=0.1773.ckpt` |
| major | `major/lr_1e-04/checkpoints/best-epoch=09-val_loss=0.2016.ckpt` |
| destroyed | `destroyed/lr_3e-05/checkpoints/best-epoch=12-val_loss=0.1417.ckpt` |

Every trial retains both its selected checkpoint and a resumable `last.ckpt`. `selection.json` records the three winner paths. `validation_sweep.csv`, `winner_test.csv`, and `summary.json` contain full-precision metrics, confusion matrices, durations, and baseline comparisons. The original checkpoints and their outputs were not overwritten.

```bash
RUN=outputs/xview2_dinov3_upernet_tier3_lr_sweep/run_001
CUDA_VISIBLE_DEVICES=0 python inference_dinov3.py \
    --input path/to/image.tif \
    --checkpoint "$RUN/any/lr_3e-05/checkpoints/best-epoch=13-val_loss=0.1773.ckpt" \
    --output outputs/image_tier3_any_predictions.tif \
    --device cuda:0
```

Total sweep wall time was **18.63 hours**, including preflight, the checkpoint-preserving resource handoff, and winner evaluation. Concurrency changed from four to seven GPU jobs, then decreased as the queue drained; per-trial durations are therefore not learning-rate speed comparisons. The first four trials resumed from six-epoch checkpoints. Training used only authorized GPUs 0, 2, 3, 4, 5, 6, and 7, never GPU 1, with cgroup-aware memory admission and no new cgroup OOM kills. All trial and evaluation processes have finished.

The separate inference optimization produced about a **2x full-scene speedup with unchanged class outputs**. See [the inference performance measurements](../DINOV3_UPERNET.md#inference-performance); that optimization is independent of the model-quality comparison above.
