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

Inference uses the training script's ImageNet normalization without clipping, 512-pixel patches, 64 pixels of discarded context per side, and batches of 8. Override these with `--patch-size`, `--padding`, and `--batch-size`. Two reader threads prepare and prefetch batches while the GPU runs; use `--num-workers 0` for serial loading or adjust `--num-workers` and `--prefetch-factor` (default 2 prepared batches per reader). Each reading thread opens and closes its own raster handle, CUDA inputs use pinned memory, and writes remain ordered. Reads and writes are windowed and prefetching is bounded rather than loading the whole image into RAM. Edge pixels are predicted too, including for images smaller than one patch.

Outputs retain the input dimensions, CRS, and transform. Pixel values are **0 = background**, **1 = undamaged**, and **2 = damaged**, where the checkpoint determines the damage grouping. Input pixels masked in any band become **255 = NoData**; background remains valid data. Outputs are compressed, tiled GeoTIFFs with a class palette and checkpoint metadata. Existing outputs are protected unless `--overwrite` is supplied, and failed runs do not replace them.

The default device is `cuda:0`; use `--device cpu` for explicit CPU inference. CUDA uses float16 autocast, while CPU inference uses float32. Device indices respect `CUDA_VISIBLE_DEVICES`, so `CUDA_VISIBLE_DEVICES=0 python inference_dinov3.py ... --device cuda:0` restricts execution to physical GPU 0.

Do not use the generic `inference.py` for these xView2 checkpoints: its preprocessing clips normalized values and its output class conventions differ. `inference_dinov3.py` specifically expects checkpoints trained with the xView2 script's RGB normalization and three-class labels, not arbitrary custom `fine_tune.py` label schemes.

## Inference performance

The original serial inference path spent 6.86 seconds of a 20.1-second inference loop preparing 1,034 patches, including 3.98 seconds in masked-array indexing. The updated reader avoids reflection/indexing for interior windows and avoids masked arrays when all pixels are valid. It preserves the original channels-last tensor layout as well as the normalization and tiling, so these data-loading changes do not change model inputs or the default class predictions.

On a V100, the warmed model-only ceiling was approximately 145 patches/second at batch size 8 and 158 at batch sizes 32-64. Increasing batch size therefore does not address the main serial data-feeding bottleneck. In repeated runs on a 17,872 x 8,409 RGB raster, the original end-to-end time was 28.58 seconds, the fast reader without prefetching took a median 23.21 seconds, and two reader threads took a median 18.06 seconds. All batch-8 configurations produced exactly the original class raster. Four and eight readers provided no further improvement on that input.

| Input (`any` checkpoint) | Original wall seconds | Optimized wall seconds | Speedup |
| --- | ---: | ---: | ---: |
| `B160001101B07110.tif` (34,404 x 105,387) | 494.16 | 246.06 | 2.01x |
| `B150001101890B10_clip_warped.tif` (20,025 x 11,175) | 40.25 | 22.13 | 1.82x |
| `B040001100075810_clip_warped.tif` (17,872 x 8,409) | 28.58 | 18.06 | 1.58x |

Optimized settings are batch size 8 and two readers. The first and third optimized entries are medians of two runs; the middle entry is a single run. The full-scene repeats took 241.93 and 250.19 seconds, with no changed class pixels across the 3.63-billion-pixel outputs. Full-scene peak sampled host memory was 2.35 GiB with the 512 MB GDAL cache, and GPU memory was about 2.9 GiB. GPU utilization samples after model allocation averaged about 75%, illustrating that a substantial wall-clock improvement does not require 100% utilization. The `major` and `destroyed` checkpoints also retained exact class outputs on both smaller rasters.

Batch sizes 16 and 32 took about 17.6 seconds on that raster, but changed 9,782 of 150,285,648 class predictions (about 0.0065%) through floating-point/kernel differences. Batch size 8 remains the default. Larger batches are an explicit choice, not an automatic tuning step.

Reproduce end-to-end measurements, including process startup and GeoTIFF writing, with a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 GDAL_CACHEMAX=512 \
python scripts/benchmark_inference_dinov3.py \
    --input path/to/image.tif \
    --checkpoint path/to/model.ckpt \
    --output-dir outputs/inference_benchmark \
    --reference path/to/baseline_predictions.tif \
    --batch-sizes 8 --num-workers 0 2 4 8 --repeats 2
```

The harness saves commands, per-run logs and outputs, wall-clock throughput, sampled GPU utilization/VRAM and process RAM, and pixel differences from the optional reference. Add `--profile` to save Python profiling traces. Use a separate invocation with `--compute-only --batch-sizes 8 16 32 64` to measure the warmed model-only ceiling using CUDA events. Benchmark before starting competing training jobs; OS/GDAL caches, CPU load, and raster compression affect throughput. `GDAL_CACHEMAX=512` bounds the GDAL cache to 512 MB for the measurements above.

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

## Original + tier3 training and learning-rate sweep

The existing original-data models used 2,520 training images and a fixed 279-image validation split from the original `train/` folder, with 933 `test/` images evaluated separately. They did not use tier3. `--include-tier3` appends tier3 to the training partition after making that same seed-0 original split; it never moves tier3 into validation or uses `hold/`.

On the complete local dataset at `~/data/xview2/`, the expanded protocol contains **8,889 training images** (2,520 original plus 6,369 tier3), the same **279 original validation images**, and **933 test images**. The uppercase `~/data/XView2/` contains only the original train/test data. The original validation images and targets were byte-identical between these two local copies.

For a single validation-only experiment:

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/train_eval_xview2_dinov3_upernet.py \
    --xview2-root ~/data/xview2 --include-tier3 \
    --grouping any --lr 1e-4 --max-epochs 15 --num-workers 4 \
    --gpu 0 --evaluation-split val \
    --output-dir outputs/xview2_tier3_single
```

The sweep launcher runs all three groupings at initial learning rates `1e-5`, `3e-5`, `1e-4`, and `3e-4`: **12 trials, 15 epochs each**. Every trial starts with the pretrained `dinov3_vits16` backbone and a fresh UPerNet head. It keeps batch size 16, four 512-pixel crops per image, seed 0, the seeded 300-mask inverse-square-root class-weight estimator, and the existing AdamW/ReduceLROnPlateau policy. Learning rates may subsequently decrease under the scheduler. Crops are uniform random crops, not building-biased crops, and DataLoader workers receive deterministic distinct seeds.

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 \
python scripts/sweep_xview2_dinov3_upernet.py \
    --xview2-root ~/data/xview2 --max-concurrent 4
```

The empty parent `CUDA_VISIBLE_DEVICES` is intentional: the controller discovers the selected physical GPUs using `nvidia-smi`, then exposes exactly one selected UUID to each child as logical GPU 0. The default selection is **4, 5, 6, and 7**; `--gpus` can additionally select **0, 2, and 3**, with up to seven concurrent trials. **GPU 1 is always excluded.** It runs at most one trial per GPU, does not take over busy GPUs, and never falls back to CPU. Each trial uses four loading workers and two CPU threads. `HF_HUB_OFFLINE=1` requires both the backbone configuration and pretrained weights to already be cached; omit that setting for an authorized initial download.

Admission is constrained by the actual Linux cgroup v2 memory limit, not host-wide RAM. The default policy reserves **8 GiB per job** and **16 GiB for safety/other workloads**. It measures active process-tree PSS to avoid double-counting forked workers and conservatively excludes only clean inactive file cache from the working-set estimate. New jobs wait if their projected peak would violate the budget. Exhausting the working-set reserve or observing a new cgroup OOM kill stops owned jobs while preserving checkpoints; no cgroup limits or global caches are modified. `state.json` records memory observations and admission decisions. Use `--memory-per-job-gib` and `--memory-reserve-gib` to explicitly configure the policy.

On the current machine, the cgroup is limited to **96 GiB and eight CPU cores**, despite much larger host-wide figures. The existing trial trees used approximately 2.6-2.7 GiB PSS each. Additional GPU capacity is therefore not a guarantee of proportional throughput: the CPU quota may become the limiting resource.

Each launch allocates a fresh `outputs/xview2_dinov3_upernet_tier3_lr_sweep/run_NNN/`. Before training, it validates all image/target pairs, saves a source/disaster split manifest with file fingerprints, and computes shared training-only class weights. Runtime state is written atomically to `state.json`; per-trial logs, epoch history, configuration, selected best-validation-loss checkpoint, and full optimizer/scheduler/scaler `last.ckpt` live under `<grouping>/lr_<rate>/`.

After **all twelve trials** complete, the launcher chooses one LR per grouping using the **highest validation damaged-class F1 at that trial's best-validation-loss checkpoint**. Ties use lower validation loss, then lower initial LR. Only those three winners are evaluated on the full original test set. `selection.json` contains their direct checkpoint paths; `validation_sweep.csv`, `winner_test.csv`, and `summary.json` contain the sweep, winner metrics, and comparisons with the original-data-only baselines.

To resume an interrupted run, use its exact directory and unchanged training code, environment, dataset, and configuration. Saved GPU and memory settings are inherited when their flags are omitted:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 \
python scripts/sweep_xview2_dinov3_upernet.py \
    --xview2-root ~/data/xview2 \
    --run-dir outputs/xview2_dinov3_upernet_tier3_lr_sweep/run_001 \
    --resume
```

For an explicitly authorized resource change, stop the existing controller after its children have saved epoch checkpoints, then resume with a scheduler-only migration:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 \
python scripts/sweep_xview2_dinov3_upernet.py \
    --xview2-root ~/data/xview2 \
    --run-dir outputs/xview2_dinov3_upernet_tier3_lr_sweep/run_001 \
    --resume --reconfigure-resources --reuse-preflight \
    --gpus 4 5 6 7 0 2 3 --max-concurrent 7 \
    --memory-per-job-gib 8 --memory-reserve-gib 16
```

Resource migrations archive the previous configuration and preserve the training recipe, data manifest, model/training/data code hashes, dependency versions, and baseline metrics. They cannot silently change epochs, learning rates, data, or model code. `--reuse-preflight` reuses a prior complete validation report only when the current manifest and file fingerprints still match; otherwise it fails rather than trusting a directory name.

Locks and recorded process identities prevent duplicate controllers or duplicate live trials. The controller retains failed/interrupted state, propagates failures, and cleans up only its own children; it does not silently omit failed trials from winner selection. Jobs stay attached to the controller rather than being detached services. `--prepare-only` performs the preflight without starting GPU jobs.

The validation/test sets still cover only the original disasters, so these results do not directly measure generalization to tier3-only disasters. Fifteen epochs over a larger training set also means more training steps; comparisons are not equal-compute ablations or statistical-significance claims. Selected checkpoints remain directly usable with `inference_dinov3.py`.

The completed `run_001` sweep selected initial LRs `3e-5` (`any`), `1e-4` (`major`), and `3e-5` (`destroyed`). On the original 933-image test set, damaged F1 was 0.5254, 0.4675, and 0.4072 respectively, versus original-only baselines of 0.5234, 0.5476, and 0.4264. Adding tier3 was therefore not a clear overall improvement on this benchmark. The [full sweep results and checkpoint paths](scripts/xview2_dinov3_upernet_RESULTS.md#original--tier3-learning-rate-sweep) include the validation sweep, winner-only test comparison, per-class metrics, and interpretation.

## Any-damage boundary and loss study

The any-only study compares longer training at initial LR `3e-5` with six loss recipes: `ce`, `ce_dice`, `boundary_ce`, `boundary_ce_dice`, `focal_dice`, and `ce_tversky`. All new trajectories start from the pretrained backbone and a fresh head, on the same 8,889 training / 279 validation split. ReduceLROnPlateau remains active; the existing fifteen-epoch winner had not yet reduced its initial learning rate.

Boundary weighting is specifically for **undamaged class 1**, not the union of undamaged and damaged buildings. The default band extends three pixels around the observed class-1 contour and multiplies CE weights by four on both sides, including the surrounding background. Class-2 contours do not create their own band. Crop edges and ignored regions do not create artificial contours. The weighted mean is normalized by the combined class/spatial weights.

Dice and Tversky cover foreground classes 1 and 2 with coefficient 1. Focal uses softmax likelihoods, gamma 2, the existing class weights, and no extra alpha rebalance. Tversky uses a false-positive coefficient of 0.7 and false-negative coefficient of 0.3. Probability and region reductions use float32 under mixed precision. The legacy standalone `ce` and `dice` modes retain their behavior.

Geometry evaluation reports class-1 contour precision, recall, and F1 with a **two-pixel Chebyshev matching radius** (a square neighborhood), along with class-1 overlap/precision/recall, predicted-to-true area ratio, building-union IoU, and damaged F1. Contour counts are aggregated over the split; empty evaluations are undefined rather than artificially perfect.

The archived any-model validation reference has boundary F1 **0.2734094**, boundary precision **0.3026636**, boundary recall **0.2493120**, undamaged IoU **0.5385607**, damaged F1 **0.5687645**, and predicted/true undamaged area ratio **1.4956332**. Its complete pixel confusion matrix remains identical to the archived evaluation. Reference metrics and twelve ground-truth-selected visual panels are in `outputs/xview2_any_reference/run_001/`. Panels show identical native-resolution crops of RGB, ground truth, predictions, and undamaged contours; panel selection does not use predictions.

The staged protocol trains the CE control through sixty epochs with fifteen/thirty/sixty-epoch milestones, trains the five alternatives through thirty epochs, then extends at most two qualifying alternatives to sixty. Selection prioritizes boundary F1 while allowing at most a 0.01 absolute decrease in damaged F1 or undamaged IoU against the archived reference and matched CE control. All promotion decisions use frozen thirty-epoch validation evidence; the original test set is used only after final model selection, and `hold/` stays unused. No variant is automatically promoted if it fails the quality guards.

Launch a fresh study after the reference and panel artifacts have been prepared:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python scripts/study_xview2_any_losses.py \
    --reference-json outputs/xview2_any_reference/run_001/reference_metrics.json \
    --panel-manifest outputs/xview2_any_reference/run_001/panels.json \
    --max-concurrent 6 --cpu-threads 1 --num-workers 1
```

The controller allocates a new numbered root under `outputs/xview2_any_loss_study/`. `--prepare-only` validates and freezes a configuration without starting GPU jobs. A prepared configuration rejects subsequent code, data, reference, or resource drift; use a fresh root if implementation inputs change before launch. The one-reader/one-CPU-thread setting is deliberate for the eight-core cgroup, and epoch-indexed sampling keeps input crops independent of reader count.

Each trajectory saves per-epoch weights-only candidates, a full resumable `last.ckpt`, and immutable `stages/epoch_015`, `epoch_030`, and, when reached, `epoch_060` snapshots. A promoted trajectory resumes from the frozen thirty-epoch continuation state, not its earlier best-weight candidate. The runner can start promoted variants while CE continues, but only frozen thirty-epoch evidence is used for promotion.

Python/NumPy/Torch/CUDA state and input-loader state are preserved across stages. Native CUDA pooling/backward operations are not bitwise deterministic: preserved inputs, randomness, optimizer-step counts, LR progression, and scaler state do not imply bitwise-identical model tensors. This is recorded rather than hidden by changing model precision or architecture.

`promotion.json` records every promotion decision. `final_selection.json` freezes selected checkpoint identities before any test scoring. `validation_candidates.csv`, `guarded_selections.csv`, `selected_test_comparisons.csv`, and `summary.json` contain the results, including explicitly ineligible recipes. Model artifacts and plots remain outside git.

To render a candidate on the already-fixed panel list, use full-image float32 predictions and crop only for display:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/render_any_panels.py \
    --checkpoint path/to/candidate.ckpt \
    --panel-manifest outputs/xview2_any_reference/run_001/panels.json \
    --output-dir outputs/candidate_panels \
    --label "candidate at 30 epochs"
```
