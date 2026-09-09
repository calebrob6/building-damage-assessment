# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Measure DINOv3 inference throughput, resources, and output parity."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def compare_outputs(output_fn: Path, reference_fn: Path) -> int:
    differences = 0
    with rasterio.open(output_fn) as output, rasterio.open(reference_fn) as reference:
        for key in ("width", "height", "crs", "transform", "count", "dtype", "nodata"):
            if output.profile[key] != reference.profile[key]:
                raise ValueError(f"Output/reference {key} mismatch")
        for _, window in output.block_windows(1):
            differences += np.count_nonzero(
                output.read(1, window=window) != reference.read(1, window=window)
            )
    return int(differences)


def sample_resources(pid: int, gpu: str | None) -> dict:
    sample = {}
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except FileNotFoundError:
        return sample
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            sample["rss_mb"] = int(line.split()[1]) / 1024
    if gpu is not None:
        result = subprocess.run(
            [
                "nvidia-smi", f"--id={gpu}",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True, capture_output=True, text=True,
        )
        utilization, memory = result.stdout.strip().split(",")
        sample.update(gpu_percent=float(utilization), gpu_memory_mb=float(memory))
    return sample


def save_results(path: Path, result) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)


def compute_ceiling(args) -> list[dict]:
    import torch
    from inference_dinov3 import IMAGENET_MEAN, IMAGENET_STD, load_model, read_patch

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Compute-ceiling mode requires a CUDA device")
    model = load_model(args.checkpoint, device)
    with rasterio.open(args.input) as src:
        patch, _ = read_patch(src, 0, 0, 512, 64)
    patch = (patch.astype(np.float32) - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    results = []
    with torch.cuda.device(device), torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for batch_size in args.batch_sizes:
            inputs = torch.from_numpy(np.stack([patch] * batch_size)).to(
                device=device, memory_format=torch.channels_last,
            )
            for _ in range(3):
                model(inputs)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.compute_batches):
                logits = model(inputs)
            end.record()
            end.synchronize()
            if not logits.isfinite().all().item():
                raise RuntimeError("Non-finite compute benchmark logits")
            seconds = start.elapsed_time(end) / 1000
            result = {
                "batch_size": batch_size,
                "seconds": seconds,
                "patches_per_second": batch_size * args.compute_batches / seconds,
                "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
            }
            results.append(result)
            print(json.dumps(result), flush=True)
            del inputs, logits
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8])
    parser.add_argument("--num-workers", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--compute-only", action="store_true")
    parser.add_argument("--compute-batches", type=int, default=20)
    parser.add_argument("--profile", action="store_true", help="Save a cProfile trace for each run")
    args = parser.parse_args()
    if any(n < 1 for n in args.batch_sizes) or any(n < 0 for n in args.num_workers):
        parser.error("Batch sizes must be positive and worker counts nonnegative")
    if args.repeats < 1 or args.compute_batches < 1:
        parser.error("Repeat and compute-batch counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result_fn = args.output_dir / "results.json"
    if args.compute_only:
        save_results(result_fn, compute_ceiling(args))
        return

    gpu = None
    if args.device.startswith("cuda"):
        index = int(args.device.split(":")[1]) if ":" in args.device else 0
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        gpu = visible.split(",")[index] if visible else str(index)
    with rasterio.open(args.input) as src:
        pixels = src.height * src.width
        patches = ((src.height + 383) // 384) * ((src.width + 383) // 384)
    results = []
    failed = False
    for batch_size in args.batch_sizes:
        for workers in args.num_workers:
            for repeat in range(args.repeats):
                name = f"batch_{batch_size}_workers_{workers}_repeat_{repeat}"
                output_fn = args.output_dir / f"{name}.tif"
                log_fn = args.output_dir / f"{name}.log"
                command = [
                    sys.executable, "-u",
                ]
                if args.profile:
                    command.extend(["-m", "cProfile", "-o", str(args.output_dir / f"{name}.prof")])
                command.extend([
                    str(ROOT / "inference_dinov3.py"),
                    "--input", str(args.input.resolve()),
                    "--checkpoint", str(args.checkpoint.resolve()),
                    "--output", str(output_fn.resolve()),
                    "--device", args.device, "--batch-size", str(batch_size),
                    "--num-workers", str(workers),
                ])
                samples = []
                started = time.perf_counter()
                with log_fn.open("x") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        while True:
                            try:
                                returncode = process.wait(timeout=1)
                                break
                            except subprocess.TimeoutExpired:
                                samples.append(sample_resources(process.pid, gpu))
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            process.wait()
                seconds = time.perf_counter() - started
                result = {
                    "batch_size": batch_size, "num_workers": workers, "repeat": repeat,
                    "wall_seconds": seconds, "patches_per_second": patches / seconds,
                    "megapixels_per_second": pixels / seconds / 1e6,
                    "exit_code": returncode, "output": str(output_fn),
                    "command": command, "resource_samples": samples,
                }
                if returncode == 0 and args.reference:
                    result["different_pixels"] = compare_outputs(output_fn, args.reference)
                results.append(result)
                save_results(result_fn, results)
                print(json.dumps({key: value for key, value in result.items()
                                  if key not in ("resource_samples", "command")}), flush=True)
                if returncode:
                    print(f"Failed configuration; see {log_fn}", file=sys.stderr)
                    failed = True
    if failed:
        raise SystemExit("One or more benchmark configurations failed; see results.json")


if __name__ == "__main__":
    main()
