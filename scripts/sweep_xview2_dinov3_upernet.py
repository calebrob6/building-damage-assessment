# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Run the approved 12-trial tier3 sweep, then test only three validation winners.

One UUID-isolated CUDA child per selected GPU; GPU 1 is always excluded.
Memory admission uses the actual cgroup limit, working set and per-job PSS.
Use --prepare-only for sequential preflight without launching training. Existing
directories require --resume --run-dir PATH and matching code/config/manifest.
Winner ties: higher validation damaged F1, lower best validation loss, lower LR.
All twelve trials must finish 15 epochs before any winner is selected or tested.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import fcntl
import hashlib
from importlib.metadata import version
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bda.xview2 import (
    GROUPINGS, assert_sweep_counts, build_manifest, check_manifest, digest,
    read_json, shared_class_weights, validate_manifest, write_json,
)
from bda.sweep_resources import (
    GIB, admission, check_pressure, cpu_quota, memory_snapshot, process_tree_pss,
)
from bda.experiment_runner import execute_jobs as execute_experiment_jobs

REPO = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO / "scripts/train_eval_xview2_dinov3_upernet.py"
DEFAULT_GPUS = (4, 5, 6, 7)
ALLOWED_GPUS = (0, 2, 3, 4, 5, 6, 7)
LEARNING_RATES = (1e-5, 3e-5, 1e-4, 3e-4)
RECIPE = {
    "backbone": "dinov3_vits16", "max_epochs": 15, "seed": 0, "batch_size": 16,
    "crop_size": 512, "crops_per_image": 4, "num_workers": 4, "cpu_threads": 2,
    "eval_batch_size": 2, "include_tier3": True, "val_fraction": 0.1,
}
SELECTION_RULE = "highest validation damaged F1; ties: lower best val_loss, then lower initial LR"


class IncompleteTrialError(ValueError):
    """An interrupted trial has not published a complete training result."""


def enumerate_trials():
    return [
        {"id": f"{grouping}/lr_{lr:.0e}", "grouping": grouping, "lr": lr, **RECIPE}
        for grouping in GROUPINGS for lr in LEARNING_RATES
    ]


def query_gpus(selected=DEFAULT_GPUS):
    if (not selected or len(set(selected)) != len(selected)
            or not set(selected).issubset(ALLOWED_GPUS)):
        raise ValueError("Select distinct allowed GPUs; GPU 1 is excluded")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], check=True, text=True, capture_output=True,
    )
    found = {}
    for row in csv.reader(result.stdout.splitlines()):
        index, uuid, memory, utilization = [value.strip() for value in row]
        index = int(index)
        if index in selected:
            found[index] = {"uuid": uuid, "memory_mib": int(memory), "utilization": int(utilization)}
    if set(found) != set(selected) or len({gpu["uuid"] for gpu in found.values()}) != len(selected):
        raise RuntimeError("Requested distinct physical GPUs are unavailable")
    return {index: found[index] for index in selected}


def available_gpu(uuid, expected_map):
    current = query_gpus(tuple(expected_map))
    if {i: g["uuid"] for i, g in current.items()} != expected_map:
        raise RuntimeError("GPU UUID mapping changed")
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        check=True, text=True, capture_output=True,
    )
    occupied = {row[0].strip() for row in csv.reader(result.stdout.splitlines()) if row}
    gpu = next(g for g in current.values() if g["uuid"] == uuid)
    return uuid not in occupied and gpu["memory_mib"] <= 256 and gpu["utilization"] <= 5


def child_environment(uuid, gpu_map):
    if (uuid not in gpu_map.values() or not gpu_map
            or not set(gpu_map).issubset(ALLOWED_GPUS)
            or len(set(gpu_map.values())) != len(gpu_map)):
        raise ValueError("Refusing an unselected GPU or GPU 1")
    environment = os.environ.copy()
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                "GROUP_RANK", "NODE_RANK", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(key, None)
    environment.update({
        "CUDA_VISIBLE_DEVICES": uuid, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
    })
    return environment


class ControllerLock:
    def __init__(self, root, filename=".controller.lock"):
        self.path = Path(root) / filename
        self.stream = None

    def __enter__(self):
        self.stream = self.path.open("a+")
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close()
            raise RuntimeError(f"A controller already holds {self.path}") from None
        self.stream.seek(0)
        self.stream.truncate()
        self.stream.write(str(os.getpid()) + "\n")
        self.stream.flush()
        return self

    def __exit__(self, *_):
        fcntl.flock(self.stream, fcntl.LOCK_UN)
        self.stream.close()


def choose_root(parent, requested=None, resume=False):
    if resume and not requested:
        raise ValueError("--resume requires an explicit --run-dir")
    if requested:
        root = Path(requested).expanduser().resolve()
        if resume:
            if not root.is_dir():
                raise FileNotFoundError(f"No sweep to resume: {root}")
        else:
            root.mkdir(parents=True, exist_ok=False)
        return root
    parent = Path(parent).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    for index in range(1, 100000):
        root = parent / f"run_{index:03d}"
        try:
            root.mkdir()
            return root
        except FileExistsError:
            continue
    raise RuntimeError("No unused run directory available")


def process_start_ticks(pid):
    try:
        # comm may contain spaces/parentheses; starttime is field 22.
        return Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def reject_live_children(state):
    for section in ("trials", "evaluations"):
        for job in state.get(section, {}).values():
            attempt = job.get("attempts", [{}])[-1] if job.get("attempts") else {}
            pid = attempt.get("pid")
            if (pid and attempt.get("process_start_ticks")
                    and process_start_ticks(pid) == attempt["process_start_ticks"]):
                raise RuntimeError(f"Recorded child {pid} is still alive; will not duplicate or terminate it")


def arm_parent_death_signal(parent_pid):
    """Linux children receive SIGTERM if their owning controller disappears."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        os._exit(125)
    if os.getppid() != parent_pid:
        os._exit(125)


def stop_owned_children(active):
    # Only process groups created by this controller's still-live Popen handles.
    for item in active.values():
        if item["process"].poll() is None:
            try:
                os.killpg(item["process"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 30
    for item in active.values():
        process = item["process"]
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        item["log"].close()


def trial_command(root, data_root, trial, uuid, resume=False, checkpoint=None):
    output = root / (f"winners/{trial['grouping']}/evaluation" if checkpoint else trial["id"])
    command = [
        sys.executable, str(TRAIN_SCRIPT), "--xview2-root", str(data_root),
        "--include-tier3", "--grouping", trial["grouping"],
        "--output-dir", str(output), "--gpu", "0", "--require-gpu-uuid", uuid,
        "--manifest", str(root / "manifest.json"),
        "--preflight-report", str(root / "preflight.json"),
        "--class-weights-json", str(root / "class_weights.json"),
        "--evaluation-split", "test" if checkpoint else "val",
    ]
    for key in ("backbone", "max_epochs", "seed", "batch_size", "crop_size",
                "crops_per_image", "num_workers", "eval_batch_size", "val_fraction", "lr"):
        command += ["--" + key.replace("_", "-"), str(trial[key])]
    if checkpoint:
        command += ["--eval-only", "--checkpoint", str(checkpoint)]
    elif resume:
        command.append("--resume")
    return command, output


def validate_trial(root, trial, manifest_hash):
    directory = root / trial["id"]
    state = read_json(directory / "training_state.json")
    if state.get("status") != "complete":
        raise IncompleteTrialError(f"{trial['id']}: training has not completed")
    config = read_json(directory / "config.json")
    metrics = read_json(directory / "val_metrics.json")
    for key in (*RECIPE.keys(), "lr", "grouping"):
        if key == "cpu_threads":
            continue
        if config.get(key) != trial[key]:
            raise ValueError(f"{trial['id']}: mismatched training configuration {key}")
    if (config.get("manifest_hash") != manifest_hash
            or any(config.get(key) is not None for key in ("limit", "limit_train_batches", "limit_val_batches"))
            or config.get("evaluation_split") != "val"):
        raise ValueError(f"{trial['id']}: incomplete/smoke or mismatched manifest")
    expected_steps = 8889 * trial["crops_per_image"] // trial["batch_size"] * trial["max_epochs"]
    if (state.get("status") != "complete" or state.get("epochs_completed") != 15
            or state.get("global_step") != expected_steps
            or state.get("counts") != {"train": 8889, "val": 279, "test": 933}
            or state.get("config_hash") != digest(config)):
        raise ValueError(f"{trial['id']}: training incomplete")
    checkpoint = Path(state["selected_checkpoint"]).resolve()
    if (not checkpoint.is_relative_to((directory / "checkpoints").resolve())
            or not checkpoint.is_file() or not (directory / "checkpoints/last.ckpt").is_file()):
        raise ValueError(f"{trial['id']}: missing or misplaced checkpoint")
    if (metrics.get("split") != "val" or metrics.get("n_val_images") != 279
            or metrics.get("smoke_test") or metrics.get("manifest_hash") != manifest_hash
            or metrics.get("grouping") != trial["grouping"]
            or Path(metrics["checkpoint"]).resolve() != checkpoint):
        raise ValueError(f"{trial['id']}: invalid validation metrics")
    f1, loss = metrics.get("damaged_f1"), state.get("best_val_loss")
    if f1 is None or loss is None or not math.isfinite(f1) or not math.isfinite(loss) or not 0 <= f1 <= 1:
        raise ValueError(f"{trial['id']}: invalid selection metrics")
    return {
        "id": trial["id"], "grouping": trial["grouping"], "lr": trial["lr"],
        "status": "complete", "checkpoint": str(checkpoint),
        "last_checkpoint": str(directory / "checkpoints/last.ckpt"),
        "epochs_completed": state["epochs_completed"], "global_step": state["global_step"],
        "selected_epoch": state["selected_epoch"], "best_val_loss": loss,
        "damaged_f1": f1, "training_seconds": sum(a.get("fit_seconds", 0) for a in state["attempts"]),
        "total_seconds": state["total_seconds"], "validation": metrics,
    }


def select_winners(rows):
    expected = {trial["id"] for trial in enumerate_trials()}
    if len(rows) != 12 or {row["id"] for row in rows} != expected:
        raise ValueError("Winner selection requires all twelve unique approved trials")
    if any(row.get("status") != "complete" or row.get("epochs_completed") != 15
           or row.get("damaged_f1") is None or not math.isfinite(row["damaged_f1"])
           or not math.isfinite(row["best_val_loss"]) for row in rows):
        raise ValueError("Failed/incomplete trials cannot be omitted from winner selection")
    return {grouping: min(
        [row for row in rows if row["grouping"] == grouping],
        key=lambda row: (-row["damaged_f1"], row["best_val_loss"], row["lr"]),
    ) for grouping in GROUPINGS}


def validate_test(path, winner, manifest_hash):
    result = read_json(path)
    if (result.get("split") != "test" or result.get("n_test_images") != 933
            or result.get("smoke_test") or result.get("manifest_hash") != manifest_hash
            or result.get("grouping") != winner["grouping"]
            or Path(result["checkpoint"]).resolve() != Path(winner["checkpoint"]).resolve()
            or result.get("damaged_f1") is None or not math.isfinite(result["damaged_f1"])):
        raise ValueError(f"Invalid full-test winner evaluation: {path}")
    return result


def execute_jobs(root, state, section, jobs, gpu_map, data_root, max_concurrent, validate,
                 gpu_wait_seconds=60, memory_policy=None):
    def command_builder(job, uuid):
        checkpoint = job.get("checkpoint") if section == "evaluations" else None
        trial = job["trial"] if checkpoint else job
        resume = not checkpoint and (root / trial["id"] / "config.json").exists()
        return trial_command(root, data_root, trial, uuid, resume, checkpoint)

    return execute_experiment_jobs(
        root, state, section, jobs, gpu_map, max_concurrent, validate,
        command_builder, sys.modules[__name__], gpu_wait_seconds, memory_policy,
    )


def flattened_metrics(metrics, prefix=""):
    result = {prefix + name: metrics.get(name) for name in ("damaged_f1", "mean_iou", "overall_accuracy")}
    for name in ("background", "undamaged", "damaged"):
        for metric in ("iou", "precision", "recall", "f1"):
            result[f"{prefix}{name}_{metric}"] = metrics["per_class"][name][metric]
    return result


def write_csv(path, rows):
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(path)


def write_summary(root, state, manifest_hash, baselines, rows=None, winners=None, tests=None):
    summary = {
        "status": state["status"], "manifest_hash": manifest_hash,
        "selection_rule": SELECTION_RULE, "trials": rows or list(state["trials"].values()),
        "winners": winners or {}, "winner_test": tests or {}, "original_only_baselines": baselines,
        "limitations": [
            "Validation/test contain original disasters; they do not directly measure new tier3 disaster generalization.",
            "15 epochs on 8889 training images means more updates; this is not an equal-compute ablation.",
            "Single seed; no statistical significance claim.",
        ],
    }
    if rows:
        write_csv(root / "validation_sweep.csv", [{
            **{k: row[k] for k in ("id", "grouping", "lr", "status", "epochs_completed",
                                  "global_step", "selected_epoch", "best_val_loss",
                                  "training_seconds", "total_seconds", "checkpoint", "last_checkpoint")},
            **flattened_metrics(row["validation"]),
        } for row in rows])
    else:
        partial_rows = []
        for trial in enumerate_trials():
            record = state["trials"][trial["id"]]
            row = {
                "id": trial["id"], "grouping": trial["grouping"], "lr": trial["lr"],
                "status": record["status"], "damaged_f1": None, "best_val_loss": None,
                "checkpoint": None, "epochs_completed": None, "global_step": None,
                "attempt_seconds": sum(a.get("seconds", 0) for a in record["attempts"]),
            }
            if record["status"] == "complete":
                try:
                    complete = validate_trial(root, trial, manifest_hash)
                except (OSError, ValueError, KeyError, TypeError):
                    row["status"] = "invalid_artifacts"
                else:
                    for key in ("damaged_f1", "best_val_loss", "checkpoint", "epochs_completed", "global_step"):
                        row[key] = complete[key]
            partial_rows.append(row)
        write_csv(root / "validation_sweep.csv", partial_rows)
    if tests:
        winner_rows = []
        for grouping, metrics in tests.items():
            winner = winners[grouping]
            baseline = baselines[grouping]["metrics"]
            row = {
                "grouping": grouping, "initial_lr": winner["lr"], "checkpoint": winner["checkpoint"],
                "validation_damaged_f1": winner["damaged_f1"], "n_test_images": 933,
                **flattened_metrics(metrics, "test_"), **flattened_metrics(baseline, "baseline_"),
                "damaged_f1_delta": metrics["damaged_f1"] - baseline["damaged_f1"],
                "mean_iou_delta": metrics["mean_iou"] - baseline["mean_iou"],
            }
            winner_rows.append(row)
        write_csv(root / "winner_test.csv", winner_rows)
        summary["baseline_comparison"] = winner_rows
    write_json(root / "summary.json", summary)


def validate_resource_change(previous, current):
    """Allow scheduler changes, never changes to training, data, metrics or versions."""
    mutable = {"schema_version", "max_concurrent", "gpu_map", "gpu_order",
               "memory_policy", "code_hashes"}
    if ({k: v for k, v in previous.items() if k not in mutable}
            != {k: v for k, v in current.items() if k not in mutable}):
        raise ValueError("Resource changes cannot alter training, dataset, versions or baselines")
    scheduler_files = {Path(__file__).name, "sweep_resources.py", "experiment_runner.py"}
    if ({k: v for k, v in previous["code_hashes"].items() if k not in scheduler_files}
            != {k: v for k, v in current["code_hashes"].items() if k not in scheduler_files}):
        raise ValueError("Resource changes cannot alter training/data/model code")
    if previous["schema_version"] not in (1, 2) or current["schema_version"] != 2:
        raise ValueError("Unsupported scheduler configuration migration")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xview2-root", default=os.path.expanduser("~/data/xview2"))
    p.add_argument("--output-parent", default="outputs/xview2_dinov3_upernet_tier3_lr_sweep")
    p.add_argument("--run-dir", help="New exact output directory, or existing directory with --resume")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--prepare-only", action="store_true", help="Preflight/config only; do not start GPU jobs")
    p.add_argument("--max-concurrent", type=int, choices=range(1, 8),
                   help="Default: saved value on resume, otherwise 4")
    p.add_argument("--gpus", type=int, nargs="+", choices=ALLOWED_GPUS,
                   help="Physical GPUs; default: saved selection or 4 5 6 7")
    p.add_argument("--memory-reserve-gib", type=float, help="Cgroup safety reserve (default: 16 GiB)")
    p.add_argument("--memory-per-job-gib", type=float, help="Reserved peak RAM per job (default: 8 GiB)")
    p.add_argument("--reconfigure-resources", action="store_true",
                   help="Explicit scheduler-only migration of an existing run; requires --resume")
    p.add_argument("--reuse-preflight", action="store_true",
                   help="Reuse a matching validated manifest report on resume")
    p.add_argument("--baseline-root", default=str(REPO / "outputs"))
    args = p.parse_args()
    if (args.reconfigure_resources or args.reuse_preflight) and not args.resume:
        raise ValueError("Resource reconfiguration and preflight reuse require --resume")
    (REPO / "outputs").mkdir(exist_ok=True)
    root = choose_root(args.output_parent, args.run_dir, args.resume)
    print(f"Sweep output: {root}", flush=True)
    with ControllerLock(REPO / "outputs", ".xview2-tier3-sweep.lock"), ControllerLock(root):
        old_state = read_json(root / "state.json") if args.resume else None
        stored_config = read_json(root / "config.json") if args.resume else {}
        previous_config = stored_config
        if old_state:
            reject_live_children(old_state)
            if old_state["config_hash"] != digest(stored_config):
                if not args.reconfigure_resources:
                    raise ValueError("Stored configuration/state hash mismatch")
                # A previous migration may have stopped between the two atomic writes.
                previous_config = read_json(root / "config_history" / f"{old_state['config_hash']}.json")
                if digest(previous_config) != old_state["config_hash"]:
                    raise ValueError("Invalid previous configuration snapshot")
                validate_resource_change(previous_config, stored_config)
        selected_gpus = args.gpus or stored_config.get(
            "gpu_order", [int(key) for key in stored_config.get("gpu_map", {})] or list(DEFAULT_GPUS)
        )
        max_concurrent = args.max_concurrent or stored_config.get("max_concurrent", min(4, len(selected_gpus)))
        if max_concurrent > len(selected_gpus):
            raise ValueError("max_concurrent exceeds the selected GPU count")
        memory_policy = dict(stored_config.get(
            "memory_policy", {"reserve_bytes": 16 * GIB, "per_job_bytes": 8 * GIB}
        ))
        for key, value in (("reserve_bytes", args.memory_reserve_gib),
                           ("per_job_bytes", args.memory_per_job_gib)):
            if value is not None:
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("Memory budgets must be finite and positive")
                memory_policy[key] = int(value * GIB)
        manifest = build_manifest(args.xview2_root, include_tier3=True)
        assert_sweep_counts(manifest)
        gpu_map = {i: gpu["uuid"] for i, gpu in query_gpus(selected_gpus).items()}
        baselines = {}
        for grouping in GROUPINGS:
            path = Path(args.baseline_root).expanduser().resolve() / f"xview2_dinov3_upernet_{grouping}/test_metrics.json"
            metrics = read_json(path)
            if metrics["grouping"] != grouping or metrics["n_test_images"] != 933:
                raise ValueError(f"Invalid original-only baseline: {path}")
            baselines[grouping] = {"path": str(path), "metrics": metrics}
        code_paths = [
            Path(__file__), TRAIN_SCRIPT, REPO / "bda/xview2.py",
            REPO / "bda/trainers.py", REPO / "bda/dinov3_upernet.py",
            REPO / "bda/sweep_resources.py",
            REPO / "bda/experiment_runner.py",
        ]
        config = {
            "schema_version": 2, "trials": enumerate_trials(), "max_concurrent": max_concurrent,
            "xview2_root": str(Path(args.xview2_root).expanduser().resolve()),
            "manifest_hash": manifest["manifest_hash"], "gpu_map": {str(k): v for k, v in gpu_map.items()},
            "gpu_order": list(selected_gpus), "memory_policy": memory_policy,
            "selection_rule": SELECTION_RULE, "baseline_hash": digest(baselines),
            "code_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths},
            "python": str(Path(sys.executable).resolve()),
            "versions": {name: version(name) for name in (
                "torch", "lightning", "torchgeo", "transformers", "kornia", "numpy", "Pillow",
            )},
        }
        if args.resume:
            check_manifest(read_json(root / "manifest.json"), manifest)
            state = old_state
            if args.reconfigure_resources:
                validate_resource_change(previous_config, config)
                if previous_config != config:
                    old_hash = digest(previous_config)
                    write_json(root / "config_history" / f"{old_hash}.json", previous_config)
                    state.setdefault("resource_changes", []).append({
                        "at": time.time(), "previous_config_hash": old_hash,
                        "config_hash": digest(config), "gpu_order": list(selected_gpus),
                        "max_concurrent": max_concurrent, "memory_policy": memory_policy,
                        "reason": "explicit scheduler-only reconfiguration",
                    })
                    write_json(root / "config.json", config)
                    state["config_hash"] = digest(config)
            elif config != stored_config:
                raise ValueError("Cannot resume: config/code/GPU mismatch; resource changes require --reconfigure-resources")
        else:
            write_json(root / "config.json", config)
            write_json(root / "manifest.json", manifest)
            state = {
                "status": "preparing", "started_at": time.time(), "config_hash": digest(config),
                "trials": {trial["id"]: {**trial, "status": "pending", "attempts": []}
                           for trial in enumerate_trials()},
                "evaluations": {grouping: {"id": grouping, "status": "pending", "attempts": []}
                                for grouping in GROUPINGS},
            }
        write_json(root / "state.json", state)
        if args.prepare_only and state["status"] == "complete":
            print("Sweep already complete; existing summary and winner results retained.", flush=True)
            return
        rows = winners = tests = None
        controller_started = time.monotonic()
        controller_attempt = {"started_at": time.time(), "resume": args.resume}
        if state.get("error"):
            if state.get("controller_attempts"):
                state["controller_attempts"][-1].setdefault("error", state["error"])
            del state["error"]
        state.setdefault("controller_attempts", []).append(controller_attempt)
        try:
            if args.reuse_preflight:
                preflight = read_json(root / "preflight.json")
                if (preflight.get("status") != "validated"
                        or preflight.get("manifest_hash") != manifest["manifest_hash"]
                        or preflight.get("image_size") != [1024, 1024]
                        or preflight.get("target_codes") != [0, 1, 2, 3, 4]
                        or preflight.get("images_validated") != 10101):
                    raise ValueError("Existing preflight does not validate this unchanged manifest")
                controller_attempt["preflight_reused"] = True
                print("Reusing validated preflight with matching file fingerprints.", flush=True)
            else:
                preflight = validate_manifest(args.xview2_root, manifest)
            write_json(root / "preflight.json", preflight)
            weights = shared_class_weights(args.xview2_root, manifest)
            if args.resume and (root / "class_weights.json").exists():
                if read_json(root / "class_weights.json") != weights:
                    raise ValueError("Cannot resume: training class weights changed")
            write_json(root / "class_weights.json", weights)
            write_json(root / "baselines.json", baselines)
            for trial in enumerate_trials():
                path = root / trial["id"] / "trial_config.json"
                trial_config = {**trial, "manifest_hash": manifest["manifest_hash"],
                                "class_weights": weights["groupings"][trial["grouping"]]}
                if args.resume and path.exists() and read_json(path) != trial_config:
                    raise ValueError(f"Cannot resume: trial configuration mismatch at {path}")
                write_json(path, trial_config)
            if args.prepare_only:
                state["status"] = "prepared"
                return
            state["status"] = "training"
            state["memory"] = memory_snapshot()
            state["cpu_quota_cores"] = cpu_quota()
            print(f"Cgroup RAM limit: {state['memory']['limit_bytes'] / GIB:.1f} GiB; "
                  f"CPU quota: {state['cpu_quota_cores']} cores; slots: {max_concurrent}", flush=True)
            write_json(root / "state.json", state)
            execute_jobs(root, state, "trials", enumerate_trials(), gpu_map,
                         config["xview2_root"], max_concurrent,
                         lambda trial: validate_trial(root, trial, manifest["manifest_hash"]),
                         memory_policy=memory_policy)
            rows = [validate_trial(root, trial, manifest["manifest_hash"]) for trial in enumerate_trials()]
            winners = select_winners(rows)
            selection = {"selection_rule": SELECTION_RULE, "manifest_hash": manifest["manifest_hash"],
                         "winners": winners, "status": "selected_using_validation_only"}
            if args.resume and (root / "selection.json").exists():
                if read_json(root / "selection.json") != selection:
                    raise ValueError("Previously selected winners changed; refusing test re-selection")
            write_json(root / "selection.json", selection)
            write_summary(root, state, manifest["manifest_hash"], baselines, rows, winners)
            state["status"] = "evaluating_winners"
            jobs = [
                {"id": grouping, "checkpoint": winner["checkpoint"],
                 "trial": next(t for t in enumerate_trials() if t["id"] == winner["id"])}
                for grouping, winner in winners.items()
            ]
            execute_jobs(
                root, state, "evaluations", jobs, gpu_map, config["xview2_root"], max_concurrent,
                lambda job: validate_test(root / f"winners/{job['id']}/evaluation/test_metrics.json",
                                          winners[job["id"]], manifest["manifest_hash"]),
                memory_policy=memory_policy,
            )
            tests = {
                grouping: validate_test(root / f"winners/{grouping}/evaluation/test_metrics.json",
                                         winner, manifest["manifest_hash"])
                for grouping, winner in winners.items()
            }
            state.update({"status": "complete", "finished_at": time.time()})
        finally:
            error = sys.exc_info()[1]
            if error is not None:
                state["status"] = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
                state["error"] = f"{type(error).__name__}: {error}"
            controller_attempt.update({
                "finished_at": time.time(), "seconds": time.monotonic() - controller_started,
                "status": state["status"],
            })
            state["controller_seconds"] = sum(
                attempt.get("seconds", 0) for attempt in state["controller_attempts"]
            )
            state["updated_at"] = time.time()
            write_json(root / "state.json", state)
            write_summary(root, state, manifest["manifest_hash"], baselines, rows, winners, tests)
        print(f"Complete: {root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Controller received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    main()
