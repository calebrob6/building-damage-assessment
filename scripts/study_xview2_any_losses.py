# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Six fresh any-damage loss trajectories, guarded 30->60 promotion, then test.

CE runs to 60; alternatives stop at 30. At most two alternatives are promoted
using immutable 30-epoch evidence, even while CE continues. Never use test
metrics for ranking. GPU 1 is excluded; the shared runner enforces cgroup/PSS
admission and cleans up only its own attached process trees.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bda.experiment_runner import durable_json, execute_jobs
from bda.xview2 import (
    assert_sweep_counts, build_manifest, check_manifest, class_weight_estimate,
    digest, read_json, split_paths, write_json,
)
from scripts import sweep_xview2_dinov3_upernet as scheduler

REPO = Path(__file__).resolve().parents[1]
OLD_RUN = REPO / "outputs/xview2_dinov3_upernet_tier3_lr_sweep/run_001"
ARCHIVED_CHECKPOINT = OLD_RUN / "any/lr_3e-05/checkpoints/best-epoch=13-val_loss=0.1773.ckpt"
LOSSES = {
    "ce": {},
    "ce_dice": {"foreground_classes": [1, 2], "region_weight": 1.0},
    "boundary_ce": {"boundary_class": 1, "boundary_radius": 3, "boundary_multiplier": 4.0},
    "boundary_ce_dice": {"foreground_classes": [1, 2], "region_weight": 1.0,
                         "boundary_class": 1, "boundary_radius": 3, "boundary_multiplier": 4.0},
    "focal_dice": {"foreground_classes": [1, 2], "region_weight": 1.0, "focal_gamma": 2.0},
    "ce_tversky": {"foreground_classes": [1, 2], "region_weight": 1.0,
                   "tversky_alpha": 0.7, "tversky_beta": 0.3},
}
RULE = ("Highest undamaged boundary F1 at 2px; ties: higher undamaged IoU, "
        "higher damaged F1, earlier epoch, then recipe name. Guard each pixel "
        "metric within 0.01 of both archived and eligible matched-budget CE.")
RECIPE = {"grouping": "any", "backbone": "dinov3_vits16", "lr": 3e-5, "seed": 0,
          "batch_size": 16, "crop_size": 512, "crops_per_image": 4,
          "max_epochs": 60, "eval_batch_size": 2, "val_fraction": 0.1}


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def initial_jobs():
    return [{"id": f"{name}/stage_{60 if name == 'ce' else 30:03d}",
             "recipe": name, "budget": 60 if name == "ce" else 30}
            for name in LOSSES]


def numbers(candidate):
    metrics = candidate["metrics"]
    return (metrics["geometry"]["undamaged_boundary_f1"],
            metrics["per_class"]["undamaged"]["iou"], metrics["damaged_f1"])


def rank(candidate):
    boundary, iou, damage = numbers(candidate)
    return (-boundary, -iou, -damage, candidate["epoch"], candidate.get("recipe", ""))


def select_candidate(candidates, archived, ce=None):
    _, archived_iou, archived_damage = numbers(archived)
    floors = {"undamaged_iou": archived_iou - 0.01, "damaged_f1": archived_damage - 0.01}
    if ce is not None:
        _, ce_iou, ce_damage = numbers(ce)
        floors = {"undamaged_iou": max(floors["undamaged_iou"], ce_iou - 0.01),
                  "damaged_f1": max(floors["damaged_f1"], ce_damage - 0.01)}
    eligible = []
    for candidate in candidates:
        boundary, iou, damage = numbers(candidate)
        if (all(value is not None and math.isfinite(value) for value in (boundary, iou, damage))
                and iou >= floors["undamaged_iou"] and damage >= floors["damaged_f1"]):
            eligible.append(candidate)
    return {"status": "eligible" if eligible else "ineligible",
            "candidate": min(eligible, key=rank) if eligible else None,
            "eligible_count": len(eligible), "floors": floors,
            "ce_reference": "eligible" if ce is not None else "archived_floors_only"}


def budget_selection(catalogs, budget, archived):
    if set(catalogs) != set(LOSSES):
        raise ValueError("Budget selection requires evidence from all six recipes")
    filtered = {name: [dict(row, recipe=name) for row in rows if row["epochs_completed"] <= budget]
                for name, rows in catalogs.items()}
    ce = select_candidate(filtered["ce"], archived)
    decisions = {"ce": ce}
    for name in LOSSES:
        if name != "ce":
            decisions[name] = select_candidate(filtered[name], archived, ce["candidate"])
    return {"budget": budget, "rule": RULE, "decisions": decisions}


def promotions(selection, archived):
    if selection["budget"] != 30 or set(selection["decisions"]) != set(LOSSES):
        raise ValueError("Promotion requires complete frozen thirty-epoch evidence")
    for name, decision in selection["decisions"].items():
        row = decision["candidate"]
        if row is not None and (row["epochs_completed"] > 30 or row["recipe"] != name):
            raise ValueError("Promotion candidate is outside its frozen thirty-epoch recipe")
    ce = selection["decisions"]["ce"]["candidate"]
    threshold = max(numbers(row)[0] for row in (archived, ce) if row is not None)
    qualifying = []
    outcomes = {}
    for name, decision in selection["decisions"].items():
        if name == "ce":
            continue
        candidate = decision["candidate"]
        qualifies = candidate is not None and numbers(candidate)[0] > threshold
        outcomes[name] = {"qualifies": qualifies, "boundary_reference": threshold,
                          "reason": "strict_shape_improvement" if qualifies else "no_guarded_strict_improvement"}
        if qualifies:
            qualifying.append(candidate)
    promoted = [row["recipe"] for row in sorted(qualifying, key=rank)[:2]]
    for name in outcomes:
        outcomes[name]["promoted"] = name in promoted
    return {"promoted": promoted, "outcomes": outcomes, "selection": selection}


def archived_candidate(reference):
    return {"recipe": "archived", "epoch": 13, "epochs_completed": 14, "budget": 15,
            "checkpoint": str(Path(reference["checkpoint"]).resolve()),
            "sha256": reference["checkpoint_sha256"], "metrics": reference}


def read_stage(root, recipe, budget, manifest_hash, verify_checkpoint=False):
    directory = root / "recipes" / recipe
    config = read_json(directory / "config.json")
    marker = read_json(directory / "stages" / f"epoch_{budget:03d}" / "ready.json")
    if (marker.get("status") != "ready" or marker.get("epochs_completed") != budget
            or marker.get("horizon") != 60 or marker.get("manifest_hash") != manifest_hash
            or marker.get("config_hash") != digest(config)):
        raise ValueError(f"Invalid {recipe} stage {budget} marker/configuration")
    for key, value in RECIPE.items():
        if config.get(key) != value:
            raise ValueError(f"Recipe configuration changed: {recipe}/{key}")
    if (config.get("loss") != recipe or config.get("loss_options") != LOSSES[recipe]
            or config.get("geometry_metrics") is not True
            or config.get("validation_precision") != "fp32"
            or config.get("epoch_candidates") is not True
            or config.get("milestones") != [15, 30, 60]
            or any(config.get(k) is not None for k in ("limit", "limit_train_batches", "limit_val_batches"))):
        raise ValueError("Invalid study loss/geometry/budget configuration")
    stage_directory = (directory / "stages" / f"epoch_{budget:03d}").resolve()
    continuation, catalog_path = Path(marker["continuation_checkpoint"]), Path(marker["catalog"])
    if (continuation.resolve().parent != stage_directory or catalog_path.resolve().parent != stage_directory
            or not continuation.is_file()):
        raise ValueError("Stage artifacts escaped their immutable namespace")
    catalog = read_json(catalog_path)
    if digest(catalog) != marker["catalog_hash"] or catalog["config_hash"] != digest(config):
        raise ValueError("Frozen candidate catalog changed")
    rows = catalog["candidates"]
    if (catalog["budget"] != budget or len(rows) != budget
            or [row["epochs_completed"] for row in rows] != list(range(1, budget + 1))):
        raise ValueError("Stage evidence is incomplete or contains later epochs")
    for row in rows:
        confusion = row["metrics"]["confusion_matrix"]
        if (row["epoch"] != row["epochs_completed"] - 1
                or row["global_step"] != 2222 * row["epochs_completed"]
                or row["n_val_images"] != 279
                or Path(row["checkpoint"]).resolve().parent != (directory / "candidates").resolve()
                or not Path(row["checkpoint"]).is_file()
                or len(confusion) != 3 or any(len(values) != 3 for values in confusion)
                or sum(map(sum, confusion)) != 279 * 1024 * 1024):
            raise ValueError("Candidate has missing/invalid native-validation evidence")
    if marker["global_step"] != budget * 2222:
        raise ValueError("Stage optimizer-step count is incomplete")
    if verify_checkpoint and file_hash(continuation) != marker["continuation_sha256"]:
        raise ValueError("Frozen continuation checkpoint changed")
    return rows


def validate_job(root, job, manifest_hash):
    state = read_json(root / "recipes" / job["recipe"] / "training_state.json")
    if state.get("stage_epochs", 0) > job["budget"]:
        promotion = read_json(root / "promotion.json")
        if job["recipe"] not in promotion["promoted"] or job["budget"] != 30:
            raise ValueError("Unapproved trajectory extension")
        return read_stage(root, job["recipe"], job["budget"], manifest_hash, verify_checkpoint=True)
    if state.get("status") not in ("stage_complete", "complete"):
        raise scheduler.IncompleteTrialError("Trajectory has not completed its active stage")
    # A promoted trajectory also proves that its earlier immutable stage completed.
    if state.get("epochs_completed", 0) < job["budget"]:
        raise scheduler.IncompleteTrialError("Requested trajectory budget is incomplete")
    config = read_json(root / "recipes" / job["recipe"] / "config.json")
    if (state.get("epochs_completed") != job["budget"] or state.get("stage_epochs") != job["budget"]
            or state.get("global_step") != 2222 * job["budget"] or state.get("config_hash") != digest(config)
            or state.get("counts") != {"train": 8889, "val": 279, "test": 933}):
        raise ValueError("Completed trajectory metadata does not match its requested stage")
    rows = read_stage(root, job["recipe"], job["budget"], manifest_hash, verify_checkpoint=True)
    return rows


def train_command(root, data_root, job, uuid, workers):
    output = root / "recipes" / job["recipe"]
    command = [
        sys.executable, str(scheduler.TRAIN_SCRIPT), "--xview2-root", str(data_root),
        "--include-tier3", "--output-dir", str(output), "--gpu", "0", "--require-gpu-uuid", uuid,
        "--manifest", str(root / "manifest.json"), "--preflight-report", str(root / "preflight.json"),
        "--class-weights-json", str(root / "class_weights.json"), "--evaluation-split", "none",
        "--loss", job["recipe"], "--loss-options", json.dumps(LOSSES[job["recipe"]]),
        "--geometry-metrics", "--validation-precision", "fp32", "--epoch-candidates",
        "--milestones", "15", "30", "60", "--stage-epochs", str(job["budget"]),
        "--num-workers", str(workers),
    ]
    for key, value in RECIPE.items():
        command += ["--" + key.replace("_", "-"), str(value)]
    if (output / "config.json").exists():
        command.append("--resume")
        prior = read_json(output / "training_state.json")
        if prior.get("stage_epochs") != job["budget"]:
            if prior.get("status") != "stage_complete":
                raise ValueError("Cannot promote a failed or incomplete trajectory")
            command.append("--promote-stage")
    return command, output


def discover_promotions(root, state, archived, manifest_hash):
    path = root / "promotion.json"
    if not path.exists():
        if any(state["trials"][job["id"]]["status"] != "complete"
               for job in initial_jobs() if job["recipe"] != "ce"):
            return []
        if not (root / "recipes/ce/stages/epoch_030/ready.json").exists():
            return []
        catalogs = {name: read_stage(root, name, 30, manifest_hash, True) for name in LOSSES}
        result = promotions(budget_selection(catalogs, 30, archived), archived)
        result.update({"manifest_hash": manifest_hash, "config_hash": state["config_hash"],
                       "evidence_budget": 30, "status": "frozen_validation_only"})
        durable_json(path, result, immutable=True)
        print("Frozen promotions:", result["promoted"], flush=True)
    result = read_json(path)
    if (result["config_hash"] != state["config_hash"] or result["manifest_hash"] != manifest_hash
            or len(result["promoted"]) > 2 or len(set(result["promoted"])) != len(result["promoted"])
            or any(name not in LOSSES or name == "ce" for name in result["promoted"])):
        raise ValueError("Invalid or changed frozen promotion plan")
    if state.get("promotion_hash"):
        if digest(result) != state["promotion_hash"]:
            raise ValueError("Frozen promotion artifact changed")
    else:
        catalogs = {name: read_stage(root, name, 30, manifest_hash) for name in LOSSES}
        expected = promotions(budget_selection(catalogs, 30, archived), archived)
        expected.update({"manifest_hash": manifest_hash, "config_hash": state["config_hash"],
                         "evidence_budget": 30, "status": "frozen_validation_only"})
        if result != expected:
            raise ValueError("Promotion does not match frozen guarded evidence")
        state["promotion_hash"] = digest(result)
    return [{"id": f"{name}/stage_060", "recipe": name, "budget": 60}
            for name in result["promoted"]]


def final_choices(root, archived, manifest_hash):
    promotion = read_json(root / "promotion.json")
    controls = {}
    for budget in (15, 30, 60):
        rows = [dict(row, recipe="ce") for row in read_stage(root, "ce", budget, manifest_hash)]
        controls[str(budget)] = select_candidate(rows, archived)
    alternatives, candidates = {}, [archived]
    if controls["60"]["candidate"]:
        candidates.append(controls["60"]["candidate"])
    for name in LOSSES:
        if name == "ce":
            continue
        budget = 60 if name in promotion["promoted"] else 30
        rows = [dict(row, recipe=name) for row in read_stage(root, name, budget, manifest_hash)]
        decision = select_candidate(rows, archived, controls[str(budget)]["candidate"])
        alternatives[name] = {"budget": budget, **decision}
        if decision["candidate"]:
            candidates.append(decision["candidate"])
    winner = min(candidates, key=rank)
    comparisons = {"archived": archived, "final_candidate": winner}
    comparisons.update({f"ce_{budget}": value["candidate"] for budget, value in controls.items()
                        if value["candidate"] is not None})
    identities = {}
    for role, candidate in comparisons.items():
        path = Path(candidate["checkpoint"]).resolve()
        checksum = file_hash(path)
        if checksum != candidate["sha256"]:
            raise ValueError("Selected inference checkpoint changed")
        entry = identities.setdefault(checksum, {"checkpoint": str(path), "sha256": checksum, "roles": []})
        entry["roles"].append(role)
    return {"status": "frozen_before_test", "manifest_hash": manifest_hash, "rule": RULE,
            "ce_controls": controls, "alternatives": alternatives, "final_candidate": winner,
            "test_identities": identities, "promoted": promotion["promoted"],
            "aggregate_epochs": 60 + 5 * 30 + 30 * len(promotion["promoted"])}


def test_command(root, data_root, job, uuid, workers):
    output = root / "test" / job["id"]
    command = [
        sys.executable, str(scheduler.TRAIN_SCRIPT), "--xview2-root", str(data_root),
        "--include-tier3", "--grouping", "any", "--output-dir", str(output), "--gpu", "0",
        "--require-gpu-uuid", uuid, "--manifest", str(root / "manifest.json"),
        "--preflight-report", str(root / "preflight.json"),
        "--class-weights-json", str(root / "class_weights.json"),
        "--eval-only", "--checkpoint", job["checkpoint"], "--evaluation-split", "test",
        "--geometry-metrics", "--validation-precision", "fp32", "--num-workers", str(workers),
    ]
    return command, output


def validate_test(root, job, manifest_hash):
    metrics = read_json(root / "test" / job["id"] / "test_metrics.json")
    if (metrics.get("split") != "test" or metrics.get("n_test_images") != 933
            or metrics.get("grouping") != "any" or metrics.get("smoke_test")
            or metrics.get("manifest_hash") != manifest_hash
            or Path(metrics["checkpoint"]).resolve() != Path(job["checkpoint"]).resolve()
            or not metrics.get("geometry")):
        raise ValueError("Incomplete or mismatched selected test evaluation")
    return metrics


def metric_columns(metrics, prefix=""):
    return {
        prefix + "damaged_f1": metrics["damaged_f1"],
        prefix + "mean_iou": metrics.get("mean_iou"),
        prefix + "overall_accuracy": metrics.get("overall_accuracy"),
        **{prefix + key: value for key, value in metrics["geometry"].items()},
    }


def publish_tables(root, selection, tests):
    rows = []
    for name in LOSSES:
        path = root / "recipes" / name / "candidates.json"
        if not path.exists():
            continue
        for row in read_json(path)["candidates"]:
            rows.append({"recipe": name, "epoch": row["epoch"],
                         "epochs_completed": row["epochs_completed"], "global_step": row["global_step"],
                         "checkpoint": row["checkpoint"], "sha256": row["sha256"],
                         "val_loss_within_recipe_only": row["val_loss"],
                         **metric_columns(row["metrics"], "val_")})
    if rows:
        scheduler.write_csv(root / "validation_candidates.csv", rows)
    if selection:
        decisions = [(f"ce_{budget}", "ce", int(budget), decision)
                     for budget, decision in selection["ce_controls"].items()]
        decisions += [(name, name, decision["budget"], decision)
                      for name, decision in selection["alternatives"].items()]
        choices = []
        for role, name, budget, decision in decisions:
            candidate = decision["candidate"]
            choices.append({
                "role": role, "recipe": name, "budget": budget, "status": decision["status"],
                "checkpoint": candidate["checkpoint"] if candidate else None,
                "selected_epoch": candidate["epoch"] if candidate else None,
                "boundary_f1": numbers(candidate)[0] if candidate else None,
                "undamaged_iou": numbers(candidate)[1] if candidate else None,
                "damaged_f1": numbers(candidate)[2] if candidate else None,
                "undamaged_iou_floor": decision["floors"]["undamaged_iou"],
                "damaged_f1_floor": decision["floors"]["damaged_f1"],
            })
        scheduler.write_csv(root / "guarded_selections.csv", choices)
    if tests:
        archived_id = next(identity for identity, item in selection["test_identities"].items()
                           if "archived" in item["roles"])
        baseline = tests[archived_id]
        rows = []
        for identity, metrics in tests.items():
            for role in selection["test_identities"][identity]["roles"]:
                rows.append({
                    "role": role, "checkpoint_identity": identity, "checkpoint": metrics["checkpoint"],
                    "n_test_images": 933, **metric_columns(metrics, "test_"),
                    "boundary_f1_delta_vs_archived": (
                        metrics["geometry"]["undamaged_boundary_f1"]
                        - baseline["geometry"]["undamaged_boundary_f1"]
                    ),
                    "damaged_f1_delta_vs_archived": metrics["damaged_f1"] - baseline["damaged_f1"],
                    "undamaged_iou_delta_vs_archived": (
                        metrics["per_class"]["undamaged"]["iou"]
                        - baseline["per_class"]["undamaged"]["iou"]
                    ),
                })
        scheduler.write_csv(root / "selected_test_comparisons.csv", rows)


def validate_reference(reference, panels, manifest):
    if (reference.get("grouping") != "any" or reference.get("split") != "val"
            or reference.get("n_val_images") != 279 or reference.get("manifest_hash") != manifest["manifest_hash"]
            or reference.get("geometry_class") != 1 or reference.get("boundary_tolerance") != 2
            or reference.get("boundary_distance") != "chebyshev"
            or reference.get("precision") != "float32"
            or Path(reference["checkpoint"]).resolve() != ARCHIVED_CHECKPOINT.resolve()
            or file_hash(reference["checkpoint"]) != reference.get("checkpoint_sha256")):
        raise ValueError("Expected the fixed archived fifteen-epoch any validation/geometry reference")
    if (abs(reference["damaged_f1"] - 0.5687644785352038) > 1e-12
            or abs(reference["per_class"]["undamaged"]["iou"] - 0.538560685417643) > 1e-12):
        raise ValueError("Archived pixel-metric reference changed")
    if any(value is None or not math.isfinite(value) for value in numbers(archived_candidate(reference))):
        raise ValueError("Archived reference lacks finite selection metrics")
    identifiers = {entry["image"] for entry in manifest["splits"]["val"]}
    if (panels.get("manifest_hash") != manifest["manifest_hash"] or not panels.get("panels")
            or any(row["image"] not in identifiers for row in panels["panels"])):
        raise ValueError("Panel manifest must contain only fixed original validation images")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xview2-root", default=str(Path.home() / "data/xview2"))
    p.add_argument("--source-run", type=Path, default=OLD_RUN)
    p.add_argument("--reference-json", type=Path, required=True)
    p.add_argument("--panel-manifest", type=Path, required=True)
    p.add_argument("--output-parent", default="outputs/xview2_any_loss_study")
    p.add_argument("--run-dir")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--gpus", type=int, nargs="+", choices=scheduler.ALLOWED_GPUS,
                   default=list(scheduler.ALLOWED_GPUS))
    p.add_argument("--max-concurrent", type=int, default=6)
    p.add_argument("--cpu-threads", type=int, choices=[1, 2], default=1)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--memory-per-job-gib", type=float, default=8)
    p.add_argument("--memory-reserve-gib", type=float, default=16)
    args = p.parse_args()
    if (not 1 <= args.max_concurrent <= min(6, len(args.gpus)) or args.num_workers < 0
            or not math.isfinite(args.memory_per_job_gib) or args.memory_per_job_gib < 8
            or not math.isfinite(args.memory_reserve_gib) or args.memory_reserve_gib < 16):
        raise ValueError("Unsafe concurrency, worker or memory limits")
    memory, quota = scheduler.memory_snapshot(), scheduler.cpu_quota()
    if quota is None or args.max_concurrent * args.cpu_threads > quota:
        raise ValueError("Concurrent CPU-thread allocation exceeds the actual cgroup quota")
    manifest = build_manifest(args.xview2_root, include_tier3=True)
    assert_sweep_counts(manifest)
    check_manifest(read_json(args.source_run / "manifest.json"), manifest)
    preflight = read_json(args.source_run / "preflight.json")
    if (preflight.get("status") != "validated" or preflight.get("manifest_hash") != manifest["manifest_hash"]
            or preflight.get("images_validated") != 10101 or preflight.get("image_size") != [1024, 1024]
            or preflight.get("target_codes") != [0, 1, 2, 3, 4]):
        raise ValueError("Preflight does not validate the current unchanged data fingerprints")
    weights = read_json(args.source_run / "class_weights.json")
    values, counts = class_weight_estimate(split_paths(args.xview2_root, manifest)["train"], "any")
    if (weights["manifest_hash"] != manifest["manifest_hash"]
            or weights["groupings"]["any"] != {"weights": values.tolist(), "pixel_counts": counts.astype(int).tolist()}):
        raise ValueError("Historical training-mask estimator changed")
    reference, panels = read_json(args.reference_json), read_json(args.panel_manifest)
    validate_reference(reference, panels, manifest)
    gpu_map = {i: row["uuid"] for i, row in scheduler.query_gpus(args.gpus).items()}
    policy = {"per_job_bytes": int(args.memory_per_job_gib * scheduler.GIB),
              "reserve_bytes": int(args.memory_reserve_gib * scheduler.GIB)}
    code = [Path(__file__), scheduler.TRAIN_SCRIPT, REPO / "bda/experiment_runner.py",
            REPO / "scripts/sweep_xview2_dinov3_upernet.py"]
    code += [REPO / "bda" / name for name in (
        "xview2.py", "trainers.py", "dinov3_upernet.py", "sweep_resources.py",
        "losses.py", "boundaries.py", "boundary_metrics.py",
    )]
    config = {
        "schema_version": 1, "recipes": LOSSES, "training": RECIPE, "selection": RULE,
        "initial_jobs": initial_jobs(), "max_promotions": 2, "max_aggregate_epochs": 270,
        "manifest_hash": manifest["manifest_hash"], "reference_hash": digest(reference),
        "panel_hash": digest(panels), "class_weights": weights["groupings"]["any"],
        "gpu_order": args.gpus, "gpu_map": {str(k): v for k, v in gpu_map.items()},
        "max_concurrent": args.max_concurrent, "num_workers": args.num_workers,
        "cpu_threads": args.cpu_threads, "memory_policy": policy,
        "xview2_root": str(Path(args.xview2_root).resolve()),
        "code_hashes": {str(path.relative_to(REPO)): file_hash(path) for path in code},
        "python": str(Path(sys.executable).resolve()),
        "versions": {name: version(name) for name in (
            "torch", "lightning", "torchgeo", "transformers", "kornia", "numpy", "Pillow",
        )},
        "validation_precision": "fp32", "boundary_class": 1, "boundary_tolerance": 2,
        "scheduler": {"name": "ReduceLROnPlateau", "factor": 0.1, "patience": 5, "monitor": "val_loss"},
        "preflight": {"source_run": str(args.source_run.resolve()), "report_hash": digest(preflight),
                      "policy": "reuse only after matching manifest and current file fingerprints"},
    }
    root = scheduler.choose_root(args.output_parent, args.run_dir, args.resume)
    print(f"Study root: {root}", flush=True)
    with scheduler.ControllerLock(REPO / "outputs", ".xview2-tier3-sweep.lock"), scheduler.ControllerLock(root):
        if args.resume:
            if read_json(root / "config.json") != config:
                raise ValueError("Study configuration/data/code/reference changed; refusing resume")
            state = read_json(root / "state.json")
            if state["config_hash"] != digest(config):
                raise ValueError("Controller state configuration mismatch")
            scheduler.reject_live_children(state)
        else:
            state = {"status": "prepared", "config_hash": digest(config),
                     "trials": {job["id"]: {**job, "status": "pending", "attempts": []} for job in initial_jobs()},
                     "evaluations": {}, "controller_attempts": []}
            for name, value in (("config", config), ("manifest", manifest), ("preflight", preflight),
                                ("class_weights", weights), ("reference", reference), ("panels", panels)):
                durable_json(root / f"{name}.json", value, immutable=True)
        state.update({"memory": memory, "cpu_quota_cores": quota})
        if args.prepare_only and state["status"] == "complete":
            print("Study already complete; preserving its final selection and report.", flush=True)
            return
        started = time.monotonic()
        attempt = {"started_at": time.time(), "resume": args.resume}
        state["controller_attempts"].append(attempt)
        selection = tests = None
        def command_builder(job, uuid, evaluation=False):
            current_quota = scheduler.cpu_quota()
            if current_quota is None or args.max_concurrent * args.cpu_threads > current_quota:
                raise RuntimeError("Cgroup CPU quota changed below the reserved thread allocation")
            state["cpu_quota_cores"] = current_quota
            builder = test_command if evaluation else train_command
            return builder(root, args.xview2_root, job, uuid, args.num_workers)
        try:
            if args.prepare_only:
                return
            state["status"] = "training"
            archived = archived_candidate(reference)
            execute_jobs(
                root, state, "trials", initial_jobs(), gpu_map, args.max_concurrent,
                lambda job: validate_job(root, job, manifest["manifest_hash"]),
                command_builder,
                scheduler, memory_policy=policy, cpu_threads=args.cpu_threads,
                discover=lambda current: discover_promotions(root, current, archived, manifest["manifest_hash"]),
            )
            selection = final_choices(root, archived, manifest["manifest_hash"])
            selection["config_hash"] = digest(config)
            durable_json(root / "final_selection.json", selection, immutable=True)
            jobs = [{"id": checksum, **entry} for checksum, entry in selection["test_identities"].items()]
            state["status"] = "testing_frozen_choices"
            execute_jobs(
                root, state, "evaluations", jobs, gpu_map, args.max_concurrent,
                lambda job: validate_test(root, job, manifest["manifest_hash"]),
                lambda job, uuid: command_builder(job, uuid, evaluation=True),
                scheduler, memory_policy=policy, cpu_threads=args.cpu_threads,
            )
            tests = {job["id"]: validate_test(root, job, manifest["manifest_hash"]) for job in jobs}
            state["status"] = "complete"
        finally:
            error = sys.exc_info()[1]
            if error:
                state["status"] = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
                attempt["error"] = f"{type(error).__name__}: {error}"
            attempt.update({"seconds": time.monotonic() - started, "finished_at": time.time(),
                            "status": state["status"]})
            durable_json(root / "state.json", state)
            publish_tables(root, selection, tests)
            durable_json(root / "summary.json", {
                "status": state["status"], "config_hash": digest(config), "trials": state["trials"],
                "evaluations": state["evaluations"], "selection": selection, "test": tests,
                "reference": reference, "limitations": [
                    "Single seed; original validation/test do not measure new tier3-disaster generalization.",
                    "Reused test benchmark, not an untouched test set or statistical significance study.",
                    "Native CUDA kernels are not bitwise deterministic; RNG/data-order and optimizer state are preserved.",
                ],
                "panel_manifest": str(root / "panels.json"),
            })
    print(f"Complete: {root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Controller received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    main()
