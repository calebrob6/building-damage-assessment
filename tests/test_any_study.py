"""Offline study guards, frozen evidence and exact staged-state continuation."""

from contextlib import ExitStack
import copy
import importlib
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bda.experiment_runner import durable_json, execute_jobs
from bda.xview2 import digest, read_json

study = importlib.import_module("scripts.study_xview2_any_losses")
train = importlib.import_module("scripts.train_eval_xview2_dinov3_upernet")


def candidate(epoch, boundary=.5, iou=.6, damage=.6, recipe="ce"):
    return {"epoch": epoch, "epochs_completed": epoch + 1, "recipe": recipe,
            "checkpoint": f"{recipe}/{epoch}.ckpt",
            "metrics": {"damaged_f1": damage, "per_class": {"undamaged": {"iou": iou}},
                        "geometry": {"undamaged_boundary_f1": boundary}}}


class StudySelectionTests(unittest.TestCase):
    def test_exact_recipes_horizon_and_no_extra_variants(self):
        self.assertEqual(len(study.LOSSES), 6)
        jobs = study.initial_jobs()
        self.assertEqual(sum(job["budget"] for job in jobs), 210)
        self.assertEqual([job["budget"] for job in jobs], [60, 30, 30, 30, 30, 30])
        self.assertEqual(study.RECIPE["lr"], 3e-5)
        self.assertEqual(study.LOSSES["boundary_ce"]["boundary_class"], 1)
        self.assertEqual(study.LOSSES["ce_tversky"]["tversky_alpha"], .7)

    def test_archived_and_ce_floors_are_both_enforced(self):
        archived, ce = candidate(13, iou=.6, damage=.6), candidate(14, iou=.7, damage=.65)
        bad = candidate(2, boundary=.9, iou=.68, damage=.64)
        good = candidate(4, boundary=.7, iou=.69, damage=.64)
        result = study.select_candidate([bad, good], archived, ce)
        self.assertEqual(result["candidate"], good)
        self.assertEqual(result["floors"], {"undamaged_iou": .69, "damaged_f1": .64})

    def test_ineligible_ce_retains_archived_floors(self):
        archived = candidate(13, boundary=.6, iou=.6, damage=.6)
        catalogs = {name: [candidate(0, boundary=.7, iou=.59, damage=.59, recipe=name)]
                    for name in study.LOSSES}
        catalogs["ce"][0]["metrics"]["damaged_f1"] = .1
        result = study.budget_selection(catalogs, 30, archived)
        self.assertEqual(result["decisions"]["ce"]["status"], "ineligible")
        self.assertEqual(result["decisions"]["ce_dice"]["floors"]["damaged_f1"], .59)
        self.assertEqual(result["decisions"]["ce_dice"]["ce_reference"], "archived_floors_only")

    def test_rank_ties_and_no_eligible_candidate(self):
        reference = candidate(13)
        rows = [candidate(3), candidate(2), candidate(1, iou=.61), candidate(0, damage=.61)]
        self.assertEqual(study.select_candidate(rows, reference)["candidate"]["epoch"], 1)
        self.assertIsNone(study.select_candidate([candidate(0, iou=.1)], reference)["candidate"])
        self.assertEqual(study.select_candidate([candidate(0, boundary=0)], reference)["status"], "eligible")

    def test_undefined_optional_precision_does_not_disqualify_primary_metrics(self):
        reference = candidate(13)
        row = candidate(0, boundary=0)
        row["metrics"]["geometry"].update(
            undamaged_boundary_precision=None, undamaged_boundary_recall=0.0,
            undamaged_precision=None,
        )
        result = study.select_candidate([row], reference)
        self.assertEqual(result["status"], "eligible")
        self.assertEqual(result["candidate"]["metrics"]["geometry"]["undamaged_boundary_f1"], 0)
        row["metrics"]["geometry"]["undamaged_boundary_f1"] = None
        self.assertEqual(study.select_candidate([row], reference)["status"], "ineligible")

    def test_promotion_is_strict_capped_and_uses_frozen_thirty_only(self):
        reference = candidate(13, boundary=.5)
        catalogs = {name: [candidate(29, boundary=.5 + index * .01, recipe=name),
                           candidate(30, boundary=1, recipe=name)]
                    for index, name in enumerate(study.LOSSES)}
        selection = study.budget_selection(catalogs, 30, reference)
        promoted = study.promotions(selection, reference)
        self.assertEqual(promoted["promoted"], ["ce_tversky", "focal_dice"])
        self.assertTrue(all(row["candidate"]["epoch"] == 29 for row in selection["decisions"].values()))
        for rows in catalogs.values():
            rows[0]["metrics"]["geometry"]["undamaged_boundary_f1"] = .5
        self.assertEqual(study.promotions(study.budget_selection(catalogs, 30, reference), reference)["promoted"], [])
        with self.assertRaises(ValueError):
            study.budget_selection({"ce": []}, 30, reference)


class SyntheticImages(train.XView2SegDataset):
    def __init__(self, training):
        super().__init__(list(range(6)), "any", crop_size=4, train=training, seed=0)

    def _load(self, idx):
        pixels = np.arange(8 * 8).reshape(8, 8)
        image = np.stack([(pixels + idx * 21) % 255, pixels + 50, pixels + 100], -1).astype(np.float32)
        return image, ((pixels + idx) % 3).astype(np.int64)

    def __getitem__(self, index):
        epoch, identifier = index if isinstance(index, tuple) else (-1, index)
        return {**super().__getitem__(index), "identifier": identifier, "sample_epoch": epoch}


class TinyTask(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1), torch.nn.Dropout(.25))
        self.records = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    def training_step(self, batch, index):
        augmentation = torch.rand(()).item()
        self.records.append((batch["identifier"].tolist(), batch["sample_epoch"].tolist(),
                             batch["image"].flatten()[:6].tolist(), augmentation))
        return F.cross_entropy(self.model(batch["image"] + augmentation), batch["mask"])

    def on_validation_epoch_start(self):
        self.confusion = torch.zeros(3, 3, dtype=torch.long)

    def validation_step(self, batch, index):
        logits = self.model(batch["image"])
        labels = logits.argmax(1)
        self.confusion += torch.bincount((batch["mask"] * 3 + labels).flatten(), minlength=9).reshape(3, 3)
        self.log("val_loss", F.cross_entropy(logits, batch["mask"]) * 0 + 1.0,
                 batch_size=len(labels))

    def on_validation_epoch_end(self):
        self.last_validation_summary = {"confusion_matrix": self.confusion.tolist(),
                                        "geometry": {"undamaged_boundary_f1": .5}}


class StageTests(unittest.TestCase):
    def setUp(self):
        self.root = (Path("outputs") / (".any-stage-tests-" + uuid.uuid4().hex)).resolve()
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)

    def run_stage(self, name, max_epochs, workers, resume=None):
        pl.seed_everything(0, workers=True)
        directory = self.root / name
        directory.mkdir(exist_ok=True)
        dataset = SyntheticImages(True)
        loader = train.EpochDataLoader(
            dataset, sampler=train.EpochSampler(dataset), batch_size=2, num_workers=workers,
            worker_init_fn=train.seed_worker, generator=torch.Generator().manual_seed(0),
        )
        val = DataLoader(SyntheticImages(False), batch_size=2, num_workers=workers,
                         generator=torch.Generator().manual_seed(100000))
        task = TinyTask()
        config = {"max_epochs": 2, "manifest_hash": "synthetic"}
        artifacts = train.StudyCheckpoints(directory, config, [1, 2], 6)
        trainer = pl.Trainer(
            max_epochs=max_epochs, accelerator="cpu", devices=1, logger=False,
            enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
            enable_checkpointing=False,
            callbacks=[train.EpochHistory(directory / "epochs.json"),
                       train.ReproducibilityState(loader, val), artifacts],
        )
        trainer.fit(task, loader, val, ckpt_path=resume)
        trainer.save_checkpoint(directory / "checkpoints/last.ckpt", weights_only=False)
        checkpoint = torch.load(directory / "checkpoints/last.ckpt", weights_only=True)
        return task, checkpoint

    def test_uninterrupted_and_staged_preserve_order_augmentation_and_training_state(self):
        for workers in (0, 1):
            with self.subTest(workers=workers):
                uninterrupted, expected = self.run_stage(f"full-{workers}", 2, workers)
                first, _ = self.run_stage(f"staged-{workers}", 1, workers)
                marker = read_json(self.root / f"staged-{workers}/stages/epoch_001/ready.json")
                second, actual = self.run_stage(f"staged-{workers}", 2, workers,
                                                marker["continuation_checkpoint"])
                self.assertEqual(uninterrupted.records, first.records + second.records)
                for name, tensor in expected["state_dict"].items():
                    torch.testing.assert_close(actual["state_dict"][name], tensor, rtol=0, atol=0)
                self.assertEqual(actual["global_step"], expected["global_step"])
                self.assertEqual(actual["lr_schedulers"], expected["lr_schedulers"])
                for key, tensor in expected["optimizer_states"][0]["state"].items():
                    for field, value in tensor.items():
                        torch.testing.assert_close(actual["optimizer_states"][0]["state"][key][field],
                                                   value, rtol=0, atol=0)
                self.assertEqual(marker, read_json(self.root / f"staged-{workers}/stages/epoch_001/ready.json"))
                frozen = read_json(marker["catalog"])
                self.assertEqual([row["epochs_completed"] for row in frozen["candidates"]], [1])
                self.assertNotIn("optimizer_states", torch.load(frozen["candidates"][0]["checkpoint"],
                                                               weights_only=True))
                self.assertIn("xview2_reproducibility", actual)

    def test_weights_only_epoch_history_does_not_patch_optimizer(self):
        history = train.EpochHistory(self.root / "epochs.json")
        history.rows = [{"epoch": 0, "val_loss": 1.0}]
        checkpoint = {}
        history.on_save_checkpoint(SimpleNamespace(validating=True, current_epoch=0), None, checkpoint)
        self.assertNotIn("lr_schedulers", checkpoint)

    def test_interrupted_candidate_publication_recovers_without_retraining_epoch(self):
        _, expected = self.run_stage("publication-control", 2, 0)
        with patch.object(train.StudyCheckpoints, "commit_candidate",
                          side_effect=RuntimeError("interrupted candidate publication")):
            with self.assertRaisesRegex(RuntimeError, "candidate publication"):
                self.run_stage("publication-interrupted", 1, 0)
        directory = self.root / "publication-interrupted"
        last = directory / "checkpoints/last.ckpt"
        self.assertTrue(last.is_file())
        self.assertFalse((directory / "candidates.json").exists())
        resumed, actual = self.run_stage("publication-interrupted", 2, 0, str(last))
        self.assertEqual([record[1] for record in resumed.records], [[1, 1]] * 3)
        for key, value in expected["state_dict"].items():
            torch.testing.assert_close(actual["state_dict"][key], value, rtol=0, atol=0)
        self.assertEqual(actual["lr_schedulers"], expected["lr_schedulers"])
        self.assertEqual(actual["global_step"], expected["global_step"])
        self.assertTrue((directory / "stages/epoch_001/ready.json").is_file())
        self.assertEqual(len(read_json(directory / "candidates.json")["candidates"]), 2)

    def test_immutable_artifact_refuses_changes(self):
        path = self.root / "ready.json"
        durable_json(path, {"epoch": 30}, immutable=True)
        durable_json(path, {"epoch": 30}, immutable=True)
        with self.assertRaises(ValueError):
            durable_json(path, {"epoch": 60}, immutable=True)

    def test_command_promotes_only_a_completed_stage(self):
        output = self.root / "recipes/ce_dice"
        durable_json(output / "config.json", {"horizon": 60})
        durable_json(output / "training_state.json", {"stage_epochs": 30, "status": "failed"})
        job = {"id": "ce_dice/stage_060", "recipe": "ce_dice", "budget": 60}
        with self.assertRaisesRegex(ValueError, "failed or incomplete"):
            study.train_command(self.root, "data", job, "GPU-four", 1)
        durable_json(output / "training_state.json", {"stage_epochs": 30, "status": "stage_complete"})
        command, _ = study.train_command(self.root, "data", job, "GPU-four", 1)
        self.assertIn("--resume", command)
        self.assertIn("--promote-stage", command)
        self.assertEqual(command[command.index("--max-epochs") + 1], "60")
        self.assertEqual(command[command.index("--lr") + 1], "3e-05")
        durable_json(output / "training_state.json", {"stage_epochs": 60, "status": "failed"})
        command, _ = study.train_command(self.root, "data", job, "GPU-four", 1)
        self.assertIn("--resume", command)
        self.assertNotIn("--promote-stage", command)

    def test_failed_extension_does_not_erase_completed_thirty_evidence(self):
        output = self.root / "recipes/ce_dice"
        durable_json(output / "training_state.json", {"stage_epochs": 60, "status": "interrupted"})
        durable_json(self.root / "promotion.json", {"promoted": ["ce_dice"]})
        first = {"id": "ce_dice/stage_030", "recipe": "ce_dice", "budget": 30}
        with patch.object(study, "read_stage", return_value=["immutable-30"]) as read:
            self.assertEqual(study.validate_job(self.root, first, "hash"), ["immutable-30"])
        read.assert_called_once_with(self.root, "ce_dice", 30, "hash", verify_checkpoint=True)
        continuation = dict(first, id="ce_dice/stage_060", budget=60)
        with self.assertRaises(study.scheduler.IncompleteTrialError):
            study.validate_job(self.root, continuation, "hash")

    def test_test_jobs_only_score_an_explicit_frozen_checkpoint(self):
        job = {"id": "identity", "checkpoint": "/selected/checkpoint.ckpt"}
        command, output = study.test_command(self.root, "data", job, "GPU-four", 1)
        self.assertIn("--eval-only", command)
        self.assertIn("--geometry-metrics", command)
        self.assertNotIn("--epoch-candidates", command)
        self.assertEqual(command[command.index("--evaluation-split") + 1], "test")
        self.assertEqual(command[command.index("--checkpoint") + 1], job["checkpoint"])
        self.assertEqual(output, self.root / "test/identity")

    def test_stage_reader_uses_frozen_catalog_not_later_live_candidates(self):
        output = self.root / "recipes/ce"
        config = {
            **study.RECIPE, "loss": "ce", "loss_options": {}, "geometry_metrics": True,
            "validation_precision": "fp32", "epoch_candidates": True, "milestones": [15, 30, 60],
        }
        durable_json(output / "config.json", config)
        stage = output / "stages/epoch_030"
        stage.mkdir(parents=True)
        continuation = stage / "last.ckpt"
        continuation.write_bytes(b"continuation")
        rows = []
        for epoch in range(30):
            path = output / "candidates" / f"epoch_{epoch + 1:03d}.ckpt"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"weights")
            rows.append({
                "epoch": epoch, "epochs_completed": epoch + 1, "global_step": (epoch + 1) * 2222,
                "n_val_images": 279, "checkpoint": str(path),
                "metrics": {"confusion_matrix": [[279 * 1024**2, 0, 0], [0, 0, 0], [0, 0, 0]]},
            })
        catalog = {"config_hash": digest(config), "budget": 30, "candidates": rows}
        durable_json(stage / "catalog.json", catalog)
        durable_json(stage / "ready.json", {
            "status": "ready", "epochs_completed": 30, "horizon": 60,
            "manifest_hash": "hash", "config_hash": digest(config),
            "catalog": str(stage / "catalog.json"), "catalog_hash": digest(catalog),
            "continuation_checkpoint": str(continuation),
            "continuation_sha256": study.file_hash(continuation), "global_step": 30 * 2222,
        })
        durable_json(output / "candidates.json", {"candidates": rows + [{"epoch": 59}]})
        self.assertEqual(len(study.read_stage(self.root, "ce", 30, "hash", True)), 30)
        catalog["candidates"].append({"epoch": 30})
        durable_json(stage / "catalog.json", catalog)
        with self.assertRaisesRegex(ValueError, "catalog changed"):
            study.read_stage(self.root, "ce", 30, "hash")

    def test_final_test_identities_are_deduplicated(self):
        path = self.root / "same.ckpt"
        path.write_bytes(b"same")
        reference = candidate(13, boundary=.5, recipe="archived")
        reference.update(checkpoint=str(path), sha256=study.file_hash(path))
        ce = candidate(0, boundary=.7)
        ce.update(checkpoint=str(path), sha256=study.file_hash(path))
        alternative = candidate(0, boundary=.6)
        alternative.update(checkpoint=str(path), sha256=study.file_hash(path))
        durable_json(self.root / "promotion.json", {"promoted": []})
        def stage_rows(root, recipe, budget, manifest_hash):
            return [ce if recipe == "ce" else alternative]
        with patch.object(study, "read_stage", side_effect=stage_rows):
            result = study.final_choices(self.root, reference, "hash")
        self.assertEqual(len(result["test_identities"]), 1)
        roles = next(iter(result["test_identities"].values()))["roles"]
        self.assertEqual(set(roles), {"archived", "final_candidate", "ce_15", "ce_30", "ce_60"})
        self.assertEqual(result["aggregate_epochs"], 210)

    def test_runner_discovers_continuation_while_ce_remains_active(self):
        backend = study.scheduler
        initial = [{"id": "ce", "budget": 60}, {"id": "alternative", "budget": 30}]
        continuation = {"id": "promoted", "budget": 60}
        state = {"trials": {row["id"]: {**row, "status": "pending", "attempts": []} for row in initial}}
        complete = set()
        processes, launch_order = {}, []
        def validate(job):
            if job["id"] not in complete:
                raise FileNotFoundError(job["id"])
        def spawn(command, **kwargs):
            name = command[0]
            launch_order.append(name)
            process = Mock(pid=2000 + len(launch_order), returncode=0)
            def poll():
                if name == "ce" and "promoted" not in launch_order:
                    return None
                complete.add(name)
                return 0
            process.poll.side_effect = poll
            processes[name] = process
            return process
        def discover(current):
            if current["trials"]["alternative"]["status"] == "complete":
                self.assertIn(current["trials"]["ce"]["status"], ("running", "complete"))
                return [continuation]
            return []
        with ExitStack() as stack:
            stack.enter_context(patch.object(backend, "available_gpu", return_value=True))
            stack.enter_context(patch.object(backend, "process_start_ticks", return_value="ticks"))
            stack.enter_context(patch("bda.experiment_runner.subprocess.Popen", side_effect=spawn))
            stack.enter_context(patch("bda.experiment_runner.time.sleep"))
            stack.enter_context(patch("builtins.print"))
            execute_jobs(
                self.root, state, "trials", initial, {4: "GPU-four", 5: "GPU-five"}, 2, validate,
                lambda job, gpu: ([job["id"]], self.root / job["id"]), backend, discover=discover,
            )
        self.assertEqual(launch_order, ["ce", "alternative", "promoted"])
        self.assertTrue(all(record["status"] == "complete" for record in state["trials"].values()))


if __name__ == "__main__":
    unittest.main()
