"""Offline split, target, weighting and training-interface regressions."""

import importlib
import csv
from pathlib import Path
import random
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import numpy as np
from PIL import Image
import torch

from bda import xview2

training = importlib.import_module("scripts.train_eval_xview2_dinov3_upernet")


class DataTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("outputs") / (".xview2-tests-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        for source, count in (("train", 20), ("tier3", 7), ("test", 3), ("hold", 2)):
            for index in range(count):
                self.sample(source, index)

    def sample(self, source, index, mask=None, mode="RGB"):
        image = self.root / source / "images" / f"{source}-disaster_{index:04d}_post_disaster.png"
        target = Path(xview2.target_for(image))
        image.parent.mkdir(parents=True, exist_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new(mode, (8, 8)).save(image)
        if mask is None:
            mask = (np.arange(64).reshape(8, 8) % 5).astype(np.uint8)
        Image.fromarray(mask).save(target)
        return image

    def test_original_membership_and_tier3_append(self):
        original = xview2.build_manifest(self.root)
        expanded = xview2.build_manifest(self.root, include_tier3=True)
        expected = xview2.post_images(self.root.resolve(), "train")
        random.Random(0).shuffle(expected)
        paths = xview2.split_paths(self.root, expanded)
        self.assertEqual(paths["val"], expected[:2])
        self.assertEqual(paths["train"][:18], expected[2:])
        self.assertEqual(original["splits"]["val"], expanded["splits"]["val"])
        self.assertEqual(original["splits"]["test"], expanded["splits"]["test"])
        self.assertEqual(expanded["counts"]["train"]["by_source"], {"train": 18, "tier3": 7})
        self.assertEqual(len(paths["train"]), 25)
        self.assertTrue(all("/hold/" not in p for files in paths.values() for p in files))
        self.assertEqual(expanded, xview2.build_manifest(self.root, include_tier3=True))
        with self.assertRaisesRegex(ValueError, "split sizes"):
            xview2.assert_sweep_counts(expanded)

    def test_manifest_tampering_and_dataset_change(self):
        manifest = xview2.build_manifest(self.root)
        xview2.check_manifest(manifest, xview2.build_manifest(self.root))
        changed = dict(manifest, seed=1)
        with self.assertRaisesRegex(ValueError, "hash"):
            xview2.check_manifest(changed, manifest)
        self.sample("train", 0, mask=np.ones((8, 8), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            xview2.check_manifest(manifest, xview2.build_manifest(self.root))

    def test_overlap_and_missing_targets_fail(self):
        source = next((self.root / "train/images").glob("*.png"))
        duplicate = self.root / "test/images" / source.name
        shutil.copy(source, duplicate)
        with self.assertRaisesRegex(ValueError, "overlap"):
            xview2.build_manifest(self.root)
        duplicate.unlink()
        Path(xview2.target_for(source)).unlink()
        with self.assertRaises(FileNotFoundError):
            xview2.build_manifest(self.root)

    def test_hold_overlap_fails_without_scoring_hold(self):
        source = next((self.root / "train/images").glob("*.png"))
        shutil.copy(source, self.root / "hold/images" / source.name)
        with self.assertRaisesRegex(ValueError, "hold"):
            xview2.build_manifest(self.root)

    def test_preflight_checks_every_mask_and_image(self):
        manifest = xview2.build_manifest(self.root, include_tier3=True)
        report = xview2.validate_manifest(self.root, manifest, expected_size=(8, 8))
        self.assertEqual(report["images_validated"], 30)
        self.assertEqual(report["status"], "validated")
        self.assertEqual(sum(map(sum, report["pixel_counts_by_split"].values())), 30 * 64)
        self.sample("tier3", 6, mask=np.full((8, 8), 255, dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "codes"):
            xview2.validate_manifest(self.root, xview2.build_manifest(self.root, True), (8, 8))

    def test_rejects_palette_rgb_targets_and_wrong_dimensions(self):
        image = self.sample("train", 0)
        target = Path(xview2.target_for(image))
        for mode, size in (("RGB", (8, 8)), ("P", (8, 8)), ("L", (7, 8))):
            with self.subTest(mode=mode, size=size):
                Image.new(mode, size).save(target)
                with self.assertRaisesRegex(ValueError, "Invalid target"):
                    xview2.load_mask(target, (8, 8))
        Image.new("L", (8, 8)).save(image)
        with self.assertRaisesRegex(ValueError, "Invalid image"):
            xview2.load_image(image, (8, 8))

    def test_all_groupings_preserve_codes(self):
        for grouping, expected in xview2.GROUPINGS.items():
            with self.subTest(grouping=grouping):
                image = self.sample("train", 0)
                weights, counts = xview2.class_weight_estimate(
                    [str(image)], grouping, expected_size=(8, 8),
                )
                raw = np.arange(64).reshape(8, 8) % 5
                wanted = np.bincount(np.asarray(expected)[raw].ravel(), minlength=3)
                np.testing.assert_array_equal(counts, wanted)
                estimate = 1 / np.sqrt(wanted / wanted.sum() + 1e-6)
                np.testing.assert_allclose(weights, estimate / estimate.mean())

    def test_seeded_300_mask_estimator(self):
        filenames = [f"train/images/x_{i}_post_disaster.png" for i in range(350)]
        used = []
        def fake_mask(path, size):
            used.append(path)
            index = int(Path(path).name.split("_")[1])
            return np.array([[index % 5]], dtype=np.uint8)
        with patch.object(xview2, "load_mask", side_effect=fake_mask):
            first = xview2.class_weight_estimate(filenames, "any")
        self.assertEqual(used, [xview2.target_for(f) for f in random.Random(0).sample(filenames, 300)])
        with patch.object(xview2, "load_mask", side_effect=fake_mask):
            second = xview2.class_weight_estimate(filenames, "any")
        np.testing.assert_array_equal(first[0], second[0])

    def test_split_stats_uses_shared_expanded_membership(self):
        stats = importlib.import_module("scripts.xview2_split_stats")
        for image in self.root.glob("*/images/*.png"):
            label = image.parent.parent / "labels" / (image.stem + ".json")
            xview2.write_json(label, {"features": {"xy": [
                {"properties": {"feature_type": "building"}},
            ]}})
        output = self.root / "sizes.csv"
        manifest = self.root / "manifest.json"
        args = ["stats", "--xview2-root", str(self.root), "--include-tier3",
                "--output", str(output), "--manifest-output", str(manifest)]
        with patch.object(stats.sys, "argv", args), patch("builtins.print"):
            stats.main()
        with output.open() as stream:
            total = list(csv.DictReader(stream))[-1]
        self.assertEqual(total["train_patches"], "25")
        self.assertEqual(total["val_patches"], "2")
        self.assertEqual(total["test_patches"], "3")
        self.assertEqual(total["total_footprints"], "30")
        self.assertEqual(xview2.read_json(manifest), xview2.build_manifest(self.root, True))


class TrainingTests(unittest.TestCase):
    def test_uniform_crop_matches_original_first_attempt(self):
        dataset = training.XView2SegDataset(["unused"], "any", crop_size=4, seed=0)
        image = np.arange(8 * 8 * 3).reshape(8, 8, 3)
        mask = np.zeros((8, 8))
        rng = random.Random(0)
        for _ in range(10):
            y, x = rng.randint(0, 4), rng.randint(0, 4)
            actual, _ = dataset._random_crop(image, mask)
            np.testing.assert_array_equal(actual, image[y:y + 4, x:x + 4])

    def test_default_original_only_and_separate_scoring_flags(self):
        args = training.parser().parse_args(["--grouping", "any", "--output-dir", "unused"])
        self.assertFalse(args.include_tier3)
        self.assertEqual(args.evaluation_split, "both")
        args = training.parser().parse_args([
            "--grouping", "major", "--output-dir", "unused", "--include-tier3",
            "--evaluation-split", "val", "--resume",
        ])
        self.assertTrue(args.include_tier3)
        self.assertTrue(args.resume)
        self.assertEqual(args.evaluation_split, "val")

    def test_evaluation_is_full_image_and_mapping_is_strict(self):
        dataset = training.XView2SegDataset(["image"], "major", crop_size=512,
                                           crops_per_image=4, train=False)
        self.assertEqual(len(dataset), 1)
        raw = np.array([[0, 1, 2, 3, 4]], dtype=np.uint8)
        with patch.object(training, "load_image", return_value=np.zeros((1, 5, 3))), \
                patch.object(training, "load_mask", return_value=raw):
            batch = dataset[0]
        self.assertEqual(tuple(batch["image"].shape), (3, 1, 5))
        self.assertEqual(batch["mask"].tolist(), [[0, 1, 1, 2, 2]])

    def test_metrics_formula_and_zero_f1(self):
        metrics = training.confusion_metrics(np.array([[10, 0, 0], [0, 5, 1], [0, 2, 3]]))
        self.assertAlmostEqual(metrics["damaged_f1"], 6 / 9)
        self.assertAlmostEqual(metrics["per_class"]["damaged"]["iou"], 3 / 6)
        zero = training.confusion_metrics(np.array([[10, 0, 0], [0, 5, 1], [0, 2, 0]]))
        self.assertEqual(zero["damaged_f1"], 0)
        no_predicted_damage = training.confusion_metrics(
            np.array([[10, 0, 0], [0, 5, 0], [0, 2, 0]])
        )
        self.assertEqual(no_predicted_damage["damaged_f1"], 0)
        self.assertEqual(no_predicted_damage["per_class"]["damaged"]["support"], 2)
        self.assertIsNone(no_predicted_damage["per_class"]["damaged"]["precision"])
        empty = training.confusion_metrics(np.zeros((3, 3), dtype=np.int64))
        self.assertIsNone(empty["damaged_f1"])
        self.assertIsNone(empty["overall_accuracy"])

    def test_compute_weights_wrapper_preserves_float32(self):
        with patch.object(training, "class_weight_estimate", return_value=(
            np.array([0.1, 1, 1.9], dtype=np.float32), np.array([1, 2, 3]),
        )):
            weights, counts = training.compute_class_weights(["unused"], "any")
        self.assertEqual(weights.dtype, torch.float32)
        self.assertEqual(counts.tolist(), [1, 2, 3])

    def test_checkpoint_includes_pending_scheduler_update_without_mutating_training(self):
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0)
        scheduler.step(0.1)
        history = training.EpochHistory(Path("outputs") / (".unused-history-" + uuid.uuid4().hex))
        history.rows = [{"epoch": 1, "val_loss": 0.2}]
        trainer = SimpleNamespace(
            validating=True, current_epoch=1, optimizers=[optimizer],
            lr_scheduler_configs=[SimpleNamespace(scheduler=scheduler)],
        )
        checkpoint = {"lr_schedulers": [scheduler.state_dict()],
                      "optimizer_states": [optimizer.state_dict()]}
        history.on_save_checkpoint(trainer, None, checkpoint)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1)
        self.assertEqual(scheduler.last_epoch, 1)
        self.assertAlmostEqual(checkpoint["optimizer_states"][0]["param_groups"][0]["lr"], 0.01)
        self.assertEqual(checkpoint["lr_schedulers"][0]["last_epoch"], 2)
        self.assertTrue(checkpoint["xview2_pending_plateau_step_included"])
        trainer.validating = False
        final_checkpoint = {"lr_schedulers": [scheduler.state_dict()],
                            "optimizer_states": [optimizer.state_dict()]}
        history.on_save_checkpoint(trainer, None, final_checkpoint)
        self.assertEqual(final_checkpoint["lr_schedulers"][0]["last_epoch"], 1)
        self.assertNotIn("xview2_pending_plateau_step_included", final_checkpoint)


if __name__ == "__main__":
    unittest.main()
