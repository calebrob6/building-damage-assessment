"""Training-task integration without downloading a backbone."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import lightning
import torch

from bda.trainers import CustomSemanticSegmentationTask


class TinyTask(CustomSemanticSegmentationTask):
    def configure_models(self):
        self.model = torch.nn.Conv2d(self.hparams["in_channels"], self.hparams["num_classes"], 1)


def task(**options):
    return TinyTask(
        model="upernet", backbone="dinov3_vits16", weights=False,
        in_channels=3, num_classes=3, class_weights=torch.tensor([0.2, 1.0, 1.8]),
        ignore_index=255, **options,
    )


class TaskTests(unittest.TestCase):
    def test_geometry_validation_uses_one_float32_forward(self):
        model = task(loss="ce_dice", geometry_metrics=True, validation_precision="fp32")
        model.log = Mock()
        model.log_dict = Mock()
        seen = []
        model.model.register_forward_hook(lambda module, args, output: seen.append(output.dtype))
        batch = {"image": torch.randn(2, 3, 16, 16), "mask": torch.randint(3, (2, 16, 16))}
        model.on_validation_epoch_start()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            model.validation_step(batch, 0)
        model.on_validation_epoch_end()
        self.assertEqual(seen, [torch.float32])
        summary = model.last_validation_summary
        self.assertEqual(sum(map(sum, summary["confusion_matrix"])), 512)
        self.assertIn("undamaged_boundary_f1", summary["geometry"])
        self.assertIn("val_undamaged_boundary_f1", model.log_dict.call_args.args[0])

    def test_legacy_and_new_checkpoint_hyperparameters_load(self):
        for name, options in (
            ("ce", {}),
            ("boundary_ce_dice", {"boundary_radius": 2, "region_weight": 0.5}),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                model = task(loss=name, loss_options=options, geometry_metrics=name != "ce")
                hparams = dict(model.hparams)
                if name == "ce":
                    for key in ("loss_options", "geometry_metrics", "validation_precision"):
                        hparams.pop(key, None)
                filename = Path(directory) / "model.ckpt"
                torch.save({
                    "state_dict": model.state_dict(), "hyper_parameters": hparams,
                    "pytorch-lightning_version": lightning.__version__,
                }, filename)
                loaded = TinyTask.load_from_checkpoint(filename, weights=False, weights_only=True)
                self.assertEqual(loaded.loss_options, options)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)

    def test_new_constraint_combinations_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "constraint"):
            task(loss="boundary_ce", use_constraint_loss=True)
        with self.assertRaisesRegex(ValueError, "constraint"):
            task(loss="ce", geometry_metrics=True, use_constraint_loss=True)
        with self.assertRaises(ValueError):
            task(loss="ce", validation_precision="fp16")


if __name__ == "__main__":
    unittest.main()
