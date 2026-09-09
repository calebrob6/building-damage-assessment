"""Offline regression coverage for DINOv3 raster inference."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
import torch
import torch.nn.functional as F

import inference_dinov3 as inference


class PixelClassifier(torch.nn.Module):
    def forward(self, images):
        # Recover labels encoded in red; this also catches clipped normalization.
        labels = (images[:, 0] * inference.IMAGENET_STD[0] + inference.IMAGENET_MEAN[0])
        return F.one_hot(labels.round().long(), num_classes=3).permute(0, 3, 1, 2).float()


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.input = self.root / "input.tif"
        self.output = self.root / "output.tif"
        self.checkpoint = self.root / "model.ckpt"

    def write_input(self, height=27, width=37, masked=False, dtype="uint8"):
        labels = (np.indices((height, width)).sum(axis=0) % 3).astype(np.uint8)
        image = np.stack([labels, np.full_like(labels, 127), np.full_like(labels, 127)])
        self.transform = from_origin(500000, 4500000, 0.5, 0.5)
        with rasterio.open(
            self.input, "w", driver="GTiff", count=3, dtype=dtype,
            height=height, width=width, crs="EPSG:32618", transform=self.transform,
        ) as dst:
            dst.write(image.astype(dtype))
            dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
            if masked:
                valid = np.full_like(labels, 255)
                valid[0, 0] = 0
                valid[-1, -1] = 0
                valid[height // 2, width // 2] = 0
                dst.write_mask(valid)
                labels[valid == 0] = 255
        return labels

    def run_model(self, **kwargs):
        with patch.object(inference, "load_model", return_value=PixelClassifier()):
            inference.run_inference(
                self.input, self.checkpoint, self.output, device="cpu",
                patch_size=32, batch_size=4, **kwargs,
            )

    def test_full_coverage_normalization_and_nodata(self):
        expected = self.write_input(masked=True)
        self.run_model(padding=8)
        with rasterio.open(self.output) as src:
            np.testing.assert_array_equal(src.read(1), expected)
            self.assertEqual(src.crs, rasterio.crs.CRS.from_epsg(32618))
            self.assertEqual(src.transform, self.transform)
            self.assertEqual(src.nodata, 255)
            self.assertEqual(src.dtypes, ("uint8",))
            self.assertEqual(src.count, 1)
            self.assertEqual(src.colormap(1)[0], (0, 0, 0, 255))
            self.assertEqual(src.tags()["CLASS_2"], "damaged")
        self.assertEqual(list(self.root.glob(".dinov3-*")), [])

    def test_zero_padding_and_small_images(self):
        for shape in ((27, 37), (1, 1), (1, 37)):
            with self.subTest(shape=shape):
                expected = self.write_input(*shape)
                self.run_model(padding=0, overwrite=True)
                with rasterio.open(self.output) as src:
                    np.testing.assert_array_equal(src.read(1), expected)

    def test_prefetched_batches_match_serial(self):
        for shape in ((47, 73), (1, 1), (1, 37)):
            for workers in (0, 1, 3):
                with self.subTest(shape=shape, workers=workers):
                    expected = self.write_input(*shape, masked=True)
                    self.run_model(padding=8, num_workers=workers, overwrite=True)
                    with rasterio.open(self.output) as src:
                        np.testing.assert_array_equal(src.read(1), expected)

    def test_normalized_batches_match_serial(self):
        self.write_input(47, 73, masked=True)
        outputs = []
        for workers in (0, 3):
            coordinates = iter([(0, 0), (16, 32), (32, 64)])
            with inference.patch_batches(
                self.input, coordinates, 2, 32, 8, torch.device("cpu"),
                num_workers=workers, prefetch_factor=1,
            ) as batches:
                outputs.append(list(batches))
        for serial, parallel in zip(*outputs):
            torch.testing.assert_close(serial[0], parallel[0], rtol=0, atol=0)
            np.testing.assert_array_equal(serial[1], parallel[1])
            self.assertEqual(serial[2], parallel[2])

    def test_prefetch_does_not_read_the_whole_raster(self):
        self.write_input()
        reads = []
        original_read = inference.read_patch

        def tracked_read(*args):
            reads.append(1)
            return original_read(*args)

        with patch.object(inference, "read_patch", side_effect=tracked_read):
            with inference.patch_batches(
                self.input, iter([(0, 0)] * 100), 2, 32, 8, torch.device("cpu"),
                num_workers=2, prefetch_factor=1,
            ) as batches:
                next(batches)
        self.assertLessEqual(len(reads), 6)

    def test_read_patch_matches_whole_image_reflection(self):
        self.write_input()
        with rasterio.open(self.input) as src:
            image = src.read()
            padded = np.pad(image, ((0, 0), (8, 13), (8, 19)), mode="reflect")
            for y in range(0, src.height, 16):
                for x in range(0, src.width, 16):
                    actual, valid = inference.read_patch(src, y, x, 32, 8)
                    np.testing.assert_array_equal(actual, padded[:, y:y + 32, x:x + 32])
                    self.assertTrue(valid.all())

    def test_existing_output_is_protected(self):
        self.write_input()
        self.output.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            self.run_model(padding=8)
        self.assertEqual(self.output.read_bytes(), b"existing")

    def test_input_cannot_be_overwritten(self):
        self.write_input()
        self.output = self.input
        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            self.run_model(padding=8, overwrite=True)

    def test_invalid_options_and_imagery(self):
        for options in (
            {"patch_size": 16}, {"patch_size": 33}, {"padding": -1},
            {"padding": 256}, {"batch_size": 0}, {"num_workers": -1},
            {"prefetch_factor": 0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                inference.run_inference(
                    self.input, self.checkpoint, self.output, device="cpu", **options,
                )
        self.write_input(dtype="uint16")
        with self.assertRaisesRegex(ValueError, "8-bit"):
            self.run_model(padding=8)

    def test_failure_leaves_existing_output_intact(self):
        self.write_input()
        self.output.write_bytes(b"existing")
        model = PixelClassifier()
        with patch.object(model, "forward", side_effect=RuntimeError("inference failed")):
            with patch.object(inference, "load_model", return_value=model):
                with self.assertRaisesRegex(RuntimeError, "inference failed"):
                    inference.run_inference(
                        self.input, self.checkpoint, self.output, device="cpu",
                        patch_size=32, padding=8, overwrite=True,
                    )
        self.assertEqual(self.output.read_bytes(), b"existing")
        self.assertEqual(list(self.root.glob(".dinov3-*")), [])

    def test_checkpoint_hyperparameters_are_checked(self):
        torch.save({"hyper_parameters": {"model": "unet"}}, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "UPerNet checkpoint"):
            inference.load_model(self.checkpoint, torch.device("cpu"))

    def test_reader_failure_cleans_up_handles_and_output(self):
        self.write_input()
        self.output.write_bytes(b"existing")
        opened = []
        original_open = rasterio.open

        def tracked_open(*args, **kwargs):
            src = original_open(*args, **kwargs)
            opened.append(src)
            return src

        with patch.object(inference.rasterio, "open", side_effect=tracked_open):
            with patch.object(inference, "read_patch", side_effect=RuntimeError("read failed")):
                with self.assertRaisesRegex(RuntimeError, "read failed"):
                    self.run_model(padding=8, num_workers=2, overwrite=True)
        self.assertTrue(opened)
        self.assertTrue(all(src.closed for src in opened))
        self.assertEqual(self.output.read_bytes(), b"existing")
        self.assertEqual(list(self.root.glob(".dinov3-*")), [])


if __name__ == "__main__":
    unittest.main()
