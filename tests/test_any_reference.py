"""Deterministic GT-only panel selection and non-mutating visualization."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from bda.xview2 import target_for
from scripts.prepare_any_reference import render_panel, select_panels


class ReferenceTests(unittest.TestCase):
    def test_panel_selection_is_deterministic_and_gt_only(self):
        root = Path("/synthetic-xview2")
        filenames, masks = [], {}
        for category in ("small", "large", "crowded", "isolated"):
            for index in range(3):
                filename = root / "train/images" / f"{category}_{index}_post_disaster.png"
                mask = np.zeros((1024, 1024), dtype=np.uint8)
                if category == "small":
                    size = index + 2
                    mask[100:100 + size, 100:100 + size] = 1
                elif category == "large":
                    size = 100 + index * 10
                    mask[300:300 + size, 300:300 + size] = 1
                elif category == "crowded":
                    for n in range(20 + index):
                        y, x = 20 * (n // 10), 20 * (n % 10)
                        mask[y:y + 8, x:x + 8] = 1
                else:
                    size = 20 + index
                    mask[600:600 + size, 600:600 + size] = 1
                filenames.append(str(filename))
                masks[target_for(filename)] = mask
        with patch("scripts.prepare_any_reference.load_mask", side_effect=lambda path: masks[path]):
            first = select_panels(root, filenames)
            second = select_panels(root, list(reversed(filenames)))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertEqual(len({row["image"] for row in first}), 12)
        for category in ("small", "large", "crowded", "isolated"):
            self.assertEqual(sum(row["category"] == category for row in first), 3)
        for row in first:
            x, y, width, height = row["window"]
            self.assertTrue(0 <= x <= 512 and 0 <= y <= 512)
            self.assertEqual((width, height), (512, 512))

    def test_render_preserves_inputs_and_uses_expected_dimensions(self):
        image = np.full((32, 32, 3), 100, dtype=np.uint8)
        truth = np.zeros((32, 32), dtype=np.uint8)
        truth[8:16, 8:16] = 1
        prediction = truth.copy()
        original = image.copy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel.png"
            render_panel(image, truth, prediction, [0, 0, 32, 32], "reference", path)
            with Image.open(path) as result:
                self.assertEqual(result.size, (128, 76))
        np.testing.assert_array_equal(image, original)
        np.testing.assert_array_equal(prediction, truth)


if __name__ == "__main__":
    unittest.main()
