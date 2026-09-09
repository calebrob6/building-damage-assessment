"""Tests for inference benchmark output comparisons."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

from scripts.benchmark_inference_dinov3 import compare_outputs, save_results


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write_raster(self, filename, value, nodata=255):
        path = self.root / filename
        with rasterio.open(
            path, "w", driver="GTiff", height=17, width=23, count=1,
            dtype="uint8", crs="EPSG:4326", transform=from_origin(1, 2, 0.1, 0.1),
            nodata=nodata,
        ) as dst:
            dst.write(np.full((17, 23), value, dtype=np.uint8), 1)
        return path

    def test_pixel_comparison(self):
        reference = self.write_raster("reference.tif", 1)
        identical = self.write_raster("same.tif", 1)
        different = self.write_raster("different.tif", 2)
        self.assertEqual(compare_outputs(identical, reference), 0)
        self.assertEqual(compare_outputs(different, reference), 17 * 23)

    def test_metadata_mismatch_is_an_error(self):
        reference = self.write_raster("reference.tif", 1)
        different = self.write_raster("different.tif", 1, nodata=0)
        with self.assertRaisesRegex(ValueError, "nodata"):
            compare_outputs(different, reference)

    def test_results_replace_atomically(self):
        path = self.root / "results.json"
        save_results(path, [{"wall_seconds": 1.5}])
        save_results(path, [{"wall_seconds": 2.5}])
        self.assertEqual(json.loads(path.read_text()), [{"wall_seconds": 2.5}])
        self.assertFalse(path.with_suffix(".tmp").exists())


if __name__ == "__main__":
    unittest.main()
