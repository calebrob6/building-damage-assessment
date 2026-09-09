"""Numerical and spatial invariants for the any-only loss study."""

import unittest

import torch
import torch.nn.functional as F

from bda.boundaries import boundary_band, inner_boundary
from bda.boundary_metrics import SegmentationGeometryMetrics
from bda.losses import LOSS_NAMES, build_loss


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.target = torch.zeros(1, 24, 24, dtype=torch.long)
        self.target[:, 6:16, 6:16] = 1
        self.target[:, 19:22, 19:22] = 2

    def test_band_weights_both_sides_of_class_one_only(self):
        band = boundary_band(self.target, radius=2)
        self.assertTrue(band[0, 6, 10])
        self.assertTrue(band[0, 5, 10])
        self.assertFalse(band[0, 10, 10])
        self.assertFalse(band[0, 0, 0])
        self.assertFalse(band[0, 19, 20])
        self.assertFalse(boundary_band(torch.full_like(self.target, 2)).any())

    def test_no_artificial_crop_or_ignored_boundaries(self):
        full = torch.ones_like(self.target)
        self.assertFalse(boundary_band(full).any())
        full[:, 10:12, 10:12] = 255
        self.assertFalse(boundary_band(full).any())
        touching = torch.zeros_like(self.target, dtype=torch.bool)
        touching[:, 3:20, :16] = True
        edges = inner_boundary(touching, torch.ones_like(touching))
        self.assertFalse(edges[0, 10, 0])
        self.assertTrue(edges[0, 10, 15])

    def test_perfect_prediction_and_empty_cases(self):
        metric = SegmentationGeometryMetrics()
        metric.update(self.target, self.target)
        scores = metric.to_dict()
        for name in ("undamaged_boundary_f1", "undamaged_iou", "undamaged_area_ratio",
                     "building_union_iou", "damaged_f1"):
            self.assertEqual(scores[name], 1)
        metric.reset()
        empty = torch.zeros_like(self.target)
        metric.update(empty, empty)
        self.assertIsNone(metric.to_dict()["undamaged_boundary_f1"])
        metric.reset()
        metric.update(self.target, empty)
        self.assertEqual(metric.to_dict()["undamaged_boundary_f1"], 0)
        self.assertEqual(metric.to_dict()["undamaged_iou"], 0)

    def test_tolerance_and_dilation_are_measured(self):
        shifted = torch.roll(self.target, 2, dims=2)
        tolerant = SegmentationGeometryMetrics(tolerance=2)
        strict = SegmentationGeometryMetrics(tolerance=0)
        tolerant.update(shifted, self.target)
        strict.update(shifted, self.target)
        self.assertEqual(tolerant.to_dict()["undamaged_boundary_f1"], 1)
        self.assertLess(strict.to_dict()["undamaged_boundary_f1"], 1)
        expanded = torch.zeros_like(self.target)
        expanded[:, 2:20, 2:20] = 1
        expanded[self.target == 2] = 2
        tolerant.reset()
        tolerant.update(expanded, self.target)
        self.assertLess(tolerant.to_dict()["undamaged_boundary_f1"], 1)
        self.assertGreater(tolerant.to_dict()["undamaged_area_ratio"], 1)

    def test_empty_image_false_positives_are_not_dropped(self):
        metric = SegmentationGeometryMetrics()
        metric.update(self.target, self.target)
        metric.update(self.target, torch.zeros_like(self.target))
        scores = metric.to_dict()
        self.assertEqual(scores["undamaged_boundary_precision"], 0.5)
        self.assertEqual(scores["undamaged_boundary_recall"], 1)
        self.assertAlmostEqual(scores["undamaged_boundary_f1"], 2 / 3)

    def test_merging_separate_buildings_loses_boundary_and_overlap_quality(self):
        target = torch.zeros(1, 40, 56, dtype=torch.long)
        target[:, 10:30, 6:22] = 1
        target[:, 10:30, 32:48] = 1
        prediction = target.clone()
        prediction[:, 10:30, 22:32] = 1
        metric = SegmentationGeometryMetrics()
        metric.update(prediction, target)
        scores = metric.to_dict()
        self.assertLess(scores["undamaged_boundary_f1"], 1)
        self.assertLess(scores["undamaged_iou"], 1)
        self.assertGreater(scores["undamaged_area_ratio"], 1)

    def test_damaged_outlines_are_not_the_boundary_objective(self):
        prediction = self.target.clone()
        prediction[prediction == 2] = 0
        metric = SegmentationGeometryMetrics()
        metric.update(prediction, self.target)
        scores = metric.to_dict()
        self.assertEqual(scores["undamaged_boundary_f1"], 1)
        self.assertEqual(scores["damaged_f1"], 0)
        self.assertLess(scores["building_union_iou"], 1)

    def test_batch_aggregation_and_ignored_pixels(self):
        target = self.target.repeat(2, 1, 1)
        target[1, :5, :5] = 255
        prediction = target.clone()
        together, separate = SegmentationGeometryMetrics(), SegmentationGeometryMetrics()
        together.update(prediction, target)
        for index in range(2):
            separate.update(prediction[index:index + 1], target[index:index + 1])
        torch.testing.assert_close(together.confusion, separate.confusion)
        self.assertEqual(together.to_dict(), separate.to_dict())
        self.assertEqual(int(together.confusion.sum()), 2 * 24 * 24 - 25)

    def test_invalid_metric_inputs_fail(self):
        with self.assertRaises(ValueError):
            SegmentationGeometryMetrics(class_index=2)
        with self.assertRaises(ValueError):
            SegmentationGeometryMetrics(ignore_index=0)
        with self.assertRaises(ValueError):
            SegmentationGeometryMetrics().update(torch.full_like(self.target, 4), self.target)


class LossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.weights = torch.tensor([0.2, 1.0, 1.8])
        self.logits = torch.randn(2, 3, 12, 12, requires_grad=True)
        self.target = torch.randint(0, 3, (2, 12, 12))
        self.target[:, 0, 0] = 255

    def criterion(self, name, **options):
        return build_loss(name, class_weights=self.weights, ignore_index=255, options=options)

    def test_legacy_ce_is_unchanged(self):
        expected = F.cross_entropy(self.logits, self.target, weight=self.weights, ignore_index=255)
        actual = self.criterion("ce")(self.logits, self.target)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(list(self.criterion("ce").state_dict()), ["weight"])

    def test_disabled_terms_and_gamma_zero_recover_ce(self):
        expected = self.criterion("ce")(self.logits, self.target)
        cases = [
            ("ce_dice", {"region_weight": 0}),
            ("boundary_ce", {"boundary_multiplier": 1}),
            ("boundary_ce", {"boundary_radius": 0}),
            ("boundary_ce_dice", {"boundary_multiplier": 1, "region_weight": 0}),
            ("focal_dice", {"focal_gamma": 0, "region_weight": 0}),
            ("ce_tversky", {"region_weight": 0}),
        ]
        for name, options in cases:
            with self.subTest(name=name, options=options):
                torch.testing.assert_close(self.criterion(name, **options)(self.logits, self.target),
                                           expected, rtol=1e-6, atol=1e-6)

    def test_all_recipes_have_finite_gradients_and_components(self):
        for name in LOSS_NAMES:
            with self.subTest(name=name):
                logits = self.logits.detach().clone().requires_grad_()
                criterion = self.criterion(name)
                value = criterion(logits, self.target)
                value.backward()
                self.assertTrue(value.isfinite())
                self.assertTrue(logits.grad.isfinite().all())
                if name not in ("ce", "dice"):
                    self.assertTrue(criterion.last_components)
                    self.assertTrue(all(not v.requires_grad for v in criterion.last_components.values()))

    def test_background_ignored_and_extreme_inputs(self):
        for name in LOSS_NAMES:
            if name in ("ce", "dice"):
                continue
            for fill in (0, 255):
                with self.subTest(name=name, fill=fill):
                    logits = torch.zeros(2, 3, 16, 16, dtype=torch.float16)
                    logits[:, 0] = 1000
                    logits[:, 1:] = -1000
                    logits.requires_grad_()
                    target = torch.full((2, 16, 16), fill, dtype=torch.long)
                    value = self.criterion(name)(logits, target)
                    value.backward()
                    self.assertTrue(value.isfinite())
                    self.assertTrue(logits.grad.isfinite().all())
                    self.assertEqual(float(value.detach()), 0)

    def test_boundary_error_has_extra_cost(self):
        target = torch.zeros(1, 16, 16, dtype=torch.long)
        target[:, 5:11, 5:11] = 1
        perfect = F.one_hot(target, 3).permute(0, 3, 1, 2).float() * 10 - 5
        edge, far = perfect.clone(), perfect.clone()
        edge[0, 0, 4, 8], edge[0, 1, 4, 8] = -5, 5
        far[0, 0, 0, 0], far[0, 1, 0, 0] = -5, 5
        ce = self.criterion("ce")
        boundary = self.criterion("boundary_ce", boundary_radius=1)
        torch.testing.assert_close(ce(edge, target), ce(far, target))
        self.assertGreater(float(boundary(edge, target)), float(boundary(far, target)))
        damage_only = target * 2
        torch.testing.assert_close(boundary(edge, damage_only), ce(edge, damage_only),
                                   rtol=1e-6, atol=1e-6)

    def test_focal_uses_unweighted_probability_before_class_weights(self):
        logits = torch.tensor([[[[1.0]], [[-1.0]], [[0.5]]]], requires_grad=True)
        target = torch.tensor([[[2]]])
        nll = -logits.log_softmax(1)[0, 2, 0, 0]
        expected = nll * (1 - (-nll).exp()) ** 2
        actual = self.criterion("focal_dice", region_weight=0)(logits, target)
        torch.testing.assert_close(actual, expected)

    def test_invalid_options_and_targets_fail(self):
        for name, options in (
            ("unknown", {}), ("ce", {"region_weight": 1}),
            ("boundary_ce", {"boundary_radius": -1}),
            ("boundary_ce", {"boundary_class": 3}),
            ("ce_dice", {"foreground_classes": [1, 1]}),
            ("focal_dice", {"focal_gamma": float("nan")}),
            ("ce_tversky", {"tversky_alpha": 0.8, "tversky_beta": 0.8}),
        ):
            with self.subTest(name=name, options=options), self.assertRaises(ValueError):
                self.criterion(name, **options)
        with self.assertRaises(ValueError):
            self.criterion("ce_dice")(self.logits, torch.full_like(self.target, 3))
        with self.assertRaises(ValueError):
            self.criterion("ce_dice")(self.logits * float("nan"), self.target)

    def test_double_precision_gradcheck(self):
        logits = torch.randn(1, 3, 3, 3, dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[[0, 1, 2], [1, 1, 2], [0, 0, 2]]])
        for name in ("ce_dice", "boundary_ce", "focal_dice", "ce_tversky"):
            with self.subTest(name=name):
                self.assertTrue(torch.autograd.gradcheck(
                    lambda x: self.criterion(name)(x, target), logits, atol=1e-5, rtol=1e-4,
                ))


if __name__ == "__main__":
    unittest.main()
