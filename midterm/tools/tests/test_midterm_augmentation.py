"""Behavioral checks for masked labels, soft memberships, and weighted fitting."""

import _bootstrap  # noqa: F401; add repository src for standalone discovery

import unittest

import numpy as np

from midterm_augmentation import (
    estimate_cluster_reason_rates, fit_reason_classifiers, membership_matrix,
    observed_label_mask, predict_reason_probabilities, recover_labels,
)


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([[1, 0, 0, 1, 0], [0, 1, 0, 1, 0], [-100]*5])

    def test_masked_rows_do_not_enter_rates_or_prior(self):
        mapping = estimate_cluster_reason_rates(np.eye(3), self.labels, 2)
        np.testing.assert_allclose(mapping["global_prior"], [.5, .5, 0, 1, 0])
        np.testing.assert_allclose(mapping["observed_membership_mass"], [1, 1, 0])
        np.testing.assert_allclose(mapping["rates"][0], [2/3, 1/3, 0, 1, 0])
        np.testing.assert_allclose(mapping["rates"][2], mapping["global_prior"])

    def test_soft_assignments_use_fractional_seed_mass(self):
        memberships = np.array([[.8, .2], [.3, .7], [.5, .5]])
        mapping = estimate_cluster_reason_rates(memberships, self.labels, 2)
        np.testing.assert_allclose(mapping["rates"][0], [1.8/3.1, 1.3/3.1, 0, 1, 0])
        recovered = recover_labels(self.labels, memberships, mapping)
        np.testing.assert_allclose(recovered["scores"][2], .5*mapping["rates"].sum(axis=0))

    def test_preserve_gold_and_recover_all_masked_labels(self):
        memberships = np.eye(3)
        mapping = estimate_cluster_reason_rates(memberships, self.labels, 2)
        recovered = recover_labels(self.labels, memberships, mapping)
        np.testing.assert_array_equal(recovered["labels"][:2], self.labels[:2])
        # Each reason uses >= 0.5 independently, including ties and multiple positives.
        np.testing.assert_array_equal(recovered["labels"][2], [1, 1, 0, 1, 0])
        np.testing.assert_allclose(recovered["sample_weight"], [1, 1, .1])

    def test_empty_label_sets_are_retained(self):
        mapping = {"rates": np.full((1, 5), .2)}
        recovered = recover_labels(self.labels, np.ones((3, 1)), mapping)
        np.testing.assert_array_equal(recovered["labels"][2], np.zeros(5))
        np.testing.assert_allclose(recovered["decision_margins"][2], .6)

    def test_partial_or_invalid_masks_are_rejected(self):
        for row in [[-100, 0, 0, 0, 0], [2, 0, 0, 0, 0], [np.nan]*5]:
            with self.assertRaises(ValueError):
                observed_label_mask(np.vstack([self.labels[:2], row]))

    def test_invalid_memberships_are_rejected(self):
        for memberships in [[[.3, .3]], [[1.1, -.1]], [[np.nan, 1]]]:
            with self.assertRaises(ValueError):
                membership_matrix(memberships)

    def test_weights_change_fit_and_masked_targets_are_rejected(self):
        features = np.array([[-2], [-1], [1], [2], [1.5]], dtype=float)
        y = np.tile(np.array([0, 0, 1, 1, 0])[:, None], (1, 5))
        full, _ = fit_reason_classifiers(features, y, np.ones(5))
        downweighted, _ = fit_reason_classifiers(features, y, [1, 1, 1, 1, .01])
        self.assertGreater(predict_reason_probabilities(downweighted, [[1.5]])[0, 0],
                           predict_reason_probabilities(full, [[1.5]])[0, 0])
        with self.assertRaises(ValueError):
            fit_reason_classifiers(features[:3], self.labels, np.ones(3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
