"""Migration contracts for fixed-recipe training; all estimator fitting is mocked."""
import _bootstrap  # noqa: F401

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch, sentinel

import numpy as np
import pandas as pd

import train_core as core


def reason_rows(ids, labels):
    frame = pd.DataFrame(labels, columns=core.REASONS)
    frame.insert(0, "ID", ids)
    return pd.concat([frame, pd.DataFrame(np.arange(len(ids))[:, None] +
        np.zeros((len(ids), 150)), columns=core.EMB)], axis=1)


class TrainingEntryTests(unittest.TestCase):
    def test_ridge_preserves_gold_unknown_scope_and_fixed_calibration(self):
        train = reason_rows(["a", "b", "u"], [[1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0], [-100] * 5])
        val = reason_rows(["v"], [[0, 0, 1, 0, 0]])
        with patch.object(core, "fit_calibrated", return_value=sentinel.model) as fit:
            result = core.fit_part1_ridge(train, val)
        self.assertIs(result, sentinel.model)
        scope, config = fit.call_args.args
        self.assertEqual(scope.data.frame.ID.tolist(), ["a", "b", "v", "u"])
        self.assertEqual(scope.gold.tolist(), [0, 1, 2])
        self.assertEqual(scope.eligible.tolist(), [0, 1, 2, 3])
        self.assertTrue(np.all(scope.data.y[3] == -100))
        self.assertEqual(config, dict(geometry="global", family="ridge", bandwidth=.5, strength=.1))
        self.assertEqual(fit.call_args.kwargs, {"seed": 942})
        with self.assertRaisesRegex(ValueError, "unique and disjoint"):
            core.ridge_data(train, val.assign(ID="a"))

    def test_part2_fixed_nested_weights_and_shared_setting(self):
        with patch.object(core, "fit_part2_baseline", return_value=sentinel.word), \
             patch.object(core, "fit_field_outcomes", return_value=(sentinel.field, None)), \
             patch.object(core, "fit_phase_baseline", return_value=(sentinel.char, None, None, None)):
            text = core.fit_part2_text_blend(sentinel.frame)
        self.assertEqual(text["weights"], [.5, .5])
        self.assertEqual(text["word_blend"]["weights"], [.5, .5])
        self.assertEqual(text["flat_weights"], {"concatenated_word": .25,
                         "separate_field_word": .25, "character": .5})
        with patch.object(core, "fit_part2_text_blend", return_value=text), \
             patch.object(core, "fit_phase_shared", return_value=sentinel.shared) as shared:
            model = core.fit_part2_shared(sentinel.frame)
        self.assertEqual(model["shared_weight"], .5)
        shared.assert_called_once_with(sentinel.frame, dict(alpha=.35, half_life=6), reference_year=2013)

    def test_ridge_training_reads_only_train_and_validation_inputs(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = {"train": reason_rows(["a", "u"], [[1, 0, 0, 0, 0], [-100] * 5]),
                      "val": reason_rows(["v"], [[0, 1, 0, 0, 0]])}
            for split, frame in frames.items():
                path = core.dataset_path(root, "part1", split)
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(path, index=False)
            # No test file exists. A test-file read would fail this check.
            with patch.object(core, "fit_part1_ridge", return_value=sentinel.ridge):
                fitted = core.train_family(root, "part1_ridge")
        metadata = fitted["metadata"]
        self.assertEqual(metadata["fit_feature_ids"], ["a", "u", "v"])
        self.assertEqual(metadata["fit_observed_label_ids"], ["a", "v"])
        self.assertTrue(metadata["validation_labels_used"])
        self.assertFalse(metadata["test_features_used"])
        self.assertFalse(metadata["hyperparameter_search"])

    def test_cli_refuses_existing_output_before_fitting(self):
        with TemporaryDirectory() as temporary, patch.object(core, "train_family") as fit:
            with self.assertRaises(FileExistsError):
                core.main(["part1_ridge", "--output", temporary])
            fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
