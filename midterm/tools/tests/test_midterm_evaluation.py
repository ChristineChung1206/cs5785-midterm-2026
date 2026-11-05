"""Behavioral checks for metric definitions, ID alignment, and leakage safeguards."""

import _bootstrap  # noqa: F401; add repository src for standalone discovery

from pathlib import Path
import json
import math
import tempfile
import unittest

import numpy as np
import pandas as pd

from midterm_evaluation import (
    REASONS, PART1_DIRECTORY, PART2_DIRECTORY, ExperimentLogger,
    evaluate_part1, evaluate_part2, initialize_protocol,
    make_part2_grouped_folds, validate_fit_scope,
)


def part1_example():
    truth = pd.DataFrame([
        ["D1", 1, 0, 0, 1, 0], ["D2", 0, 1, 0, 0, 0],
        ["D3", 0, 0, 1, 0, 1], ["D4", 0, 0, 0, 1, 0],
    ], columns=["ID", *REASONS])
    probabilities = pd.DataFrame([
        ["D1", .8, .1, .1, .65, .2], ["D2", .1, .45, .1, .6, .1],
        ["D3", .1, .1, .7, .2, .4], ["D4", .1, .1, .1, .9, .6],
    ], columns=["ID", *REASONS])
    return truth, probabilities


def part2_example():
    reference = pd.DataFrame([
        ["V1_I", "V1", "I", 0], ["V2_I", "V2", "I", 1],
        ["V3_II", "V3", "II", 0], ["V4_II", "V4", "II", 1],
        ["V5_III", "V5", "III", 0], ["V6_III", "V6", "III", 1],
    ], columns=["ID", "trial_id", "phase", "label"])
    probabilities = pd.DataFrame({"ID": reference["ID"], "p_success": [.2, .8, .6, .7, .1, .4]})
    return reference, probabilities


def grouped_training():
    return pd.DataFrame([
        {"ID": f"T{i}_{phase}", "trial_id": f"T{i}", "phase": phase,
         "label": i % 2, "criteria": "example eligibility criteria"}
        for i in range(24) for phase in ["I", "II", "III"]
    ])


def fixture_project(root):
    for directory in [PART1_DIRECTORY, PART2_DIRECTORY]:
        (root / directory).mkdir(parents=True)
    reason_val, _ = part1_example()
    reason_val["emb_0"] = [.1, .2, .3, .4]
    reason_train = reason_val.copy()
    reason_train["ID"] = [f"Rtrain{i}" for i in range(4)]
    reason_test = reason_train.iloc[:1].drop(columns=list(REASONS)).copy()
    reason_test["ID"] = "Rtest"
    outcome_val, _ = part2_example()
    outcome_val["criteria"] = "validation criteria"
    outcome_train = grouped_training()
    outcome_test = outcome_val.iloc[:3].drop(columns="label").copy()
    outcome_test["trial_id"] = [f"Test{i}" for i in range(3)]
    outcome_test["ID"] = outcome_test["trial_id"] + "_" + outcome_test["phase"]
    for split, frame in [("train", reason_train), ("val", reason_val), ("test", reason_test)]:
        frame.to_csv(root / PART1_DIRECTORY / f"part1_{split}.csv", index=False)
    for split, frame in [("train", outcome_train), ("val", outcome_val), ("test", outcome_test)]:
        frame.to_csv(root / PART2_DIRECTORY / f"outcome_{split}.csv", index=False)
    initialize_protocol(root, n_splits=3)
    return reason_train, outcome_train


class MetricTests(unittest.TestCase):
    def test_part1_hand_calculation_and_shuffled_ids(self):
        truth, probabilities = part1_example()
        result = evaluate_part1(truth, probabilities.iloc[::-1])
        self.assertAlmostEqual(result.summary["f1_samples"], 7/12)
        self.assertAlmostEqual(result.summary["f1_macro"], .56)
        self.assertEqual(result.predictions["ID"].tolist(), truth["ID"].tolist())
        self.assertEqual(result.by_group["n_positive"].tolist(), [1, 1, 1, 2, 1])

    def test_threshold_equality_and_empty_label_sets(self):
        truth = pd.DataFrame([["X", 1, 0, 0, 0, 0]], columns=["ID", *REASONS])
        probabilities = pd.DataFrame([["X", .5, .1, .1, .1, .1]], columns=["ID", *REASONS])
        self.assertEqual(evaluate_part1(truth, probabilities).summary["f1_samples"], 1)
        truth.loc[:, list(REASONS)] = 0
        probabilities.loc[:, list(REASONS)] = 0
        self.assertEqual(evaluate_part1(truth, probabilities).summary["f1_samples"], 0)

    def test_rejects_masked_targets(self):
        truth, probabilities = part1_example()
        truth.loc[0, "Funding"] = -100
        with self.assertRaisesRegex(ValueError, "masked -100"):
            evaluate_part1(truth, probabilities)

    def test_rejects_invalid_probabilities_and_ids(self):
        truth, probabilities = part1_example()
        for value in [-.01, 1.01, np.nan, np.inf]:
            with self.subTest(value=value):
                bad = probabilities.copy()
                bad.loc[0, "Safety"] = value
                with self.assertRaises(ValueError):
                    evaluate_part1(truth, bad)
        for bad in [probabilities.iloc[:-1], pd.concat([probabilities, probabilities.iloc[:1]])]:
            with self.assertRaises(ValueError):
                evaluate_part1(truth, bad)
        bad = probabilities.copy()
        bad.loc[0, "ID"] = "unexpected"
        with self.assertRaisesRegex(ValueError, "missing, 1 extra"):
            evaluate_part1(truth, bad)

    def test_part2_uses_probabilities_for_auc_and_ap(self):
        truth, probabilities = part2_example()
        result = evaluate_part2(truth, probabilities.iloc[::-1])
        self.assertTrue(np.allclose(result.by_group["f1"], [1, 2/3, 0]))
        self.assertTrue(np.allclose(result.by_group["roc_auc"], 1))
        self.assertTrue(np.allclose(result.by_group["average_precision"], 1))
        expected = np.mean([-math.log(.8), -math.log(.8), -math.log(.4),
                            -math.log(.7), -math.log(.9), -math.log(.4)])
        self.assertAlmostEqual(result.summary["log_loss_equal_phase"], expected)

    def test_equal_phase_loss_does_not_follow_phase_row_counts(self):
        truth, probabilities = part2_example()
        extra = truth.iloc[:2].copy()
        extra["trial_id"] = ["Extra1", "Extra2"]
        extra["ID"] = extra["trial_id"] + "_I"
        extra_prob = pd.DataFrame({"ID": extra["ID"], "p_success": [.2, .8]})
        original = evaluate_part2(truth, probabilities)
        changed = evaluate_part2(pd.concat([truth, extra]), pd.concat([probabilities, extra_prob]))
        self.assertAlmostEqual(original.summary["log_loss_equal_phase"], changed.summary["log_loss_equal_phase"])
        self.assertNotAlmostEqual(changed.summary["log_loss_equal_phase"], changed.summary["log_loss_pooled"])

    def test_undefined_auc_is_explicit_and_json_uses_null(self):
        truth, probabilities = part2_example()
        truth.loc[truth["phase"].eq("I"), "label"] = 0
        result = evaluate_part2(truth, probabilities)
        phase = result.by_group.set_index("phase").loc["I"]
        self.assertTrue(np.isnan(phase["roc_auc"]))
        self.assertTrue(np.isnan(phase["average_precision"]))
        self.assertEqual(phase["roc_auc_status"], "undefined_single_class")
        with tempfile.TemporaryDirectory() as directory:
            result.save(Path(directory))
            text = (Path(directory) / "metrics.json").read_text()
            self.assertNotIn("NaN", text)
            self.assertIsNone(json.loads(text)["by_group"][0]["roc_auc"])

    def test_missing_phase_has_no_equal_phase_score(self):
        truth, probabilities = part2_example()
        truth = truth.loc[truth["phase"].ne("III")]
        probabilities = probabilities.loc[probabilities["ID"].isin(truth["ID"])]
        self.assertTrue(np.isnan(evaluate_part2(truth, probabilities).summary["log_loss_equal_phase"]))

    def test_log_loss_at_probability_boundaries_is_finite(self):
        truth, probabilities = part2_example()
        probabilities["p_success"] = 1-truth["label"].to_numpy()
        result = evaluate_part2(truth, probabilities)
        self.assertTrue(np.isfinite(result.summary["log_loss_equal_phase"]))
        self.assertGreater(result.summary["log_loss_equal_phase"], 30)


class ProtocolAndLoggingTests(unittest.TestCase):
    def test_protocol_reuses_saved_folds_without_regenerating(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_project(root)
            path = root / 'data/audit/protocol/part2_train_folds.csv'
            original = path.read_bytes()
            with patch("midterm_evaluation.make_part2_grouped_folds", side_effect=RuntimeError("Unexpected regeneration")):
                initialize_protocol(root, n_splits=3)
            self.assertEqual(path.read_bytes(), original)

    def test_protocol_rejects_saved_fold_leakage_and_changed_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_project(root)
            path = root / 'data/audit/protocol/part2_train_folds.csv'
            original = pd.read_csv(path)
            broken = original.copy()
            broken.loc[0, "fold"] = (broken.loc[0, "fold"] + 1) % 3
            broken.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "keep each trial together"):
                initialize_protocol(root, n_splits=3)
            broken = original.copy()
            broken.loc[0, "label"] = 1 - broken.loc[0, "label"]
            broken.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "training rows and labels"):
                initialize_protocol(root, n_splits=3)

    def test_grouped_folds_are_reproducible_and_keep_trials_together(self):
        data = grouped_training()
        first = make_part2_grouped_folds(data, n_splits=3)
        second = make_part2_grouped_folds(data, n_splits=3)
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(first.groupby("trial_id")["fold"].nunique().eq(1).all())
        self.assertEqual(set(first["fold"]), {0, 1, 2})
        for fold in range(3):
            fitted = set(first.loc[first["fold"].ne(fold), "trial_id"])
            held_out = set(first.loc[first["fold"].eq(fold), "trial_id"])
            self.assertFalse(fitted & held_out)

    def test_declared_fit_scope_rejects_nontraining_ids(self):
        with self.assertRaisesRegex(ValueError, "nontraining"):
            validate_fit_scope(["TRAIN", "VAL"], ["TRAIN"])
        scope = validate_fit_scope(["TRAIN", "TRAIN"], ["TRAIN"])
        self.assertEqual(scope["declared_fit_rows"], 2)
        self.assertEqual(scope["unique_training_trials"], 1)

    def test_logger_records_real_reference_and_prevents_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training, _ = fixture_project(root)
            logger = ExperimentLogger(root)
            _, probabilities = part1_example()
            arguments = dict(task="part1", probabilities=probabilities, model_name="fixture_model",
                feature_spec={"input_columns": ["emb_0"], "representation": "fixture"}, hyperparameters={"C": 1},
                fit_trial_ids={"scaler": training["ID"].tolist(), "classifier": training["ID"].tolist()},
                hypothesis="Verify the logging contract.", notes="Synthetic test fixture only.", run_id="fixture_run")
            np.random.seed(123)
            expected_next_random = np.random.random()
            np.random.seed(123)
            folder = logger.log_validation_run(**arguments)
            self.assertEqual(np.random.random(), expected_next_random)
            self.assertTrue((folder / "validation_predictions.csv").exists())
            config = json.loads((folder / "config.json").read_text())
            self.assertEqual(config["declared_fit_scope"]["scaler"]["unique_training_trials"], 4)
            self.assertEqual(config["thresholds"]["Funding"], .5)
            registry = pd.read_csv(root / 'models/new_experiments/experiment_log.csv')
            self.assertEqual(len(registry), 1)
            self.assertAlmostEqual(registry.iloc[0]["primary_value"], 7/12)
            with self.assertRaises(FileExistsError):
                logger.log_validation_run(**arguments)
            self.assertEqual(len(logger.rebuild_registry()), 1)

    def test_new_logger_preserves_curated_historical_index_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = root / "models/experiment_log.csv"
            history.parent.mkdir()
            frozen = b"run_id,primary_value\r\nhistorical_run,0.88568\r\n"
            history.write_bytes(frozen)
            training, _ = fixture_project(root)
            self.assertEqual(history.read_bytes(), frozen)
            logger = ExperimentLogger(root)
            _, probabilities = part1_example()
            folder = logger.log_validation_run(task="part1", probabilities=probabilities,
                model_name="fixture_model", feature_spec={"input_columns": ["emb_0"]},
                hyperparameters={}, fit_trial_ids={"classifier": training["ID"].tolist()},
                hypothesis="Keep new runs separate.", notes="Synthetic fixture.", run_id="new_run")
            self.assertEqual(folder, (root / "models/new_experiments/runs/new_run").resolve())
            self.assertEqual(len(logger.rebuild_registry()), 1)
            self.assertEqual(history.read_bytes(), frozen)

    def test_logger_blocks_leakage_and_forbidden_outcome_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, training = fixture_project(root)
            logger = ExperimentLogger(root)
            _, probabilities = part2_example()
            arguments = dict(task="part2", probabilities=probabilities, model_name="fixture_model",
                feature_spec={"input_columns": ["criteria"]}, hyperparameters={},
                fit_trial_ids={"classifier": training["trial_id"].tolist()},
                hypothesis="Check guards.", notes="Synthetic test fixture only.")
            bad_scope = dict(arguments, fit_trial_ids={"classifier": ["V1"]})
            with self.assertRaisesRegex(ValueError, "nontraining"):
                logger.log_validation_run(**bad_scope)
            for forbidden in ["trial_id", "label", "Safety", "emb_0"]:
                with self.subTest(forbidden=forbidden):
                    bad = dict(arguments, feature_spec={"input_columns": [forbidden]})
                    with self.assertRaisesRegex(ValueError, "forbidden"):
                        logger.log_validation_run(**bad)
            self.assertEqual(len(pd.read_csv(root / 'models/new_experiments/experiment_log.csv')), 0)

    def test_changed_validation_file_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training, _ = fixture_project(root)
            logger = ExperimentLogger(root)
            path = root / PART1_DIRECTORY / "part1_val.csv"
            path.write_text(path.read_text()+"\n")
            _, probabilities = part1_example()
            with self.assertRaisesRegex(ValueError, "changed"):
                logger.log_validation_run(task="part1", probabilities=probabilities, model_name="fixture",
                    feature_spec={"input_columns": ["emb_0"]}, hyperparameters={},
                    fit_trial_ids={"model": training["ID"].tolist()}, hypothesis="Guard check.", notes="Fixture.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
