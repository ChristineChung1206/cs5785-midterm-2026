"""Check feature vocabulary isolation, submission alignment, and reason provenance."""

import _bootstrap  # noqa: F401; add repository src for standalone discovery

import unittest
from types import SimpleNamespace
from pathlib import Path
import copy
import tempfile
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

from midterm_evaluation import REASONS
from midterm_part1_finalization import (
    BroadPathwayEncoder, binary_decisions, build_reason_lookup, build_submission,
    fit_classifiers, link_unsuccessful_validation, select_primary,
    keep_or_create_selection, validate_saved_configuration,
    require_same_inference_model,
)


class FinalizationTests(unittest.TestCase):
    def test_inference_identity_allows_metadata_changes_but_not_weight_changes(self):
        scaler = StandardScaler()
        scaler.mean_, scaler.var_, scaler.scale_ = np.array([0.]), np.array([1.]), np.array([1.])
        scaler.n_features_in_, scaler.n_samples_seen_ = 1, 5
        classifier = LogisticRegression(C=.1)
        classifier.coef_, classifier.intercept_, classifier.classes_ = np.array([[1.]]), np.array([0.]), np.array([0, 1])
        classifier.n_features_in_ = 1
        bundle = {"scaler": scaler, "classifiers": [copy.deepcopy(classifier) for _ in range(5)],
            "feature_columns": ["emb_0"], "reason_columns": list(REASONS), "thresholds": dict.fromkeys(REASONS, .35),
            "step5_parameters": {"C": .1}, "step5_lineage": {"training_code_sha256": "old"}}
        same = copy.deepcopy(bundle)
        same["step5_lineage"]["training_code_sha256"] = "new"
        self.assertTrue(require_same_inference_model(bundle, same))
        same["classifiers"][0].coef_[0, 0] += .01
        with self.assertRaises(ValueError):
            require_same_inference_model(bundle, same)

    def test_saved_identity_ignores_code_and_output_provenance(self):
        parameters = {"classifier": {"C": .1}, "global_decision_threshold": .35}
        inputs = {"part1_train": {"sha256": "train"}, "part1_val": {"sha256": "val"}}
        config = {"task": "part1", "split": "validation", "hyperparameters": parameters,
            "input_files": inputs, "seed": 42, "thresholds": dict.fromkeys(REASONS, .35),
            "feature_spec": {"input_columns": ["emb_0"]}, "training_code_sha256": "original",
            "runtime_versions": {"example": "old"}, "output_directory": "old_folder"}
        for changed in [config, {**config, "training_code_sha256": "new", "output_directory": "new_folder",
                                "runtime_versions": {"example": "new"}}]:
            validate_saved_configuration(changed, parameters=parameters, input_columns=["emb_0"],
                input_files=inputs, seed=42, thresholds=dict.fromkeys(REASONS, .35))

    def test_saved_identity_still_rejects_parameters_data_and_decision_changes(self):
        parameters = {"classifier": {"C": .1}}
        inputs = {"part1_train": {"sha256": "train"}}
        config = {"task": "part1", "split": "validation", "hyperparameters": parameters,
            "input_files": inputs, "seed": 42, "thresholds": dict.fromkeys(REASONS, .35),
            "feature_spec": {"input_columns": ["emb_0"]}}
        for field, replacement in [("hyperparameters", {"classifier": {"C": 10}}),
                ("input_files", {"part1_train": {"sha256": "changed"}}), ("seed", 43),
                ("thresholds", dict.fromkeys(REASONS, .5)), ("feature_spec", {"input_columns": ["emb_1"]})]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_saved_configuration({**config, field: replacement}, parameters=parameters,
                    input_columns=["emb_0"], input_files=inputs, seed=42, thresholds=dict.fromkeys(REASONS, .35))

    def test_selection_preserves_original_provenance_and_rejects_new_model(self):
        original = {"selected_run_id": "model_v1", "selected_C": .1, "thresholds": dict.fromkeys(REASONS, .35),
            "train_sha256": "train", "selected_model_file_sha256": "model", "training_code_sha256": "old",
            "finalization_module_sha256": "old_helper", "runtime_versions": {"example": "old"}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection_locked.json"
            keep_or_create_selection(path, original)
            before = path.read_bytes()
            changed = {**original, "training_code_sha256": "new", "finalization_module_sha256": "new_helper",
                       "runtime_versions": {"example": "new"}}
            self.assertEqual(keep_or_create_selection(path, changed), original)
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(ValueError):
                keep_or_create_selection(path, {**changed, "selected_C": 10})

    def test_vocabulary_uses_training_counts_and_broad_levels_only(self):
        catalog = pd.DataFrame({"reactome_id": ["A", "B", "C", "D"],
            "pathway_name": ["a", "b", "c", "d"], "pathway_level": [0, 0, 1, 0]})
        train = pd.DataFrame({"ID": ["t1", "t2"], "drug_pathways": ["A;A;B;C", "A"],
                              "matched_drugs": ["d", "d"]})
        encoder = BroadPathwayEncoder(min_df=2).fit(train, catalog)
        self.assertEqual(encoder.vocabulary_, ["A"])
        self.assertEqual(encoder.training_document_frequency_["A"], 2)
        test = pd.DataFrame({"drug_pathways": ["D;C", np.nan], "matched_drugs": [np.nan, np.nan]})
        np.testing.assert_array_equal(encoder.transform(test), [[0, 0, 1], [0, 0, 0]])
        self.assertEqual(encoder.vocabulary_, ["A"])

    def test_submission_aligns_ids_and_rejects_empty_reasons(self):
        sample = pd.DataFrame({"ID": ["a", "b"], "Predicted": ["ignored", "ignored"]})
        pred = pd.DataFrame([["b", 0, 0, 0, 0, 0], ["a", 1, 0, 0, 1, 0]], columns=["ID", *REASONS])
        with self.assertRaises(ValueError):
            build_submission(sample,pred)
        pred.loc[0,"Funding"]=1
        submission = build_submission(sample, pred)
        self.assertEqual(submission.Predicted.tolist(), ["Safety Recruitment", "Funding"])
        self.assertEqual(submission.ID.tolist(), ["a", "b"])
        with self.assertRaises(ValueError):
            build_submission(sample, pred.assign(ID=["a", "a"]))
        pred.loc[0, "Funding"] = -100
        with self.assertRaises(ValueError):
            build_submission(sample, pred)

    def test_fixed_threshold_equality_and_primary_metric_ties(self):
        decisions = binary_decisions(np.full((1, 5), .35), dict.fromkeys(REASONS, .35))
        np.testing.assert_array_equal(decisions, np.ones((1, 5)))
        first = {"run_id": "first", "evaluation": SimpleNamespace(summary={"f1_samples": .8})}
        tied = {"run_id": "tied", "evaluation": SimpleNamespace(summary={"f1_samples": .8})}
        self.assertIs(select_primary([first, tied]), first)

    def test_lookup_preserves_gold_and_marks_training_predictions(self):
        train = pd.DataFrame([["t1", 1, 0, 0, 0, 0], ["t2", *([-100]*5)]], columns=["ID", *REASONS])
        val = pd.DataFrame([["v1", 0, 1, 0, 0, 0]], columns=["ID", *REASONS])
        test = pd.DataFrame({"ID": ["x1"]})
        scores = {"train": np.full((2, 5), .1), "val": np.full((1, 5), .1), "test": np.full((1, 5), .8)}
        lookup = build_reason_lookup({"train": train, "val": val, "test": test}, scores, dict.fromkeys(REASONS, .5), "run")
        by_id = lookup.set_index("ID")
        self.assertEqual(by_id.loc["t1", "Safety"], 1)
        self.assertEqual(by_id.loc["v1", "Funding"], 1)
        self.assertTrue(by_id.loc["t2", "prediction_is_training_resubstitution"])
        self.assertFalse(by_id.loc["x1", "prediction_is_training_resubstitution"])
        self.assertTrue(pd.isna(by_id.loc["t1", "model_score_Safety"]))
        self.assertEqual(by_id.loc["t2", "reason_source"], "model_prediction_train")
        self.assertFalse(by_id.loc["t2", "has_assigned_reason"])

    def test_linkage_retains_unmatched_trials_and_distinguishes_empty_predictions(self):
        lookup = pd.DataFrame({"ID": ["t1", "t2"], "reason_count": [1, 0],
            **{r: [int(r == "Safety"), 0] for r in REASONS}})
        outcomes = pd.DataFrame({"ID": ["t1_I", "t2_I", "u_I", "t1_II", "s_I"],
            "trial_id": ["t1", "t2", "u", "t1", "s"], "phase": ["I", "I", "I", "II", "I"],
            "label": [0, 0, 0, 0, 1]})
        linked = link_unsuccessful_validation(outcomes, lookup).set_index("ID")
        self.assertEqual(len(linked), 4)
        self.assertEqual(linked.loc["t2_I", "reason_group"], "matched_no_predicted_reason")
        self.assertEqual(linked.loc["u_I", "reason_group"], "no_recorded_stopping_reason")
        self.assertTrue(pd.isna(linked.loc["u_I", "Safety"]))
        self.assertEqual(linked.loc["t2_I", "Safety"], 0)

    def test_duplicate_part1_ids_are_rejected(self):
        frame = pd.DataFrame([["same", 1, 0, 0, 0, 0]], columns=["ID", *REASONS])
        with self.assertRaises(ValueError):
            build_reason_lookup({"train": frame, "val": frame, "test": pd.DataFrame({"ID": ["x"]})},
                {s: np.zeros((1, 5)) for s in ["train", "val", "test"]}, dict.fromkeys(REASONS, .5), "run")

    def test_declared_regularization_reaches_estimators(self):
        x = np.array([[-2], [-1], [1], [2]])
        y = np.tile([0, 0, 1, 1], (5, 1)).T
        models, _ = fit_classifiers(x, y, np.ones(4), C=.1)
        self.assertTrue(all(model.C == .1 and model.solver == "liblinear" for model in models))


if __name__ == "__main__":
    unittest.main(verbosity=2)
