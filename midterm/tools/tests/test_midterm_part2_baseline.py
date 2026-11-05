"""Behavioral checks for text isolation, phase routing, and reason-group denominators."""

import _bootstrap  # noqa: F401; add repository src for standalone discovery

import unittest

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from midterm_evaluation import REASONS
from midterm_part2_baseline import (
    analyze_failure_reasons, build_trial_text, predict_outcome_probabilities,
    validate_outcome_rows, wilson_interval,
)


class ConstantPipeline:
    classes_ = np.array([0, 1])

    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, text):
        return np.tile([1-self.probability, self.probability], (len(text), 1))


class Part2BaselineTests(unittest.TestCase):
    def fixture(self):
        lookup = pd.DataFrame({"ID": ["t1", "t2", "t3"],
            "Safety": [1, 1, 0], "Funding": [0, 0, 0], "Efficacy": [0, 0, 0],
            "Recruitment": [1, 0, 0], "Other": [0, 0, 0], "reason_count": [2, 1, 0],
            "is_observed": [True, False, False],
            "reason_source": ["observed_train", "model_prediction_train", "model_prediction_train"],
            "prediction_is_training_resubstitution": [False, True, True]})
        predictions = pd.DataFrame({"ID": ["t1_I", "t2_I", "t3_I", "u_I", "s_I", "t1_II"],
            "trial_id": ["t1", "t2", "t3", "u", "s", "t1"],
            "phase": ["I", "I", "I", "I", "I", "II"], "label": [0, 0, 0, 0, 1, 0],
            "prediction": [1, 0, 1, 1, 0, 0], "p_success": [.8, .2, .7, .6, .3, .1]})
        return predictions, lookup

    def test_text_uses_only_three_allowed_fields_and_handles_missing_criteria(self):
        frame = pd.DataFrame({"criteria": [np.nan], "diseases_cleaned": ["['condition']"],
            "drugs": ["['compound']"], "ID": ["secret_identifier"], "label": [1],
            "Safety": ["secret_reason"], "emb_0": ["secret_embedding"], "start_year": ["secret_year"]})
        text = build_trial_text(frame).iloc[0]
        self.assertIn("condition", text)
        self.assertIn("compound", text)
        for secret in ["secret_identifier", "secret_reason", "secret_embedding", "secret_year", "nan"]:
            self.assertNotIn(secret, text)

    def test_validation_tokens_cannot_expand_training_vocabulary(self):
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        vectorizer.fit(["common training compound", "common training condition"])
        before = vectorizer.vocabulary_.copy()
        matrix = vectorizer.transform(["validationonlysentinel common"])
        self.assertEqual(vectorizer.vocabulary_, before)
        self.assertNotIn("validationonlysentinel", vectorizer.vocabulary_)
        self.assertEqual(matrix.shape[1], len(before))

    def test_phase_routing_preserves_input_row_order(self):
        frame = pd.DataFrame({"ID": ["c_III", "a_I", "b_II"], "trial_id": ["c", "a", "b"],
            "phase": ["III", "I", "II"], "criteria": ["text"]*3,
            "diseases_cleaned": ["condition"]*3, "drugs": ["drug"]*3}, index=[7, 2, 9])
        bundle = {"text_columns": ["criteria", "diseases_cleaned", "drugs"], "positive_class": 1,
            "pipelines": {"I": ConstantPipeline(.1), "II": ConstantPipeline(.2), "III": ConstantPipeline(.3)}}
        pred = predict_outcome_probabilities(bundle, frame)
        self.assertEqual(pred.ID.tolist(), frame.ID.tolist())
        np.testing.assert_allclose(pred.p_success, [.3, .1, .2])
        with self.assertRaises(ValueError):
            predict_outcome_probabilities(bundle, frame.assign(phase=["IV", "I", "II"]))

    def test_reason_denominator_uses_only_failed_trials_and_allows_overlap(self):
        predictions, lookup = self.fixture()
        analysis = analyze_failure_reasons(predictions, lookup)
        rates = analysis["reason_rates"].set_index(["phase", "reason_group"])
        self.assertEqual(len(analysis["failed_rows"]), 5)
        safety = rates.loc[("I", "Safety")]
        self.assertEqual(safety.n_trials, 2)
        self.assertEqual(safety.n_predicted_success, 1)
        self.assertEqual(safety.false_success_rate, .5)
        self.assertEqual(rates.loc[("I", "Recruitment"), "n_trials"], 1)
        self.assertEqual(rates.loc[("II", "Safety"), "n_trials"], 1)

    def test_observed_and_predicted_sources_are_separate(self):
        predictions, lookup = self.fixture()
        analysis = analyze_failure_reasons(predictions, lookup)
        rates = analysis["source_rates"].set_index(["phase", "reason_group", "source_type"])
        self.assertEqual(rates.loc[("I", "Safety", "observed"), "false_success_rate"], 1)
        self.assertEqual(rates.loc[("I", "Safety", "predicted"), "false_success_rate"], 0)
        self.assertEqual(rates.loc[("I", "Safety", "predicted"), "n_training_resubstitution_reasons"], 1)

    def test_unmatched_and_empty_predictions_remain_distinct(self):
        predictions, lookup = self.fixture()
        analysis = analyze_failure_reasons(predictions, lookup)
        rows = analysis["failed_rows"].set_index("ID")
        self.assertTrue(pd.isna(rows.loc["u_I", "Safety"]))
        self.assertEqual(rows.loc["u_I", "reason_group"], "no_recorded_stopping_reason")
        self.assertEqual(rows.loc["t3_I", "reason_group"], "matched_no_predicted_reason")
        self.assertEqual(rows.loc["t3_I", "Safety"], 0)

    def test_zero_support_has_undefined_rate_not_zero(self):
        predictions, lookup = self.fixture()
        rates = analyze_failure_reasons(predictions, lookup)["reason_rates"].set_index(["phase", "reason_group"])
        row = rates.loc[("III", "Funding")]
        self.assertEqual(row.n_trials, 0)
        self.assertTrue(pd.isna(row.false_success_rate))
        self.assertEqual(row.rate_status, "undefined_no_trials")

    def test_duplicate_trial_within_phase_is_rejected(self):
        predictions, lookup = self.fixture()
        duplicate = pd.concat([predictions, predictions.iloc[[0]].assign(ID="different_id")], ignore_index=True)
        with self.assertRaises(ValueError):
            analyze_failure_reasons(duplicate, lookup)

    def test_wilson_interval_handles_boundaries_and_no_trials(self):
        lower, upper = wilson_interval(0, 1)
        self.assertAlmostEqual(lower, 0)
        self.assertAlmostEqual(upper, .7934506856)
        self.assertTrue(all(np.isnan(x) for x in wilson_interval(0, 0)))
        with self.assertRaises(ValueError):
            wilson_interval(2, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
