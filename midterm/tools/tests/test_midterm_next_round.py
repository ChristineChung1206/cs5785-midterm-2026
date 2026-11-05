"""Behavioral checks for OOF isolation, pseudo weights, and text features."""

import _bootstrap  # noqa: F401; add repository src for standalone discovery
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from midterm_evaluation import REASONS, evaluate_part2
from midterm_next_round import (
    cache_complete, finish_cache, fit_character, fit_reason_weight,
    predict_candidate_outcomes, validate_fold_scope, verify_oof, validate_grouped_assignment,
)


def outcome_rows():
    return pd.DataFrame([
        {"ID": f"{phase}_{i}", "trial_id": f"t{i}", "phase": phase,
         "label": i % 2, "criteria": "eligible adult active" if i % 2 else "eligible adult placebo",
         "diseases_cleaned": "common disease", "drugs": "common medicine"}
        for phase in ["I", "II", "III"] for i in range(8)
    ])


class NextRoundTests(unittest.TestCase):
    def test_shared_trial_cannot_cross_phase_fold_boundaries(self):
        rows = outcome_rows()
        with self.assertRaisesRegex(ValueError, "trial crosses"):
            validate_fold_scope(rows.loc[rows.phase.eq("I")], rows.loc[rows.phase.eq("II")], part=2)

    def test_overlapping_row_ids_are_rejected(self):
        rows = outcome_rows()
        with self.assertRaisesRegex(ValueError, "row IDs overlap"):
            validate_fold_scope(rows.iloc[:4], rows.iloc[2:6], part=1)

    def test_zero_weight_is_independent_of_pseudo_targets(self):
        frame = pd.DataFrame({"ID": ["a", "b", "c", "d", "u", "v"]})
        for reason in REASONS:
            frame[reason] = [0, 0, 1, 1, -100, -100]
        context = {"observed": np.array([True] * 4 + [False] * 2),
                   "x": np.array([[-2.], [-1.], [1.], [2.], [-4.], [4.]]),
                   "pseudo_q": np.zeros((2, 5)), "scaler": None,
                   "clusterer": None, "mapping": {}}
        first = fit_reason_weight(frame, context, 0)
        context["pseudo_q"] = np.ones((2, 5))
        second = fit_reason_weight(frame, context, 0)
        for a, b in zip(first["models"], second["models"]):
            np.testing.assert_array_equal(a.coef_, b.coef_)
            np.testing.assert_array_equal(a.intercept_, b.intercept_)

    def test_negative_weight_is_rejected_before_fitting(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            fit_reason_weight(pd.DataFrame(), {}, -.1)

    def test_character_features_ignore_outcomes_and_reason_columns(self):
        train = outcome_rows()
        character, _, _, _ = fit_character(train)
        bundle = {"kind": "character_only", "character": character}
        before = predict_candidate_outcomes(bundle, train)
        changed = train.copy()
        changed["label"] = 1 - changed.label
        changed["Safety"] = 1
        changed["emb_0"] = 999999
        changed["start_year"] = 2099
        changed["ID"] = "new_" + changed.ID
        after = predict_candidate_outcomes(bundle, changed)
        np.testing.assert_array_equal(before.p_success, after.p_success)
        held = changed.copy()
        held["criteria"] = "zzqvzzqv unseen heldout token"
        predict_candidate_outcomes(bundle, held)
        for pipeline in character["pipelines"].values():
            self.assertNotIn("zzq", pipeline.named_steps["tfidf"].vocabulary_)

    def test_duplicate_oof_rows_are_rejected(self):
        rows = outcome_rows()
        predicted = evaluate_part2(rows, rows[["ID"]].assign(p_success=.5)).predictions.assign(fold=0)
        assignment = rows[["ID"]].assign(fold=0)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            verify_oof(rows, pd.concat([predicted, predicted.iloc[:1]]), assignment, part=2)

    def test_oof_labels_and_trial_groups_are_checked(self):
        rows = outcome_rows()
        predicted = evaluate_part2(rows, rows[["ID"]].assign(p_success=.5)).predictions.assign(fold=0)
        assignment = rows[["ID"]].assign(fold=0)
        valid = verify_oof(rows, predicted, assignment, part=2)
        self.assertEqual(len(valid), len(rows))
        corrupted = predicted.copy()
        corrupted.loc[0, "label"] = 1 - corrupted.loc[0, "label"]
        with self.assertRaisesRegex(ValueError, "labels changed"):
            verify_oof(rows, corrupted, assignment, part=2)
        predicted.loc[0, "fold"] = 1
        assignment.loc[0, "fold"] = 1
        with self.assertRaisesRegex(ValueError, "same fold"):
            verify_oof(rows, predicted, assignment, part=2)

    def test_saved_group_assignments_are_used_without_resplitting(self):
        rows = pd.DataFrame([{"ID": f"{phase}_{i}", "trial_id": f"t{i}", "phase": phase,
                              "label": i % 2} for phase in ["I", "II", "III"] for i in range(10)])
        assignment = rows[["ID", "trial_id", "phase", "label"]].copy()
        assignment["fold"] = assignment.trial_id.str[1:].astype(int) // 2
        ordered = validate_grouped_assignment(rows, assignment.sample(frac=1, random_state=7))
        pd.testing.assert_frame_equal(ordered, assignment)
        incomplete = ~rows.trial_id.eq("t9")
        with self.assertRaisesRegex(ValueError, "both outcomes"):
            validate_grouped_assignment(rows.loc[incomplete], assignment.loc[incomplete])
        assignment.loc[assignment.ID.eq("II_0"), "fold"] = 1
        with self.assertRaisesRegex(ValueError, "each trial"):
            validate_grouped_assignment(rows, assignment)

    def test_completed_study_rejects_changed_prediction_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            path = folder / "predictions.csv"
            path.write_text("ID,p\na,0.5\n")
            finish_cache(folder)
            self.assertTrue(cache_complete(folder))
            path.write_text("ID,p\na,0.9\n")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                cache_complete(folder)


if __name__ == "__main__":
    unittest.main()
