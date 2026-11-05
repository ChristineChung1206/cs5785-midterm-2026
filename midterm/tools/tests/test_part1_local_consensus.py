"""Small behavioral checks for the local/consensus experiment (no full grid)."""
from __future__ import annotations

import _bootstrap  # noqa: F401; add repository src for standalone discovery

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

import midterm_part1_local_consensus as study


def synthetic_data():
    rng = np.random.default_rng(80)
    primary = np.tile(np.arange(5), 10)
    y = np.eye(5, dtype=int)[primary]
    x = np.eye(5)[primary] + rng.normal(scale=.2, size=(50, 5))
    # An unlabeled counterpart of held gold must also be excluded from fitting.
    unknown = np.vstack([x[0], rng.normal(size=(9, 5))])
    x = np.vstack([x, unknown])
    return study.Data(
        frame=pd.DataFrame({"ID": [f"S{i:03}" for i in range(60)]}),
        x=x,
        y=np.vstack([y, np.full((10, 5), -100, dtype=int)]),
        groups=study.near_groups(x),
        gold=np.arange(50),
        unlabeled=np.arange(50, 60),
    )


class LocalConsensusBehaviorTests(unittest.TestCase):
    def test_near_groups_are_transitive_and_do_not_merge_separated_points(self):
        tolerance = 1e-4
        x = np.array([[0., 0.], [.75 * tolerance, 0.], [1.5 * tolerance, 0.],
                      [1., 1.], [1., 1.]])
        groups = study.near_groups(x, tolerance)
        assert groups[0] == groups[1] == groups[2]
        assert groups[3] == groups[4]
        assert groups[0] != groups[3]
        assert np.linalg.norm(x[0] - x[2]) > tolerance


    def test_scope_without_removes_labeled_and_unlabeled_counterparts(self):
        data = synthetic_data()
        scope = data.scope(data.gold, np.arange(len(data.x)))
        clean = scope.without(np.array([0, 1, 2, 3, 4]))
        assert 0 not in clean.gold and 0 not in clean.eligible
        assert 50 not in clean.eligible
        assert len(clean.gold) == 45 and len(clean.eligible) == 54
        assert 51 in clean.eligible
        assert not np.intersect1d(data.groups[clean.eligible], data.groups[:5]).size
        assert len(scope.gold) == 50 and len(scope.eligible) == 60
        with self.assertRaisesRegex(ValueError, "observed binary"):
            data.scope([50], [50])
        with self.assertRaisesRegex(ValueError, "feature fitting scope"):
            data.scope([0], [1])


    def test_all_inner_folds_keep_near_duplicate_groups_out_of_fitting(self):
        data = synthetic_data()
        scope = data.scope(data.gold, np.arange(len(data.x)))
        visited = []
        for held, inner in scope.folds(42):
            held_groups = data.groups[scope.gold[held]]
            assert not np.intersect1d(held_groups, data.groups[inner.eligible]).size
            visited.extend(held.tolist())
        assert sorted(visited) == list(range(50))


    def test_pseudo_targets_abstain_per_label_and_cap_each_sign_without_mutation(self):
        gold = np.zeros((10, 5), dtype=int)
        gold[0] = 1
        local = np.full((3, 70, 5), .9)
        linear = np.full((3, 70, 5), .95)
        local[:, 20:, 0] = .02
        linear[:, 20:, 0] = .04
        linear[:, :, 1] = .5  # One teacher family disagrees: abstain on this label only.
        local[:, :, 2] = linear[:, :, 2] = .8  # Positive boundary is inclusive.
        local[:, :, 3] = linear[:, :, 3] = .1  # Negative boundary is inclusive.
        local[:, :, 4] = linear[:, :, 4] = .79
        original = [a.copy() for a in (gold, local, linear)]

        targets, weights, report = study.pseudo_targets(local, linear, gold)

        np.testing.assert_allclose(targets[:20, 0], .925)
        np.testing.assert_allclose(targets[20:, 0], .03)
        assert np.isnan(targets[:, 1]).all() and np.isnan(targets[:, 4]).all()
        assert np.all(weights[:, [1, 4]] == 0)
        np.testing.assert_allclose(weights[:20, 0].sum(), .5)
        np.testing.assert_allclose(weights[20:, 0].sum(), 2.)
        assert np.isfinite(targets[:, [0, 2, 3]]).all()
        assert len(report) == 10
        assert all(row["effective_mass"] <= row["cap"] + 1e-12 for row in report)
        for before, after in zip(original, (gold, local, linear)):
            np.testing.assert_array_equal(before, after)


    def test_zero_gold_positive_support_gives_zero_pseudo_weight(self):
        gold = np.zeros((10, 5), dtype=int)
        committee = np.full((3, 8, 5), .02)
        targets, weights, report = study.pseudo_targets(committee, committee, gold)
        assert np.isfinite(targets).all()
        assert np.all(weights == 0)
        assert all(r["effective_mass"] == 0 for r in report)


    def test_sigmoid_handles_constant_classes_and_keeps_valid_probabilities(self):
        signal = np.linspace(-3., 3., 30)
        raw = np.tile(signal[:, None], (1, 5))
        truth = np.tile((signal > 0).astype(int)[:, None], (1, 5))
        truth[:, 1] = 0
        truth[:, 2] = 1
        calibration = study.Sigmoid.fit(raw, truth)
        p = calibration.transform(raw)
        assert p.shape == (30, 5) and np.isfinite(p).all()
        assert ((p > 0) & (p < 1)).all()
        assert np.all(np.diff(p[:, 0]) > 0)
        np.testing.assert_allclose(p[:, 1], 1 / 32)
        np.testing.assert_allclose(p[:, 2], 31 / 32)


    def test_calibrated_model_ignores_labels_outside_scope(self):
        data = synthetic_data()
        clean = data.scope(data.gold, np.arange(len(data.x))).without(np.arange(5))
        config = dict(geometry="global", family="svm", bandwidth=.5, strength=.3)
        fitted = study.fit_calibrated(clean, config, seed=42)
        original_y = data.y.copy()
        altered_y = data.y.copy()
        altered_y[:5] = 1 - altered_y[:5]
        alternate = study.Data(data.frame, data.x, altered_y, data.groups, data.gold, data.unlabeled)
        altered_scope = alternate.scope(clean.gold, clean.eligible)
        fitted_alternate = study.fit_calibrated(altered_scope, config, seed=42)

        np.testing.assert_allclose(fitted.predict(data.x), fitted_alternate.predict(data.x), atol=0, rtol=0)
        np.testing.assert_allclose(fitted.committee(data.x), fitted_alternate.committee(data.x), atol=0, rtol=0)
        np.testing.assert_array_equal(data.y, original_y)
        assert fitted.committee(data.x).shape == (3, 60, 5)
        assert np.isfinite(fitted.predict(data.x)).all()


    def test_consensus_keeps_gold_targets_and_unit_weights(self):
        data = synthetic_data()
        scope = data.scope(data.gold, np.arange(len(data.x)))
        y_before = data.y.copy()
        calls = []

        class RecordingLR:
            def __init__(self, **kwargs):
                pass

            def fit(self, x, y, sample_weight):
                calls.append((x.copy(), y.copy(), sample_weight.copy()))
                return self

        class FixedTeacher:
            def committee(self, x):
                # Reason 0 accepted, other four uncertain. Every gold target must survive.
                values = np.full((3, len(x), 5), .5)
                values[:, :, 0] = .9
                return values

        replacement = patch.object(study, "LogisticRegression", RecordingLR)
        replacement.start()
        self.addCleanup(replacement.stop)
        _, _, accepted = study.fit_consensus(scope, FixedTeacher(), FixedTeacher())
        assert len(calls) == 5
        for j, (_, y, weights) in enumerate(calls):
            np.testing.assert_array_equal(y[:50], y_before[:50, j])
            np.testing.assert_array_equal(weights[:50], np.ones(50))
            assert len(y) == (70 if j == 0 else 50)
        # Soft targets contribute one positive and one negative copy, totaling base weight.
        np.testing.assert_allclose(calls[0][2][50:60], .09)
        np.testing.assert_allclose(calls[0][2][60:70], .01)
        np.testing.assert_array_equal(data.y, y_before)
        assert accepted.ID.tolist() == data.frame.iloc[data.unlabeled].ID.tolist()
        assert accepted["target_Funding"].isna().all()
        assert accepted["weight_Funding"].eq(0).all()


    def test_saved_calibrated_model_loads_in_fresh_process(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        tmp_path = Path(folder.name)
        data = synthetic_data()
        scope = data.scope(data.gold, np.arange(len(data.x))).without(np.arange(5))
        config = dict(geometry="global", family="ridge", bandwidth=.5, strength=.1)
        fitted = study.fit_calibrated(scope, config, seed=42)
        joblib.dump(fitted, tmp_path / "model.joblib")
        np.save(tmp_path / "features.npy", data.x[:7])
        expected = fitted.predict(data.x[:7])
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(study.__file__).resolve().parent) + os.pathsep + env.get("PYTHONPATH", "")
        command = (
            "import joblib, numpy as np; "
            "m=joblib.load('model.joblib'); "
            "np.save('predicted.npy',m.predict(np.load('features.npy')))"
        )
        result = subprocess.run([sys.executable, "-c", command], cwd=tmp_path,
                                env=env, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        np.testing.assert_allclose(np.load(tmp_path / "predicted.npy"), expected, atol=1e-12, rtol=0)


if __name__ == "__main__":
    unittest.main()
