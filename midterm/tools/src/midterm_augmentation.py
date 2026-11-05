"""Training-only cluster label recovery and weighted multilabel logistic regression."""

from __future__ import annotations

import inspect
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


def observed_label_mask(labels: np.ndarray) -> np.ndarray:
    """Require either five observed binary labels or five unknown labels per row."""
    labels = np.asarray(labels, dtype=float)
    if labels.ndim != 2 or labels.shape[0] == 0 or labels.shape[1] != 5:
        raise ValueError("Labels must be a nonempty n-by-5 array.")
    observed = np.isin(labels, [0, 1]).all(axis=1)
    masked = (labels == -100).all(axis=1)
    if not (observed | masked).all() or not observed.any():
        raise ValueError("Each row must be fully observed (0/1) or fully masked (-100), with observed seeds present.")
    return observed


def membership_matrix(values: np.ndarray) -> np.ndarray:
    """Validate hard one-hot assignments or soft responsibilities."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("Memberships must be a finite, nonempty matrix.")
    if ((values < 0) | (values > 1)).any() or not np.allclose(values.sum(axis=1), 1, atol=1e-8, rtol=0):
        raise ValueError("Memberships must be nonnegative and sum to one in every row.")
    return values


def cluster_memberships(clusterer, features: np.ndarray, method: str) -> np.ndarray:
    """K-Means uses hard assignments; GMM uses posterior component responsibilities."""
    if method == "kmeans":
        return membership_matrix(np.eye(clusterer.n_clusters)[clusterer.predict(features)])
    if method == "gmm":
        return membership_matrix(clusterer.predict_proba(features))
    raise ValueError("Method must be kmeans or gmm.")


def estimate_cluster_reason_rates(memberships, labels, smoothing_strength: float = 2.0):
    """Smooth visible-seed reason rates toward the visible-seed global prior.

    For cluster k and reason r:
      q[k,r] = (sum_i w[i,k]*y[i,r] + alpha*prior[r]) / (sum_i w[i,k] + alpha)
    Every sum over i uses only observed training seeds. A cluster with no observed
    membership mass falls back to the prior. These estimates are not calibrated
    population prevalences because the visible seeds were not sampled at random.
    """
    weights = membership_matrix(memberships)
    labels = np.asarray(labels, dtype=float)
    observed = observed_label_mask(labels)
    if len(weights) != len(labels):
        raise ValueError("Membership and label rows must align.")
    if not np.isfinite(smoothing_strength) or smoothing_strength <= 0:
        raise ValueError("Smoothing strength must be finite and positive.")
    prior = labels[observed].mean(axis=0)
    mass = weights[observed].sum(axis=0)
    positive_mass = weights[observed].T @ labels[observed]
    rates = (positive_mass + smoothing_strength * prior) / (mass[:, None] + smoothing_strength)
    return {"rates": rates, "global_prior": prior, "observed_membership_mass": mass,
            "positive_membership_mass": positive_mass, "smoothing_strength": float(smoothing_strength)}


def recover_labels(labels, memberships, mapping, *, threshold: float = 0.5, pseudo_weight: float = 0.1):
    """Fill every masked row, preserving observed labels exactly.

    Decision margins describe distance from the fixed decision boundary. They do
    not estimate label accuracy. Cluster assignment certainty has a separate meaning.
    """
    labels = np.asarray(labels, dtype=float)
    observed = observed_label_mask(labels)
    weights = membership_matrix(memberships)
    rates = np.asarray(mapping["rates"], dtype=float)
    if len(weights) != len(labels) or rates.shape != (weights.shape[1], 5):
        raise ValueError("Labels, memberships, and cluster rates have incompatible shapes.")
    if not np.isfinite(rates).all() or ((rates < 0) | (rates > 1)).any():
        raise ValueError("Reason rates must be finite values in [0, 1].")
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("The decision threshold must lie strictly between zero and one.")
    if not np.isfinite(pseudo_weight) or not 0 < pseudo_weight <= 1:
        raise ValueError("The pseudo-label weight must be in (0, 1].")
    scores = weights @ rates
    recovered = labels.copy().astype(int)
    recovered[~observed] = (scores[~observed] >= threshold).astype(int)
    # At threshold 0.5, the normalized margin equals 2*abs(score-0.5).
    margins = np.where(scores >= threshold, (scores-threshold)/(1-threshold), (threshold-scores)/threshold)
    entropy = -(weights * np.log(np.clip(weights, 1e-300, 1))).sum(axis=1)
    normalized_entropy = entropy / np.log(weights.shape[1]) if weights.shape[1] > 1 else np.zeros(len(weights))
    return {"labels": recovered, "observed": observed, "scores": scores,
            "sample_weight": np.where(observed, 1.0, pseudo_weight),
            "decision_margins": margins, "dominant_cluster": weights.argmax(axis=1),
            "max_membership": weights.max(axis=1), "normalized_membership_entropy": normalized_entropy}


def logistic_parameters(seed: int = 42) -> dict:
    """Use the baseline's effective L2 settings across sklearn versions."""
    parameters = {"C": 1.0, "solver": "liblinear", "max_iter": 500, "tol": 1e-4,
                  "class_weight": None, "fit_intercept": True, "random_state": seed}
    # sklearn 1.8+ expresses L2 through l1_ratio; earlier releases use penalty.
    if inspect.signature(LogisticRegression).parameters["penalty"].default == "deprecated":
        parameters["l1_ratio"] = 0.0
    else:
        parameters["penalty"] = "l2"
    return parameters


def fit_reason_classifiers(features, labels, sample_weight, *, seed: int = 42):
    """Fit five independent binary models with explicit per-row sample weights."""
    labels = np.asarray(labels)
    if not observed_label_mask(labels).all():
        raise ValueError("Classifier targets must be binary; recover masked labels first.")
    sample_weight = np.asarray(sample_weight, dtype=float)
    if sample_weight.shape != (len(labels),) or not np.isfinite(sample_weight).all() or (sample_weight <= 0).any():
        raise ValueError("Provide a finite positive weight for every training row.")
    estimators, warning_messages = [], []
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        for column in range(labels.shape[1]):
            if len(np.unique(labels[:, column])) != 2:
                raise ValueError("Every reason requires both observed classes.")
            estimator = LogisticRegression(**logistic_parameters(seed))
            estimator.fit(features, labels[:, column], sample_weight=sample_weight)
            estimators.append(estimator)
    for item in recorded:
        if issubclass(item.category, ConvergenceWarning):
            raise RuntimeError(f"Logistic regression did not converge: {item.message}")
        warning_messages.append(str(item.message))
    if any(int(estimator.n_iter_.max()) >= estimator.max_iter for estimator in estimators):
        raise RuntimeError("A classifier reached the fixed iteration limit.")
    return estimators, warning_messages


def predict_reason_probabilities(estimators, features) -> np.ndarray:
    """Return independent reason scores in the estimator list's column order."""
    if len(estimators) != 5:
        raise ValueError("Exactly five reason classifiers are required.")
    return np.column_stack([model.predict_proba(features)[:, list(model.classes_).index(1)] for model in estimators])


def predict_bundle(bundle: dict, frame):
    """Predict with a saved plain-data bundle, without refitting any component."""
    scaled = bundle["scaler"].transform(frame[bundle["feature_columns"]])
    return predict_reason_probabilities(bundle["classifiers"], scaled)
