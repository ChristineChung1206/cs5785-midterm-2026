"""Optional replay of the historical nested Part 1 local-model / consensus study.

Run: python tools/src/midterm_part1_local_consensus.py --out NEW_DIRECTORY
This reruns the recorded model search, so an explicit new output directory is
required. For fixed selected recipes, use python tools/src/train_core.py --help.
No recorded public submission is replaced or uploaded.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from midterm_augmentation import (cluster_memberships, estimate_cluster_reason_rates,
    logistic_parameters, predict_reason_probabilities, recover_labels)
from midterm_evaluation import REASONS, dataset_path, part1_decisions
from midterm_part1_finalization import fit_classifiers

EMB = [f"emb_{j}" for j in range(150)]
SEEDS = (42, 137, 2026)
THRESHOLDS = (.35, .50, .25)  # This order resolves ties, not a tuned ordering.
TOL = 1e-4
EPS = 1e-12
OUTNAME = "part1_local_consensus_2026_11_02"
CONFIGS = [dict(geometry=g, family=f, bandwidth=b, strength=s)
           for g in ("column", "global", "unit")
           for b in (.5, 2.)
           for f, strengths in (("svm", (.3, 3.)), ("ridge", (.1, 1.)))
           for s in strengths]
LINEAR = dict(geometry="column", family="linear", bandwidth=1., strength=1.)
PLAN = {
    "version": 1, "gold_rows": 235, "unlabeled_rows": 1014,
    "outer_folds": 4, "repeat_seeds": list(SEEDS), "inner_folds": 3,
    "calibration": "three grouped cross-fits within each fitting partition; regularized sigmoid",
    "grouping": "connected components at raw Euclidean distance <= 1e-4 across gold and unlabeled",
    "fit_scope": "remove entire held group, including unlabeled counterparts; no test features in any fit",
    "bandwidth": "gamma = multiplier / median nonzero squared distance among training gold vectors",
    "local_configs": CONFIGS, "shared_thresholds": list(THRESHOLDS),
    "controls": ["current KMeans+LR threshold .35", "same model with threshold selected by inner OOF"],
    "inner_complementarity_gate": {
        "minimum_better_groups": 3, "minimum_oracle_gain": .01,
        "maximum_local_F1_deficit": .04,
    },
    "consensus": {
        "teachers": "gold-only linear and selected local model, three calibrated committee members each",
        "positive_minimum_across_all_six": .8, "negative_maximum_across_all_six": .1,
        "otherwise": "abstain independently for each reason",
        "target": "mean committee probability", "base_pseudo_weight": .1,
        "positive_weight_cap": ".5 * gold positive count",
        "negative_weight_cap": "min(.25 * gold negative count, 2 * gold positive count)",
        "student": "five LR C=1 with exact soft cross-entropy via weighted positive/negative copies",
        "rounds": 1,
    },
    "blends": "current model + best eligible local/consensus source; source weights .25/.5/.75, inner only",
    "candidate_gate": {
        "mean_OOF_gain_vs_retuned_control": .01, "positive_repeats_at_least": 2,
        "worst_repeat_gain_minimum": -.01, "each_source_gain_minimum": -.01,
        "late_56_gain_minimum": -.02,
    },
    "caveat": "All 235 labels were previously inspected. Nested resampling is selection evidence, not a fresh external test.",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def row_f1(y, pred):
    denom = y.sum(1) + pred.sum(1)
    return np.divide(2 * (y * pred).sum(1), denom, out=np.zeros(len(y), float), where=denom > 0)


def decide(p, threshold):
    return part1_decisions(np.clip(p, 0, 1), threshold, empty_set_policy="argmax")


def f1(y, p, threshold):
    return float(row_f1(y, decide(p, threshold)).mean())


def best_threshold(y, p):
    values = [f1(y, p, t) for t in THRESHOLDS]
    k = int(np.argmax(values))
    return THRESHOLDS[k], values[k]


def near_groups(x, tolerance=TOL):
    distances = cdist(x, x)
    _, groups = connected_components(csr_matrix(distances <= tolerance), directed=False)
    return groups


@dataclass
class Data:
    frame: pd.DataFrame
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    gold: np.ndarray
    unlabeled: np.ndarray

    def scope(self, gold, eligible):
        gold, eligible = np.asarray(gold, int), np.asarray(eligible, int)
        if not set(gold) <= set(eligible):
            raise ValueError("Gold rows must belong to the feature fitting scope")
        if not np.isin(self.y[gold], [0, 1]).all():
            raise ValueError("Only observed binary labels may train supervised fits")
        return Scope(self, gold, eligible)


@dataclass
class Scope:
    data: Data
    gold: np.ndarray
    eligible: np.ndarray

    def without(self, held):
        held_groups = np.unique(self.data.groups[held])
        keep = ~np.isin(self.data.groups[self.eligible], held_groups)
        gold = self.gold[~np.isin(self.data.groups[self.gold], held_groups)]
        return self.data.scope(gold, self.eligible[keep])

    def folds(self, seed, n=3):
        y = self.data.y[self.gold]
        splitter = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
        for _, held in splitter.split(self.gold, y.argmax(1), self.data.groups[self.gold]):
            yield held, self.without(self.gold[held])


@dataclass
class Geometry:
    kind: str
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x, kind):
        if kind == "unit":
            return cls(kind, np.zeros(x.shape[1]), np.ones(x.shape[1]))
        mean = x.mean(0)
        scale = x.std(0) if kind == "column" else np.full(x.shape[1], np.sqrt(np.mean((x - mean) ** 2)))
        return cls(kind, mean, np.where(scale > 1e-12, scale, 1.))

    def transform(self, x):
        if self.kind == "unit":
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            return x / np.maximum(norms, 1e-12)
        return (x - self.mean) / self.scale


@dataclass
class BaseModel:
    geometry: Geometry
    family: str
    train_x: np.ndarray
    gamma: float
    estimators: list
    coefficients: np.ndarray | None

    def raw(self, x):
        z = self.geometry.transform(x)
        matrix = z if self.family == "linear" else np.exp(-self.gamma * cdist(z, self.train_x, "sqeuclidean"))
        if self.family == "ridge":
            return matrix @ self.coefficients
        return np.column_stack([np.full(len(z), m) if isinstance(m, float)
                                else m.decision_function(matrix) for m in self.estimators])


def fit_base(scope, config):
    d = scope.data
    geom = Geometry.fit(d.x[scope.eligible], config["geometry"])
    z = geom.transform(d.x[scope.gold])
    y = d.y[scope.gold]
    family = config["family"]
    distances = pdist(z, "sqeuclidean")
    median = float(np.median(distances[distances > 1e-12])) if np.any(distances > 1e-12) else 1.
    gamma = config["bandwidth"] / median
    matrix = z if family == "linear" else np.exp(-gamma * cdist(z, z, "sqeuclidean"))
    if family == "ridge":
        coefficients = solve(matrix + config["strength"] * np.eye(len(z)), y, assume_a="pos")
        return BaseModel(geom, family, z, gamma, [], coefficients)
    estimators = []
    for j in range(5):
        if len(np.unique(y[:, j])) == 1:
            rate = (y[:, j].sum() + 1) / (len(y) + 2)
            estimators.append(float(np.log(rate / (1 - rate))))
            continue
        if family == "svm":
            model = SVC(C=config["strength"], kernel="precomputed", max_iter=20000, tol=1e-4)
        else:
            model = LogisticRegression(**{**logistic_parameters(42), "C": config["strength"]})
        model.fit(matrix, y[:, j])
        estimators.append(model)
    return BaseModel(geom, family, z, gamma, estimators, None)


@dataclass
class Sigmoid:
    scalers: list
    estimators: list

    @classmethod
    def fit(cls, raw, y):
        scalers, models = [], []
        for j in range(5):
            s = StandardScaler().fit(raw[:, [j]])
            if len(np.unique(y[:, j])) == 1:
                m = float((y[:, j].sum() + 1) / (len(y) + 2))
            else:
                m = LogisticRegression(C=1., max_iter=500, solver="lbfgs")
                m.fit(s.transform(raw[:, [j]]), y[:, j])
            scalers.append(s)
            models.append(m)
        return cls(scalers, models)

    def transform(self, raw):
        return np.column_stack([np.full(len(raw), m) if isinstance(m, float)
                                else m.predict_proba(s.transform(raw[:, [j]]))[:, 1]
                                for j, (s, m) in enumerate(zip(self.scalers, self.estimators))])


@dataclass
class Calibrated:
    base: BaseModel
    sigmoid: Sigmoid
    members: list

    def predict(self, x):
        return self.sigmoid.transform(self.base.raw(x))

    def committee(self, x):
        return np.stack([self.sigmoid.transform(m.raw(x)) for m in self.members])


def fit_calibrated(scope, config, seed):
    raw = np.full((len(scope.gold), 5), np.nan)
    members = []
    for held, inner in scope.folds(seed):
        m = fit_base(inner, config)
        raw[held] = m.raw(scope.data.x[scope.gold[held]])
        members.append(m)
    if not np.isfinite(raw).all():
        raise RuntimeError("Incomplete calibration cross-fit")
    return Calibrated(fit_base(scope, config), Sigmoid.fit(raw, scope.data.y[scope.gold]), members)


@dataclass
class LRModel:
    geometry: Geometry
    estimators: list

    def predict(self, x):
        return predict_reason_probabilities(self.estimators, self.geometry.transform(x))


def fit_current(scope):
    d = scope.data
    # K-Means seeding is order-sensitive: retain original CSV train order.
    eligible = scope.eligible[np.argsort(d.frame.iloc[scope.eligible]["fit_order"].to_numpy(), kind="stable")]
    geom = Geometry.fit(d.x[eligible], "column")
    z = geom.transform(d.x[eligible])
    labels = np.full((len(z), 5), -100, int)
    position = {idx: k for k, idx in enumerate(eligible)}
    for idx in scope.gold:
        labels[position[idx]] = d.y[idx]
    km = KMeans(n_clusters=10, n_init=20, max_iter=500, tol=1e-4, random_state=42, algorithm="lloyd").fit(z)
    memberships = cluster_memberships(km, z, "kmeans")
    mapping = estimate_cluster_reason_rates(memberships, labels, smoothing_strength=2.)
    augmented = recover_labels(labels, memberships, mapping, threshold=.5, pseudo_weight=.1)
    models, _ = fit_classifiers(z, augmented["labels"], augmented["sample_weight"], C=.1, seed=42)
    return LRModel(geom, models)


def pseudo_targets(local_committee, linear_committee, gold_y):
    """Abstentions stay NaN; caps apply independently by reason and predicted sign."""
    all_scores = np.concatenate([local_committee, linear_committee], axis=0)
    positive = all_scores.min(0) >= .8
    negative = all_scores.max(0) <= .1
    accepted = positive | negative
    targets = np.where(accepted, all_scores.mean(0), np.nan)
    weights = accepted.astype(float) * .1
    report = []
    for j, reason in enumerate(REASONS):
        npos, nneg = int(gold_y[:, j].sum()), int((1 - gold_y[:, j]).sum())
        caps = (.5 * npos, min(.25 * nneg, 2. * npos))
        for sign, mask, cap in zip(("positive", "negative"), (positive[:, j], negative[:, j]), caps):
            mass = float(weights[mask, j].sum())
            if mass > cap:
                weights[mask, j] *= cap / mass
            report.append(dict(reason=reason, sign=sign, accepted=int(mask.sum()),
                               uncapped_mass=mass, effective_mass=float(weights[mask, j].sum()), cap=float(cap)))
    return targets, weights, report


def fit_consensus(scope, local, linear):
    d = scope.data
    unlab = np.intersect1d(scope.eligible, d.unlabeled)
    q, weights, report = pseudo_targets(local.committee(d.x[unlab]), linear.committee(d.x[unlab]), d.y[scope.gold])
    geom = Geometry.fit(d.x[scope.eligible], "column")
    gold_x, unknown_x = geom.transform(d.x[scope.gold]), geom.transform(d.x[unlab])
    models = []
    for j in range(5):
        accepted = np.isfinite(q[:, j]) & (weights[:, j] > 0)
        x = np.concatenate([gold_x, unknown_x[accepted], unknown_x[accepted]])
        y = np.concatenate([d.y[scope.gold, j], np.ones(accepted.sum()), np.zeros(accepted.sum())])
        w = np.concatenate([np.ones(len(scope.gold)), weights[accepted, j] * q[accepted, j],
                            weights[accepted, j] * (1 - q[accepted, j])])
        model = LogisticRegression(**logistic_parameters(42)).fit(x, y, sample_weight=w)
        models.append(model)
    accepted_frame = pd.DataFrame({"ID": d.frame.iloc[unlab]["ID"].to_numpy()})
    for j, reason in enumerate(REASONS):
        accepted_frame[f"target_{reason}"] = q[:, j]
        accepted_frame[f"weight_{reason}"] = weights[:, j]
    return LRModel(geom, models), report, accepted_frame


def complementary(y, local_p, local_t, current_p, current_t, groups):
    a, b = row_f1(y, decide(local_p, local_t)), row_f1(y, decide(current_p, current_t))
    group_deltas = pd.DataFrame({"group": groups, "delta": a - b}).groupby("group").delta.mean()
    stats = {"better_groups": int((group_deltas > EPS).sum()),
             "oracle_gain": float(np.maximum(a, b).mean() - b.mean()),
             "local_gap": float(a.mean() - b.mean())}
    passed = stats["better_groups"] >= 3 and stats["oracle_gain"] >= .01 and stats["local_gap"] >= -.04
    return bool(passed), stats


@dataclass
class FittedProcedure:
    current: LRModel
    local: Calibrated
    consensus: LRModel | None
    recipes: dict
    selection: dict

    def predict_all(self, x):
        p = {"current": self.current.predict(x), "local": self.local.predict(x)}
        if self.consensus is not None:
            p["consensus"] = self.consensus.predict(x)
        result = {}
        for name, recipe in self.recipes.items():
            w = recipe["source_weight"]
            result[name] = (1 - w) * p["current"] + w * p[recipe["source"]]
        return result

    def predict(self, x):
        return self.predict_all(x)["final"]


def fit_procedure(scope, seed, progress=None):
    """Every trigger, configuration, threshold and blend weight uses inner data only."""
    d, y = scope.data, scope.data.y[scope.gold]
    folds = list(scope.folds(seed))
    control_p = np.full((len(y), 5), np.nan)
    for held, inner in folds:
        control_p[held] = fit_current(inner).predict(d.x[scope.gold[held]])
    control_t, control_f = best_threshold(y, control_p)
    best, candidates = None, []
    for n, config in enumerate(CONFIGS):
        p, models = np.full((len(y), 5), np.nan), []
        for k, (held, inner) in enumerate(folds):
            m = fit_calibrated(inner, config, seed + 100 + k)
            p[held] = m.predict(d.x[scope.gold[held]])
            models.append(m)
        threshold, value = best_threshold(y, p)
        candidates.append(dict(config=config, threshold=threshold, inner_f1=value))
        if best is None or value > best[0] + EPS:
            best = value, config, threshold, p, models
        if progress and (n + 1) % 8 == 0:
            progress(f"local configurations {n+1}/{len(CONFIGS)}")
    local_f, config, local_t, local_p, inner_locals = best
    enabled, complement = complementary(y, local_p, local_t, control_p, control_t, d.groups[scope.gold])
    inner_probs = {"current": control_p, "local": local_p}
    thresholds, values = {"current": control_t, "local": local_t}, {"current": control_f, "local": local_f}
    if enabled:
        student_p = np.full((len(y), 5), np.nan)
        for k, (held, inner) in enumerate(folds):
            linear = fit_calibrated(inner, LINEAR, seed + 100 + k)
            student, _, _ = fit_consensus(inner, inner_locals[k], linear)
            student_p[held] = student.predict(d.x[scope.gold[held]])
        student_t, student_f = best_threshold(y, student_p)
        inner_probs["consensus"], thresholds["consensus"], values["consensus"] = student_p, student_t, student_f
    # Ties prefer the current method, then the local model, then the student.
    stage2 = max(values, key=values.get)
    recipes = {
        "current_fixed": dict(source="current", source_weight=0., threshold=.35),
        "current_retuned": dict(source="current", source_weight=0., threshold=control_t),
        "local": dict(source="local", source_weight=1., threshold=local_t),
        "after_consensus": dict(source=stage2, source_weight=float(stage2 != "current"), threshold=thresholds[stage2]),
    }
    final_recipe, final_f = recipes["after_consensus"].copy(), values[stage2]
    blend_records = []
    # Only complementary sources enter the blend stage.
    sources = [s for s in ("local", "consensus") if s in inner_probs]
    for source in sources:
        permitted, _ = complementary(y, inner_probs[source], thresholds[source], control_p, control_t, d.groups[scope.gold])
        if not permitted:
            continue
        for weight in (.25, .5, .75):
            p = (1 - weight) * control_p + weight * inner_probs[source]
            threshold, value = best_threshold(y, p)
            blend_records.append(dict(source=source, weight=weight, threshold=threshold, inner_f1=value))
            if value > final_f + EPS:
                final_f, final_recipe = value, dict(source=source, source_weight=weight, threshold=threshold)
    recipes["final"] = final_recipe
    current = fit_current(scope)
    local = fit_calibrated(scope, config, seed + 900)
    student, pseudo_report, pseudo_rows = None, [], None
    if enabled:
        linear = fit_calibrated(scope, LINEAR, seed + 900)
        student, pseudo_report, pseudo_rows = fit_consensus(scope, local, linear)
    selection = dict(local_config=config, local_threshold=local_t, current_threshold=control_t,
                     consensus_enabled=enabled, complementarity=complement,
                     inner_f1=values, final_inner_f1=final_f, recipes=recipes,
                     candidates=candidates, blends=blend_records, pseudo_summary=pseudo_report,
                     fit_gold_ids=d.frame.iloc[scope.gold]["ID"].tolist(),
                     fit_feature_ids=d.frame.iloc[scope.eligible]["ID"].tolist())
    return FittedProcedure(current, local, student, recipes, selection), pseudo_rows


def load_data(root):
    train = pd.read_csv(dataset_path(root, "part1", "train")).copy()
    val = pd.read_csv(dataset_path(root, "part1", "val")).copy()
    train["fit_order"] = np.arange(len(train))
    val["fit_order"] = len(train) + np.arange(len(val))
    observed = train[list(REASONS)].isin([0, 1]).all(axis=1)
    labeled = train.loc[observed].assign(source="train")
    val = val.assign(source="validation")
    unknown = train.loc[~observed].assign(source="unlabeled")
    frame = pd.concat([labeled, val, unknown], ignore_index=True)
    x, y = frame[EMB].to_numpy(), frame[list(REASONS)].to_numpy(dtype=int)
    groups = near_groups(x)
    data = Data(frame, x, y, groups, np.arange(len(labeled) + len(val)), np.arange(len(labeled) + len(val), len(frame)))
    if len(data.gold) != 235 or len(data.unlabeled) != 1014:
        raise ValueError("Unexpected input counts")
    return data, pd.read_csv(dataset_path(root, "part1", "test"))


def prediction_frame(data, evaluated, fitted, tag, repeat=-1, fold=-1):
    frame = data.frame.iloc[evaluated][["ID", "source", "start_year"]].reset_index(drop=True).copy()
    frame["group"], frame["evaluation"], frame["repeat"], frame["fold"] = data.groups[evaluated], tag, repeat, fold
    y = data.y[evaluated]
    for j, reason in enumerate(REASONS):
        frame[f"true_{reason}"] = y[:, j]
    for name, probs in fitted.predict_all(data.x[evaluated]).items():
        t = fitted.recipes[name]["threshold"]
        pred = decide(probs, t)
        frame[f"{name}_f1"] = row_f1(y, pred)
        frame[f"{name}_exact"] = (y == pred).all(1)
        frame[f"{name}_set"] = [" ".join(r for r, v in zip(REASONS, row) if v) for row in pred]
        frame[f"{name}_threshold"] = t
        for j, reason in enumerate(REASONS):
            frame[f"{name}_prob_{reason}"] = probs[:, j]
            frame[f"{name}_pred_{reason}"] = pred[:, j]
    return frame


def summarize(frame):
    names = [c[:-3] for c in frame if c.endswith("_f1")]
    return {name: {"sample_f1": float(frame[f"{name}_f1"].mean()),
                   "exact_set_rate": float(frame[f"{name}_exact"].mean())} for name in names}


def group_bootstrap(frame, seed=42):
    # Keep all repeats and all members of a group together in each bootstrap draw.
    x = frame.assign(delta=frame.final_f1 - frame.current_retuned_f1).groupby("group").delta.agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(len(x), size=(2000, len(x)))
    means = x["sum"].to_numpy()[sampled].sum(1) / x["count"].to_numpy()[sampled].sum(1)
    return [float(v) for v in np.quantile(means, [.025, .975])]


def protected_hashes(root):
    paths = [root / "models/selection.json"]
    paths += list((root / "models").rglob("*.joblib"))
    paths += list((root / "models").rglob("submission.csv"))
    return {str(p): digest(p) for p in paths if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, required=True,
                        help="New replay directory; existing directories are rejected.")
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    plan_path = out / "experiment_plan.json"
    write_json(plan_path, PLAN)
    before = protected_hashes(root)
    write_json(out / "protected_before.json", before)
    data, test = load_data(root)
    original_scope = data.scope(np.flatnonzero(data.frame.source.to_numpy() == "train"),
                                np.flatnonzero(data.frame.source.to_numpy() != "validation"))
    validation_indices = np.flatnonzero(data.frame.source.to_numpy() == "validation")
    original_probabilities = fit_current(original_scope).predict(data.x[validation_indices])
    recorded = pd.read_csv(root / 'models/part1/02_kmeans_lr/evidence/selected/selected_validation/validation_predictions.csv').set_index("ID")
    recorded = recorded.loc[data.frame.iloc[validation_indices].ID, [f"prob_{r}" for r in REASONS]].to_numpy()
    parity_gap = float(np.max(np.abs(original_probabilities - recorded)))
    if parity_gap > 1e-10:
        raise RuntimeError(f"Preflight baseline parity failed: {parity_gap}")
    metadata = {"data_hashes": {str(dataset_path(root, "part1", s)): digest(dataset_path(root, "part1", s))
                                for s in ("train", "val", "test")},
                "source_sha256": digest(__file__),
                "versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy")},
                "gold_near_groups": int(len(np.unique(data.groups[data.gold]))),
                "all_near_groups": int(len(np.unique(data.groups))),
                "preflight_current_validation_probability_gap": parity_gap}
    write_json(out / "metadata.json", metadata)
    data.frame[["ID", "source", "start_year"]].assign(group=data.groups).to_csv(out / "groups.csv", index=False)
    full_scope = data.scope(data.gold, np.arange(len(data.frame)))
    pieces, selections, pseudo = [], [], []
    started = time.time()
    for repeat, seed in enumerate(SEEDS):
        for fold, (held, scope) in enumerate(full_scope.folds(seed, n=4)):
            label = f"repeat {repeat+1}/3 fold {fold+1}/4"
            print(f"{label}: fitting {len(scope.gold)} gold rows", flush=True)
            fitted, pseudo_rows = fit_procedure(scope, seed + fold * 17,
                                                lambda s: print(f"{label}: {s}", flush=True))
            result = prediction_frame(data, data.gold[held], fitted, "grouped_oof", repeat, fold)
            pieces.append(result)
            record = dict(repeat=repeat, fold=fold, **fitted.selection)
            selections.append(record)
            if pseudo_rows is not None:
                pseudo.append(pseudo_rows.assign(repeat=repeat, fold=fold))
            pd.concat(pieces, ignore_index=True).to_csv(out / "oof_predictions.csv", index=False)
            write_json(out / "fold_selections.json", selections)
            print(f"{label}: {json.dumps(summarize(result))}", flush=True)
    oof = pd.concat(pieces, ignore_index=True)
    if not oof.groupby(["repeat", "ID"]).size().eq(1).all() or len(oof) != 235 * len(SEEDS):
        raise RuntimeError("Incomplete or duplicated outer OOF rows")
    if pseudo:
        pd.concat(pseudo, ignore_index=True).to_csv(out / "oof_pseudo_targets.csv", index=False)

    # Historical validation is diagnostic: previous rounds already inspected it.
    source = data.frame.source.to_numpy()
    years = data.frame.start_year.to_numpy()
    diagnostics, diag_tables = {}, []
    scopes = {
        "historical_validation": (np.flatnonzero(source == "train"), np.flatnonzero(source != "validation"), np.flatnonzero(source == "validation")),
        "late_train_20": (np.flatnonzero((source == "train") & (years <= 2006)), np.flatnonzero((source != "validation") & (years <= 2006)), np.flatnonzero((source == "train") & (years >= 2007) & (years <= 2009))),
        "late_combined_56": (np.flatnonzero((source != "unlabeled") & (years <= 2006)), np.flatnonzero(years <= 2006), np.flatnonzero((source != "unlabeled") & (years >= 2007) & (years <= 2009))),
    }
    for name, (gold, eligible, held) in scopes.items():
        print(f"diagnostic {name}", flush=True)
        scope = data.scope(gold, eligible)
        fitted, _ = fit_procedure(scope, 42, lambda s: print(f"{name}: {s}", flush=True))
        result = prediction_frame(data, held, fitted, name)
        result["near_early_gold_match"] = np.isin(data.groups[held], data.groups[gold])
        result.to_csv(out / f"{name}_predictions.csv", index=False)
        diag_tables.append(result)
        diagnostics[name] = {"metrics": summarize(result), "rows": len(result), "selection": fitted.selection,
                             "novel_group_metrics": summarize(result.loc[~result.near_early_gold_match])}
        if name == "historical_validation":
            saved = pd.read_csv(root / 'models/part1/02_kmeans_lr/evidence/selected/selected_validation/validation_predictions.csv').set_index("ID").loc[result.ID]
            actual = result[[f"current_fixed_prob_{r}" for r in REASONS]].to_numpy()
            expected = saved[[f"prob_{r}" for r in REASONS]].to_numpy()
            gap = float(np.max(np.abs(actual - expected)))
            if gap > 1e-10:
                raise RuntimeError(f"Current baseline does not reproduce saved validation: {gap}")
            diagnostics[name]["current_saved_probability_max_gap"] = gap
        write_json(out / "diagnostics.json", diagnostics)

    repeats = {str(k): summarize(v) for k, v in oof.groupby("repeat")}
    by_source = {str(k): summarize(v) for k, v in oof.groupby("source")}
    delta = lambda x: x["final"]["sample_f1"] - x["current_retuned"]["sample_f1"]
    repeat_gains = [delta(x) for x in repeats.values()]
    average = summarize(oof)
    time_gain = delta(diagnostics["late_combined_56"]["metrics"])
    gates = {"mean_oof_gain": delta(average) >= .01 - EPS,
             "positive_at_least_two_repeats": sum(v > EPS for v in repeat_gains) >= 2,
             "worst_repeat": min(repeat_gains) >= -.01 - EPS,
             "both_sources": all(delta(x) >= -.01 - EPS for x in by_source.values()),
             "late_56_no_large_regression": time_gain >= -.02 - EPS}
    table = pd.concat([oof, *diag_tables], ignore_index=True)
    changed = table.loc[table.final_set != table.current_retuned_set].copy()
    changed["row_f1_delta"] = changed.final_f1 - changed.current_retuned_f1
    changed.to_csv(out / "changed_trials.csv", index=False)
    reason_rows = []
    for split, frame in table.groupby("evaluation"):
        for model in average:
            for reason in REASONS:
                y, pred = frame[f"true_{reason}"].to_numpy(), frame[f"{model}_pred_{reason}"].to_numpy()
                reason_rows.append(dict(evaluation=split, model=model, reason=reason, observations=len(y),
                    support=int(y.sum()), predicted=int(pred.sum()), f1=float(f1_score(y, pred, zero_division=0))))
    pd.DataFrame(reason_rows).to_csv(out / "metrics_by_reason.csv", index=False)
    year_rows = []
    for (split, year), frame in table.groupby(["evaluation", "start_year"]):
        year_rows.append(dict(evaluation=split, year=int(year), observations=len(frame),
                             **{m: v["sample_f1"] for m, v in summarize(frame).items()}))
    pd.DataFrame(year_rows).to_csv(out / "metrics_by_year.csv", index=False)
    group_delta = oof.assign(delta=oof.final_f1 - oof.current_retuned_f1).groupby("group").delta.mean()
    summary = dict(overall=average, by_repeat=repeats, by_source=by_source, repeat_gains=repeat_gains,
                   mean_gain=delta(average), group_bootstrap_95_percent=group_bootstrap(oof),
                   group_wins=int((group_delta > EPS).sum()), group_losses=int((group_delta < -EPS).sum()),
                   diagnostic_metrics={k: v["metrics"] for k, v in diagnostics.items()},
                   consensus_enabled_folds=sum(s["consensus_enabled"] for s in selections),
                   candidate_gate=gates, candidate_allowed=all(gates.values()),
                   elapsed_seconds=time.time() - started,
                   caveat=PLAN["caveat"], candidate=None)
    write_json(out / "summary.json", summary)
    if all(gates.values()):
        print("Candidate gate passed; fitting locked procedure on all 235 gold labels", flush=True)
        fitted, pseudo_rows = fit_procedure(full_scope, 42, lambda s: print(f"final refit: {s}", flush=True))
        probabilities = fitted.predict(test[EMB].to_numpy())
        predicted = decide(probabilities, fitted.recipes["final"]["threshold"])
        submission = pd.DataFrame({"ID": test.ID, "Predicted": [" ".join(r for r, v in zip(REASONS, row) if v) for row in predicted]})
        contents = submission.to_csv(index=False).encode()
        sha = hashlib.sha256(contents).hexdigest()
        filename = f"part1_local_consensus_{sha[:12]}.csv"
        (out / filename).write_bytes(contents)
        joblib.dump(fitted, out / "candidate_model.joblib")
        restored = joblib.load(out / "candidate_model.joblib")
        np.testing.assert_allclose(restored.predict(test[EMB].to_numpy()), probabilities, atol=1e-12, rtol=0)
        if len(submission) != 312 or not submission.ID.is_unique or submission.Predicted.str.len().eq(0).any():
            raise RuntimeError("Invalid submission format")
        if pseudo_rows is not None:
            pseudo_rows.to_csv(out / "final_pseudo_targets.csv", index=False)
        write_json(out / "final_selection.json", fitted.selection)
        saved = pd.read_csv(root / 'models/part1/02_kmeans_lr/submission.csv').set_index("ID").loc[test.ID]
        summary["candidate"] = dict(filename=filename, sha256=sha, rows=312,
                                    changed_vs_public_best=int((submission.Predicted.to_numpy() != saved.Predicted.to_numpy()).sum()),
                                    model_sha256=digest(out / "candidate_model.joblib"), kaggle_uploaded=False)
        write_json(out / "summary.json", summary)
    after = protected_hashes(root)
    if after != before:
        raise RuntimeError("A protected file changed during this experiment")
    write_json(out / "verification.json", dict(protected_hashes_unchanged=True, protected_hashes=after,
               oof_rows=len(oof), unique_gold_ids=oof.ID.nunique(), gold_groups=oof.group.nunique(),
               outer_fold_count=len(selections), output_hashes={p.name: digest(p) for p in out.glob("*") if p.is_file()}))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    # Use an importable class namespace in serialized models, not __main__.
    from midterm_part1_local_consensus import main as run_main
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", category=ConvergenceWarning)
        run_main()
