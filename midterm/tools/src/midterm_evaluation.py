"""Consistent validation metrics, grouped folds, and experiment records for the midterm."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import json
import random
import re
import shutil
import tempfile
import uuid

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold


REASONS = ("Safety", "Funding", "Efficacy", "Recruitment", "Other")
PHASES = ("I", "II", "III")
PART1_DIRECTORY = 'data/part1'
PART2_DIRECTORY = 'data/part2'
LOG_LOSS_EPSILON = 1e-15
REGISTRY_COLUMNS = (
    "run_id", "created_at_utc", "task", "model_name", "seed", "hypothesis", "notes",
    "feature_spec_json", "hyperparameters_json", "thresholds_json", "validation_rows",
    "primary_metric", "primary_value", "higher_is_better", "run_directory",
)


def _validate_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")


def set_random_seed(seed: int = 42) -> None:
    """Seed Python and NumPy; also pass this seed to each estimator's random_state."""
    _validate_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_ready(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_ids(frame: pd.DataFrame, name: str) -> None:
    if not isinstance(frame, pd.DataFrame) or "ID" not in frame:
        raise ValueError(f"{name} must be a DataFrame containing ID.")
    if frame.columns.duplicated().any():
        raise ValueError(f"{name} has duplicate column names.")
    if frame.empty or frame["ID"].isna().any() or not frame["ID"].is_unique:
        raise ValueError(f"{name} must have nonempty, unique, nonmissing IDs.")
    if not frame["ID"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError(f"{name} IDs must be nonempty strings.")


def _align(reference: pd.DataFrame, predictions: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    _unique_ids(reference, "reference")
    _unique_ids(predictions, "predictions")
    if set(predictions.columns) != {"ID", *columns}:
        raise ValueError(f"Prediction columns must be exactly ID and {list(columns)}.")
    reference_ids, prediction_ids = set(reference["ID"]), set(predictions["ID"])
    if reference_ids != prediction_ids:
        raise ValueError(
            f"Prediction IDs do not match reference IDs: "
            f"{len(reference_ids-prediction_ids)} missing, {len(prediction_ids-reference_ids)} extra."
        )
    return predictions.set_index("ID").loc[reference["ID"], list(columns)].reset_index()


def _numeric(frame: pd.DataFrame, columns: Sequence[str], *, probabilities: bool) -> np.ndarray:
    try:
        values = frame[list(columns)].to_numpy(dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Expected numeric columns: {list(columns)}.") from error
    if not np.isfinite(values).all():
        raise ValueError("Targets and probabilities must be finite and nonmissing.")
    valid = ((values >= 0) & (values <= 1)) if probabilities else np.isin(values, [0, 1])
    if not valid.all():
        label = "Probabilities must be in [0, 1]." if probabilities else "Targets must be 0/1; masked -100 is not a negative label."
        raise ValueError(label)
    return values


def _thresholds(value: float | Mapping[str, float], groups: Sequence[str]) -> dict[str, float]:
    if isinstance(value, Mapping):
        if set(value) != set(groups):
            raise ValueError(f"Threshold mappings must contain exactly {list(groups)}.")
        result = {group: float(value[group]) for group in groups}
    else:
        result = {group: float(value) for group in groups}
    if any(not np.isfinite(t) or not 0 <= t <= 1 for t in result.values()):
        raise ValueError("Every threshold must be finite and in [0, 1].")
    return result


@dataclass
class EvaluationResult:
    task: str
    summary: dict[str, float | int]
    by_group: pd.DataFrame
    predictions: pd.DataFrame
    thresholds: dict[str, float]

    def save(self, directory: Path) -> None:
        """Save metrics, support counts, and ID-aligned probabilities and decisions."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        _write_json(directory / "metrics.json", {
            "task": self.task, "summary": self.summary, "thresholds": self.thresholds,
            "by_group": self.by_group.to_dict(orient="records"),
        })
        self.by_group.to_csv(directory / "metrics_by_group.csv", index=False)
        self.predictions.to_csv(directory / "validation_predictions.csv", index=False)


def part1_decisions(scores, thresholds=0.5, *, empty_set_policy="allow"):
    """Threshold independently, optionally assigning the highest score to an empty row.

    Ties use REASONS order. The default retains historical experiment semantics;
    the final submission explicitly requests the argmax policy.
    """
    values = np.asarray(scores, dtype=float)
    limits = _thresholds(thresholds, REASONS)
    if values.ndim != 2 or values.shape[1] != len(REASONS) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Provide five finite reason probabilities in [0, 1].")
    if empty_set_policy not in {"allow", "argmax"}:
        raise ValueError("empty_set_policy must be allow or argmax.")
    decisions = (values >= np.array([limits[r] for r in REASONS])).astype(int)
    if empty_set_policy == "argmax":
        empty = np.flatnonzero(decisions.sum(axis=1) == 0)
        decisions[empty, values[empty].argmax(axis=1)] = 1
    return decisions


def evaluate_part1(
    reference: pd.DataFrame, probabilities: pd.DataFrame,
    thresholds: float | Mapping[str, float] = 0.5, *, empty_set_policy: str = "allow",
) -> EvaluationResult:
    """Evaluate the same reason-decision policy used at inference."""
    aligned = _align(reference, probabilities, REASONS)
    truth = _numeric(reference, REASONS, probabilities=False).astype(int)
    scores = _numeric(aligned, REASONS, probabilities=True)
    limits = _thresholds(thresholds, REASONS)
    decisions = part1_decisions(scores, limits, empty_set_policy=empty_set_policy)
    rows = []
    saved = reference[["ID"]].reset_index(drop=True).copy()
    for j, reason in enumerate(REASONS):
        y, pred = truth[:, j], decisions[:, j]
        rows.append({"reason": reason, "n_positive": int(y.sum()), "n_negative": int((1-y).sum()),
                     "f1": f1_score(y, pred, zero_division=0),
                     "precision": precision_score(y, pred, zero_division=0),
                     "recall": recall_score(y, pred, zero_division=0)})
        saved[f"true_{reason}"] = y
        saved[f"prob_{reason}"] = scores[:, j]
        saved[f"pred_{reason}"] = pred
    summary = {"n_rows": len(reference),
               "f1_samples": f1_score(truth, decisions, average="samples", zero_division=0),
               "f1_macro": f1_score(truth, decisions, average="macro", zero_division=0)}
    if empty_set_policy == "argmax":
        fallback = (scores >= np.array([limits[r] for r in REASONS])).sum(axis=1) == 0
        saved["empty_set_fallback_applied"] = fallback
        summary["n_empty_set_fallbacks"] = int(fallback.sum())
    return EvaluationResult("part1", summary, pd.DataFrame(rows), saved, limits)


def _outcome_reference(reference: pd.DataFrame) -> None:
    _unique_ids(reference, "outcome reference")
    required = {"trial_id", "phase", "label"}
    if not required.issubset(reference.columns):
        raise ValueError(f"Outcome references require {sorted(required)}.")
    _numeric(reference, ["label"], probabilities=False)
    if reference["trial_id"].isna().any() or not reference["phase"].isin(PHASES).all():
        raise ValueError("Trial IDs must be present and phases must be I, II, or III.")
    if reference.duplicated(["trial_id", "phase"]).any():
        raise ValueError("Each trial-phase pair must appear only once.")
    if reference.groupby("trial_id")["label"].nunique().gt(1).any():
        raise ValueError("A trial must have the same outcome label across phases.")


def evaluate_part2(
    reference: pd.DataFrame, probabilities: pd.DataFrame,
    thresholds: float | Mapping[str, float] = 0.5,
) -> EvaluationResult:
    """Evaluate success (label=1) per phase; compute AP and ROC-AUC from probabilities."""
    _outcome_reference(reference)
    aligned = _align(reference, probabilities, ["p_success"])
    scores = _numeric(aligned, ["p_success"], probabilities=True).ravel()
    limits = _thresholds(thresholds, PHASES)
    saved = reference[["ID", "trial_id", "phase", "label"]].reset_index(drop=True).copy()
    saved["p_success"] = scores
    saved["prediction"] = (scores >= saved["phase"].map(limits).to_numpy()).astype(int)
    rows = []
    for phase in PHASES:
        sub = saved.loc[saved["phase"].eq(phase)]
        y = sub["label"].to_numpy(dtype=int)
        p = sub["p_success"].to_numpy()
        pred = sub["prediction"].to_numpy()
        row = {"phase": phase, "n_rows": len(sub), "n_unique_trials": sub["trial_id"].nunique(),
               "n_positive": int(y.sum()), "n_negative": int((1-y).sum())}
        if not len(sub):
            row.update({k: np.nan for k in ("f1", "roc_auc", "average_precision", "log_loss", "tn", "fp", "fn", "tp")})
            row.update(roc_auc_status="undefined_no_rows", average_precision_status="undefined_no_rows")
        else:
            both_classes = len(np.unique(y)) == 2
            row.update(
                f1=f1_score(y, pred, zero_division=0),
                roc_auc=roc_auc_score(y, p) if both_classes else np.nan,
                average_precision=average_precision_score(y, p) if y.sum() else np.nan,
                log_loss=log_loss(y, np.clip(p, LOG_LOSS_EPSILON, 1-LOG_LOSS_EPSILON), labels=[0, 1]),
                roc_auc_status="ok" if both_classes else "undefined_single_class",
                average_precision_status="ok" if y.sum() else "undefined_no_positives",
            )
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            row.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
        rows.append(row)
    groups = pd.DataFrame(rows)
    summary = {"n_rows": len(saved), "n_unique_trials": saved["trial_id"].nunique(),
               "log_loss_pooled": log_loss(saved["label"], np.clip(scores, LOG_LOSS_EPSILON, 1-LOG_LOSS_EPSILON), labels=[0, 1]),
               "log_loss_equal_phase": float(groups["log_loss"].mean()) if groups["n_rows"].gt(0).all() else np.nan}
    return EvaluationResult("part2", summary, groups, saved, limits)


def validate_fit_scope(fit_trial_ids: Sequence[str], allowed_training_ids: Sequence[str]) -> dict:
    """Validate declared fit IDs against this task's training trial IDs."""
    if isinstance(fit_trial_ids, (str, bytes)):
        raise ValueError("Pass a sequence of trial IDs, not a string.")
    ids = list(fit_trial_ids)
    if not ids or any(not isinstance(item, str) or not item.strip() for item in ids):
        raise ValueError("Fit IDs must be nonempty strings and contain at least one row.")
    unexpected = set(ids) - set(allowed_training_ids)
    if unexpected:
        raise ValueError(f"Fit scope contains {len(unexpected)} nontraining trial IDs: {sorted(unexpected)[:5]}.")
    digest = hashlib.sha256("\n".join(sorted(set(ids))).encode()).hexdigest()
    return {"declared_fit_rows": len(ids), "unique_training_trials": len(set(ids)), "unique_trial_ids_sha256": digest}


def make_part2_grouped_folds(training: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """Assign training rows to optional CV folds, grouping all phases of each trial."""
    _outcome_reference(training)
    set_random_seed(seed)
    if not isinstance(n_splits, int) or n_splits < 2 or n_splits > training["trial_id"].nunique():
        raise ValueError("n_splits must be between 2 and the number of training trials.")
    strata = training["phase"].astype(str) + "_" + training["label"].astype(str)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    assignment = np.full(len(training), -1, dtype=int)
    for fold, (_, held_out) in enumerate(splitter.split(np.zeros(len(training)), strata, groups=training["trial_id"])):
        assignment[held_out] = fold
    result = training[["ID", "trial_id", "phase", "label"]].reset_index(drop=True).copy()
    result["fold"] = assignment
    if (assignment < 0).any() or result.groupby("trial_id")["fold"].nunique().gt(1).any():
        raise RuntimeError("Invalid grouped fold assignment.")
    return result


def dataset_path(project_root: Path, task: str, split: str) -> Path:
    if task not in ("part1", "part2") or split not in ("train", "val", "test"):
        raise ValueError("Expected task part1/part2 and split train/val/test.")
    directory, prefix = (PART1_DIRECTORY, "part1") if task == "part1" else (PART2_DIRECTORY, "outcome")
    return Path(project_root) / directory / f"{prefix}_{split}.csv"


def initialize_protocol(project_root: Path, seed: int = 42, n_splits: int = 5) -> dict:
    """Create reproducible split records and optional grouped folds without fitting models."""
    project_root = Path(project_root).resolve()
    set_random_seed(seed)
    protocol = {
        "version": "1.0", "seed": seed, "selection_split": "provided_validation",
        "part1": {"primary_metric": "f1_samples", "higher_is_better": True,
                  "report": ["f1_samples", "f1_macro", "f1_per_reason", "positive_support_per_reason"]},
        "part2": {"primary_metric": "log_loss_equal_phase", "higher_is_better": False,
                  "report_per_phase": ["f1", "roc_auc", "average_precision", "log_loss", "positive_support", "negative_support"],
                  "positive_class": 1, "PR_AUC_definition": "sklearn average_precision_score; no trapezoidal PR integration"},
        "decision_rule": "probability >= threshold", "initial_threshold": 0.5,
        "f1_zero_division": 0, "log_loss_clipping_epsilon": LOG_LOSS_EPSILON,
        "undefined_auc_policy": "ROC-AUC: missing for single-class/no-row phases; AP: missing for no-positive/no-row phases; JSON uses null.",
        "preprocessing": "Fit each model, scaler, vocabulary, clusterer, and imputer using training data only; transform validation/test.",
        "part2_forbidden_inputs": ["ID", "trial_id", "label", *REASONS, "emb_0 through emb_149"],
        "hyperparameter_search": "Start with a small prespecified set. Record a hypothesis and every tried setting, including thresholds and unsuccessful runs. No automatic search is performed here.",
        "part2_optional_cv": {"n_splits": n_splits, "group": "trial_id", "strata": "phase + outcome label",
                              "method": "StratifiedGroupKFold", "fit_preprocessing_inside_each_training_fold": True},
    }
    split_manifest, task_data = {}, {}
    for task in ("part1", "part2"):
        task_data[task] = {}
        ids_per_split = {}
        for split in ("train", "val", "test"):
            path = dataset_path(project_root, task, split)
            df = pd.read_csv(path)
            _unique_ids(df, f"{task}_{split}")
            task_data[task][split] = df
            group_col = "ID" if task == "part1" else "trial_id"
            ids_per_split[split] = set(df[group_col])
            split_manifest[f"{task}_{split}"] = {"relative_path": str(path.relative_to(project_root)),
                "sha256": file_sha256(path), "rows": len(df), "unique_trials": df[group_col].nunique()}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if ids_per_split[a] & ids_per_split[b]:
                raise ValueError(f"{task}: trial overlap between {a} and {b}.")
    folder = project_root / 'data/audit/protocol'
    folder.mkdir(parents=True, exist_ok=True)
    for filename, contents in (("protocol.json", protocol), ("data_split_manifest.json", split_manifest)):
        path = folder / filename
        if path.exists() and json.loads(path.read_text()) != _json_ready(contents):
            raise ValueError(f"Existing {filename} differs. Review and version the protocol before replacing it.")
        _write_json(path, contents)
    fold_path = folder / "part2_train_folds.csv"
    if fold_path.exists():
        # Retain the assignments used by saved OOF predictions across library versions.
        folds = pd.read_csv(fold_path)
        reference = task_data["part2"]["train"][["ID", "trial_id", "phase", "label"]].reset_index(drop=True)
        if (list(folds.columns) != [*reference.columns, "fold"] or
                not folds[list(reference.columns)].equals(reference)):
            raise ValueError("Saved folds do not match the current training rows and labels.")
        if (folds["fold"].isna().any() or set(folds["fold"]) != set(range(n_splits)) or
                folds.groupby("trial_id")["fold"].nunique().gt(1).any()):
            raise ValueError("Saved folds must contain every fold and keep each trial together.")
    else:
        folds = make_part2_grouped_folds(task_data["part2"]["train"], n_splits, seed)
        folds.to_csv(fold_path, index=False)
    fold_summary = folds.groupby(["fold", "phase"]).agg(
        rows=("ID", "size"), unique_trials=("trial_id", "nunique"), positives=("label", "sum"), success_rate=("label", "mean")
    ).reset_index()
    fold_summary.to_csv(folder / "part2_fold_summary.csv", index=False)
    log_dir = project_root / 'models/new_experiments'
    (log_dir / "runs").mkdir(parents=True, exist_ok=True)
    if not (log_dir / "experiment_log.csv").exists():
        pd.DataFrame(columns=REGISTRY_COLUMNS).to_csv(log_dir / "experiment_log.csv", index=False)
    return {"protocol": protocol, "split_manifest": split_manifest, "fold_summary": fold_summary}


class ExperimentLogger:
    """Save new runs separately from the curated historical experiment index."""

    def __init__(self, project_root: Path):
        self.root = Path(project_root).resolve()
        protocol_path = self.root / 'data/audit/protocol' / "protocol.json"
        if not protocol_path.exists():
            initialize_protocol(self.root)
        self.protocol = json.loads(protocol_path.read_text())
        self.manifest = json.loads((protocol_path.parent / "data_split_manifest.json").read_text())
        self.directory = self.root / 'models/new_experiments'
        (self.directory / "runs").mkdir(parents=True, exist_ok=True)

    def log_validation_run(
        self, *, task: str, probabilities: pd.DataFrame, model_name: str,
        feature_spec: Mapping[str, Any], hyperparameters: Mapping[str, Any],
        fit_trial_ids: Mapping[str, Sequence[str]], hypothesis: str, notes: str,
        seed: int = 42, thresholds: float | Mapping[str, float] = 0.5, run_id: str | None = None,
    ) -> Path:
        if task not in ("part1", "part2"):
            raise ValueError("task must be part1 or part2.")
        if not model_name.strip() or not hypothesis.strip() or not notes.strip():
            raise ValueError("Supply model_name, hypothesis, and notes for each completed run.")
        _validate_seed(seed)
        data = {}
        for split in ("train", "val"):
            path = dataset_path(self.root, task, split)
            if file_sha256(path) != self.manifest[f"{task}_{split}"]["sha256"]:
                raise ValueError(f"{task}_{split} changed after protocol initialization.")
            data[split] = pd.read_csv(path)
        input_columns = feature_spec.get("input_columns", [])
        if not isinstance(input_columns, (list, tuple)) or not input_columns:
            raise ValueError("feature_spec must declare a nonempty input_columns list of raw feature names.")
        forbidden = {"ID", "trial_id", "label", *REASONS}
        if task == "part2":
            forbidden |= {f"emb_{i}" for i in range(150)}
        bad_columns = set(input_columns) & forbidden
        if bad_columns or not set(input_columns).issubset(data["train"].columns):
            raise ValueError(f"Invalid or forbidden input columns: {sorted(bad_columns | (set(input_columns)-set(data['train'].columns)))}.")
        group_column = "ID" if task == "part1" else "trial_id"
        if not fit_trial_ids:
            raise ValueError("Declare fit_trial_ids separately for the model and every learned preprocessor.")
        scope = {component: validate_fit_scope(ids, data["train"][group_column]) for component, ids in fit_trial_ids.items()}
        evaluator = evaluate_part1 if task == "part1" else evaluate_part2
        result = evaluator(data["val"], probabilities, thresholds)
        primary_metric = self.protocol[task]["primary_metric"]
        primary_value = result.summary[primary_metric]
        if not np.isfinite(primary_value):
            raise ValueError("The primary validation metric is undefined; investigate before logging a run.")
        now = datetime.now(timezone.utc)
        run_id = run_id or f"{task}_{now.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", run_id):
            raise ValueError("run_id must be a safe filename of 1-80 characters.")
        destination = self.directory / "runs" / run_id
        if destination.exists():
            raise FileExistsError(f"Run already exists: {run_id}")
        config = {"run_id": run_id, "created_at_utc": now.isoformat(), "task": task, "split": "validation",
                  "model_name": model_name, "feature_spec": feature_spec, "hyperparameters": hyperparameters,
                  "seed": seed, "thresholds": result.thresholds, "hypothesis": hypothesis, "notes": notes,
                  "declared_fit_scope": scope, "protocol": self.protocol,
                  "input_files": {f"{task}_{split}": self.manifest[f"{task}_{split}"] for split in ("train", "val")},
                  "evaluation_module_sha256": file_sha256(Path(__file__))}
        staged = Path(tempfile.mkdtemp(prefix=".pending_", dir=self.directory))
        try:
            _write_json(staged / "config.json", config)
            result.save(staged)
            staged.rename(destination)
        except Exception:
            shutil.rmtree(staged, ignore_errors=True)
            raise
        self.rebuild_registry()
        return destination

    def rebuild_registry(self) -> pd.DataFrame:
        """Rebuild the summary from saved runs so it can be recovered without losing history."""
        records = []
        for folder in sorted((self.directory / "runs").iterdir()):
            if not folder.is_dir():
                continue
            config = json.loads((folder / "config.json").read_text())
            metrics = json.loads((folder / "metrics.json").read_text())
            primary = config["protocol"][config["task"]]
            record = {key: config[key] for key in ("run_id", "created_at_utc", "task", "model_name", "seed", "hypothesis", "notes")}
            record.update({f"{key}_json": json.dumps(config[key], sort_keys=True) for key in ("feature_spec", "hyperparameters", "thresholds")})
            record.update(validation_rows=metrics["summary"]["n_rows"], primary_metric=primary["primary_metric"],
                          primary_value=metrics["summary"][primary["primary_metric"]], higher_is_better=primary["higher_is_better"],
                          run_directory=str(folder.relative_to(self.root)))
            records.append(record)
        registry = pd.DataFrame(records, columns=REGISTRY_COLUMNS)
        temporary = self.directory / f".experiment_log_{uuid.uuid4().hex}.tmp"
        registry.to_csv(temporary, index=False)
        temporary.replace(self.directory / "experiment_log.csv")
        return registry
