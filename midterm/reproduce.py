"""Verify both frozen submissions and the best public Part 1 and Part 2 files.

This runner uses saved fitted models and the supplied test features. It never
fits a model, uses hidden labels, or changes the recorded submission files.
After verification, --output-dir exports byte-identical best submissions as
part1.csv and part2.csv into a new or empty directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'tools/src'))

from midterm_evaluation import REASONS, dataset_path, part1_decisions
from midterm_finalization import predict_final_outcomes
from midterm_part1_finalization import predict_bundle
from midterm_part2_improvement import validate_submission
from midterm_rank_part2 import blend, predict_phase_shared


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def check_part1(root: Path) -> dict:
    source = dataset_path(root, "part1", "test")
    test = pd.read_csv(source)
    bundle = joblib.load(root / 'models/part1/02_kmeans_lr/model.joblib')
    probabilities = predict_bundle(bundle, test)
    selected = part1_decisions(probabilities, bundle["thresholds"], empty_set_policy="argmax")
    predicted = [" ".join(reason for reason, active in zip(REASONS, row) if active)
                 for row in selected]
    generated = pd.DataFrame({"ID": test.ID, "Predicted": predicted})
    recorded = pd.read_csv(root / 'models/part1/02_kmeans_lr/submission.csv')
    sample = pd.read_csv(source.parent / "part1_sample_submission.csv")
    validate_submission(generated, sample, 1)
    if not generated.equals(recorded):
        raise AssertionError("Part 1 selected model differs from recorded submission.")
    return {"rows": len(generated), "model": 'models/part1/02_kmeans_lr/model.joblib',
            "submission": 'models/part1/02_kmeans_lr/submission.csv',
            "submission_sha256": digest(root / 'models/part1/02_kmeans_lr/submission.csv')}


def check_part1_best_public(root: Path) -> dict:
    """Recreate the current local-kernel submission from its saved fitted bundle."""
    selection = json.loads((root / 'models/selection.json').read_text())
    record = selection["part1"]["best_public"]
    source = dataset_path(root, "part1", "test")
    test = pd.read_csv(source)
    fitted = joblib.load(root / record["model"])
    probabilities = fitted.predict(test[[f"emb_{j}" for j in range(150)]].to_numpy())
    if probabilities.shape != (len(test), len(REASONS)) or not np.isfinite(probabilities).all():
        raise AssertionError("Best public Part 1 model returned invalid probabilities.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise AssertionError("Best public Part 1 probabilities are outside [0, 1].")
    selected = part1_decisions(probabilities, fitted.recipes["final"]["threshold"],
                               empty_set_policy="argmax")
    generated = pd.DataFrame({
        "ID": test.ID,
        "Predicted": [" ".join(reason for reason, active in zip(REASONS, row) if active)
                      for row in selected],
    })
    sample = pd.read_csv(source.parent / "part1_sample_submission.csv")
    validate_submission(generated, sample, 1)
    recorded = pd.read_csv(root / record["submission"])
    if not generated.equals(recorded):
        raise AssertionError("Best public Part 1 model differs from its recorded submission.")
    return {"rows": len(generated), "model": record["model"],
            "submission": record["submission"],
            "submission_sha256": digest(root / record["submission"])}


def check_part2(root: Path) -> dict:
    source = dataset_path(root, "part2", "test")
    test = pd.read_csv(source)
    original = joblib.load(root / 'models/part2/02_text_blend/model.joblib')
    shared = joblib.load(root / 'models/part2/03_phase_shared/component.joblib')
    original_p = predict_final_outcomes(original, test)
    shared_p = predict_phase_shared(shared, test)
    blended = blend(test, original_p, shared_p, 0.5)
    sample = pd.read_csv(source.parent / "part2_sample_submission.csv")
    results = {}
    for label, computed, relative in (
        ("submitted_original", original_p, 'models/part2/02_text_blend/submission.csv'),
        ("best_public", blended, 'models/part2/03_phase_shared/submission.csv'),
    ):
        generated = computed.rename(columns={"p_success": "label"})
        recorded = pd.read_csv(root / relative)
        validate_submission(generated, sample, 2)
        if generated.ID.tolist() != recorded.ID.tolist():
            raise AssertionError(f"Part 2 {label} ID order differs.")
        gap = float(np.max(np.abs(generated.label.to_numpy() - recorded.label.to_numpy())))
        if gap > 1e-12:
            raise AssertionError(f"Part 2 {label} probabilities differ: max gap {gap}.")
        results[label] = {"rows": len(generated), "submission": relative,
                          "submission_sha256": digest(root / relative),
                          "max_absolute_probability_gap": gap}
    return results


def submission_records(value):
    """Visit every registered submission, including prior public selections."""
    if isinstance(value, dict):
        if "submission" in value:
            if "sha256" not in value:
                raise AssertionError(f"Registered submission has no checksum: {value['submission']}")
            yield value
        for child in value.values():
            yield from submission_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from submission_records(child)


def check_registry(root: Path) -> dict:
    selection = json.loads((root / 'models/selection.json').read_text())
    # Keep the original registry checks mandatory when registering a new best.
    for task, label in (("part1", "best_public"), ("part1", "previous_best_public"),
                        ("part1", "rejected_uploaded_graph"), ("part2", "best_public"),
                        ("part2", "frozen_november_1_delivery"),
                        ("part2", "earlier_word_field_candidate")):
        if not isinstance(selection.get(task, {}).get(label), dict):
            raise AssertionError(f"Missing required submission record: {task}.{label}")
    records = list(submission_records(selection))
    if not records:
        raise AssertionError("Submission registry is empty.")
    for record in records:
        if digest(root / record["submission"]) != record["sha256"]:
            raise AssertionError(f"Recorded submission bytes changed: {record['submission']}")
        for model, expected in record.get("model_sha256", {}).items():
            if digest(root / model) != expected:
                raise AssertionError(f"Recorded model bytes changed: {model}")
    source = root / 'models/part2/03_phase_shared/submission.csv'
    if digest(source) != selection["part2"]["best_public"]["sha256"]:
        raise AssertionError("Canonical best public Part 2 file does not match its registered hash.")
    return {"status": "PASS", "recorded_submissions": len(records)}


def validate_output_directory(path: Path) -> None:
    """Reject files and nonempty directories before running inference or writing."""
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(f"Output directory must be new or empty: {path}")


def export_best_submissions(root: Path, output_dir: Path) -> dict:
    """Export canonical bytes after main has verified their saved-model predictions.

    Copying the checked canonical CSV bytes preserves the exact submitted float
    serialization and registered hashes. No model or recorded CSV is rewritten.
    """
    validate_output_directory(output_dir)
    selection = json.loads((root / "models/selection.json").read_text())
    payloads = {}
    for task in ("part1", "part2"):
        record = selection[task]["best_public"]
        contents = (root / record["submission"]).read_bytes()
        if hashlib.sha256(contents).hexdigest() != record["sha256"]:
            raise AssertionError(f"Recorded submission bytes changed: {record['submission']}")
        payloads[task] = contents
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_output_directory(output_dir)
    exported = {}
    for task, contents in payloads.items():
        path = output_dir / f"{task}.csv"
        with path.open("xb") as destination:
            destination.write(contents)
        exported[task] = {"filename": path.name, "sha256": hashlib.sha256(contents).hexdigest()}
    return exported


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path,
                        help="Optional new or empty directory for verified part1.csv and part2.csv.")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else None
    if output_dir is not None:
        validate_output_directory(output_dir)
    registry = check_registry(root)
    report = {"part1": check_part1(root), "part1_best_public": check_part1_best_public(root),
              "part2": check_part2(root),
              "registry": registry, "status": "PASS"}
    if output_dir is not None:
        report["exported_best"] = export_best_submissions(root, output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
