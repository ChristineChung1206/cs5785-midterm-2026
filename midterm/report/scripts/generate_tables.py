"""Render report tables from saved, audited metrics; no fitting or inference."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "analysis/report_metrics.json"

def main():
    data = json.loads(METRICS.read_text())
    out = ROOT / "tables"
    out.mkdir(exist_ok=True)
    names = {"supervised_lr": "Supervised LR", "initial_kmeans_lr": "Initial K-Means + LR",
             "retained_kmeans_lr": "Retained K-Means + LR", "fixed_ridge_157_gold": "Fixed ridge recipe (157 gold)"}
    reasons = ["Safety", "Funding", "Efficacy", "Recruitment", "Other"]
    row_end = r" \\" + "\n"
    tab = r"\begin{tabular}{lrrrrrrr}" + "\n" + r"\toprule" + "\n"
    tab += "Model & Sample F1 & Macro F1 & Safety & Funding & Efficacy & Recruit. & Other" + row_end
    tab += r"\midrule" + "\n"
    for row in data["part1_validation"]:
        per = {r["reason"]: r["f1"] for r in data["part1_validation_by_reason"] if r["model"] == row["model"]}
        values = [row["f1_samples"], row["f1_macro"], *[per[r] for r in reasons]]
        tab += names[row["model"]] + " & " + " & ".join(f"{v:.4f}" for v in values) + row_end
    tab += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"
    (out / "part1_validation.tex").write_text(tab)

    tab = r"\begin{tabular}{llrrrr}" + "\n" + r"\toprule" + "\n"
    tab += "Model & Phase & F1 & ROC-AUC & AP & Log loss" + row_end + r"\midrule" + "\n"
    for row in data["part2_validation_by_phase"]:
        name = "Baseline" if row["model"] == "baseline" else "Current phase-shared blend"
        values = [row[k] for k in ("f1", "roc_auc", "average_precision", "log_loss")]
        tab += name + " & " + row["phase"] + " & " + " & ".join(f"{v:.4f}" for v in values) + row_end
    tab += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"
    (out / "part2_phase.tex").write_text(tab)

    tab = r"\begin{tabular}{llrrll}" + "\n" + r"\toprule" + "\n"
    tab += "Phase & Reason group & $n$ & Obs./Pred. & Baseline errors & Current blend errors" + row_end + r"\midrule" + "\n"
    rows = [r for r in data["part2_failure_reason_comparison"] if r["reason_group"] != "matched_no_predicted_reason"]
    previous = None
    for row in rows:
        if previous is not None and row["phase"] != previous: tab += r"\midrule" + "\n"
        previous = row["phase"]
        n = row["n_trials"]
        reason = "No recorded reason" if row["reason_group"] == "no_recorded_stopping_reason" else row["reason_group"]
        origin = "--" if row["n_unmatched"] else f"{row['n_observed_reasons']}/{row['n_predicted_reasons']}"
        rates = []
        for model in ("baseline", "best"):
            count = row[f"n_predicted_success_{model}"]
            rates.append("--" if n == 0 else f"{count}/{n} ({100*count/n:.1f}" + r"\%)")
        tab += f"{row['phase']} & {reason} & {n} & {origin} & " + " & ".join(rates) + row_end
    tab += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"
    (out / "reason_errors.tex").write_text(tab)
    p = data["part2_reason_provenance"]
    (out / "reason_provenance_text.tex").write_text(
        f"The {p['failed_trial_phase_rows']} failed trial-phase rows represent {p['unique_failed_trials']} unique trials. "
        f"Of {p['matched_trial_phase_rows']} matched rows, {p['observed_reason_trial_phase_rows']} have observed reasons and "
        f"{p['predicted_reason_trial_phase_rows']} have predicted reasons; {p['unmatched_trial_phase_rows']} rows have no Part 1 match. "
        f"Among predicted rows, {p['predicted_training_resubstitution_rows']} are training resubstitution estimates. "
        "These are not out-of-fold or confirmed labels. Matched empty-reason groups have zero support "
        "after fallback and are omitted.\n")
    byphase = {(r["model"], r["phase"]): r for r in data["part2_validation_by_phase"]}
    fragments = []
    for phase in ("I", "II", "III"):
        a,b = byphase[("baseline",phase)],byphase[("phase_shared_best",phase)]
        fragments.append(f"Phase {phase} false-success counts change from {a['fp']}/{a['n_negative']} to {b['fp']}/{b['n_negative']}, "
                         f"while true-success false negatives change from {a['fn']} to {b['fn']}.")
    fragments.append("Phase I Recruitment errors fall from 10/22 to 7/22, while Safety errors rise from 4/10 to 5/10. "
                     "Phase II Other improves from 5/15 to 4/15, but Efficacy worsens from 4/18 to 5/18. "
                     "All five Phase I and all ten Phase III Efficacy failures remain false successes. "
                     "These groups include estimated reasons.")
    (out / "reason_interpretation_text.tex").write_text("\n\n".join(fragments) + "\n")

if __name__ == "__main__":
    main()
