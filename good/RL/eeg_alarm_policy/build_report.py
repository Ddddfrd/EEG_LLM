"""Build the reproducible R1-R4 alarm-policy result report and figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .artifacts import load_prediction_artifact
from .evaluator import _alarm_episodes

METHOD_ORDER = (
    "inherited_rule",
    "robust_fixed_rule",
    "logistic_regression",
    "mlp_32x32",
)
METHOD_LABELS = {
    "inherited_rule": "Inherited rule",
    "robust_fixed_rule": "Robust fixed rule",
    "logistic_regression": "Logistic",
    "mlp_32x32": "MLP 32x32",
}
METHOD_COLORS = {
    "inherited_rule": "#777777",
    "robust_fixed_rule": "#2563A6",
    "logistic_regression": "#D38A1F",
    "mlp_32x32": "#A23B72",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#252525",
            "axes.titlecolor": "#171717",
            "axes.grid": True,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.dpi": 180,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def _plot_validation_pareto(r1: dict[str, Any], output: Path) -> None:
    rows = r1["rows"]
    pareto = r1["pareto_frontier"]
    selected = r1["selected"]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.scatter(
        [row["pooled_event_metrics"]["false_alarms_per_hour"] for row in rows],
        [row["pooled_event_metrics"]["event_sensitivity"] for row in rows],
        s=12,
        alpha=0.15,
        color="#777777",
        edgecolors="none",
        label=f"Grid candidates (n={len(rows)})",
    )
    pareto_x = [
        row["pooled_event_metrics"]["false_alarms_per_hour"] for row in pareto
    ]
    pareto_y = [row["pooled_event_metrics"]["event_sensitivity"] for row in pareto]
    order = np.argsort(pareto_x)
    ax.plot(
        np.asarray(pareto_x)[order],
        np.asarray(pareto_y)[order],
        color="#D38A1F",
        marker="o",
        markersize=3.5,
        linewidth=1.4,
        label=f"Pareto frontier (n={len(pareto)})",
    )
    x = selected["pooled_event_metrics"]["false_alarms_per_hour"]
    y = selected["pooled_event_metrics"]["event_sensitivity"]
    ax.scatter(
        [x],
        [y],
        s=100,
        marker="*",
        color="#2563A6",
        edgecolor="#171717",
        linewidth=0.7,
        zorder=5,
        label="Frozen robust rule",
    )
    ax.annotate(
        f"  sensitivity {y:.2f}\n  FA/h {x:.3f}",
        (x, y),
        xytext=(8, -38),
        textcoords="offset points",
        fontsize=9,
    )
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("False alarm episodes per normal monitoring hour (symlog)")
    ax.set_ylabel("Event sensitivity")
    ax.set_title("R1 deterministic-rule validation frontier")
    ax.legend(loc="lower right")
    _save(fig, output)


def _plot_final_tradeoff(r4: dict[str, Any], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    for method in METHOD_ORDER:
        evaluation = r4["methods"][method]["evaluation"]
        pooled = evaluation["pooled"]
        x = pooled["false_alarms_per_hour"]
        y = pooled["mean_detection_latency_seconds"]
        sensitivity = pooled["event_sensitivity"]
        marker = "o" if evaluation["guardrail_pass"] else "X"
        ax.scatter(
            [x],
            [y],
            s=100,
            marker=marker,
            color=METHOD_COLORS[method],
            edgecolor="#171717",
            linewidth=0.7,
            label=METHOD_LABELS[method],
            zorder=3,
        )
        ax.annotate(
            f"{METHOD_LABELS[method]}\n{x:.3f} FA/h, {sensitivity:.2f} sens.",
            (x, y),
            xytext=(7, 7 if method != "logistic_regression" else -30),
            textcoords="offset points",
            fontsize=8.5,
        )
    ax.set_xlim(0, 1.85)
    ax.set_ylim(0, 11.2)
    ax.set_xlabel("False alarm episodes per normal monitoring hour")
    ax.set_ylabel("Mean detection latency (seconds)")
    ax.set_title("Frozen latency-false-alarm tradeoff on chb22-chb23")
    ax.legend(loc="lower right")
    _save(fig, output)


def _plot_patient_comparison(r4: dict[str, Any], output: Path) -> None:
    subjects = r4["subjects"]
    x = np.arange(len(subjects), dtype=float)
    width = 0.18
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.0), sharex=True)
    for index, method in enumerate(METHOD_ORDER):
        offset = (index - 1.5) * width
        per_subject = r4["methods"][method]["evaluation"]["per_subject"]
        sensitivities = [
            per_subject[subject]["event_metrics"]["event_sensitivity"]
            for subject in subjects
        ]
        false_alarms = [
            per_subject[subject]["event_metrics"]["false_alarms_per_hour"]
            for subject in subjects
        ]
        axes[0].bar(
            x + offset,
            sensitivities,
            width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        axes[1].bar(
            x + offset,
            false_alarms,
            width,
            color=METHOD_COLORS[method],
        )
    axes[0].axhline(0.8, color="#333333", linestyle="--", linewidth=1.0)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Event sensitivity")
    axes[0].set_title("Per-patient event sensitivity and false-alarm burden")
    axes[0].legend(
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8.5,
    )
    axes[1].set_ylabel("False alarms / hour")
    axes[1].set_xticks(x, subjects)
    axes[1].set_xlabel("Held-out patient")
    fig.subplots_adjust(right=0.79)
    _save(fig, output)


def _plot_ppo_stability(r3: dict[str, Any], output: Path) -> None:
    runs = r3["ppo"]["results"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7))
    for run in runs:
        history = run["history"]
        axes[0].plot(
            [row["epoch"] for row in history],
            [row["mean_episode_return"] for row in history],
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=f"seed {run['seed']}",
        )
    axes[0].set_xlabel("PPO epoch")
    axes[0].set_ylabel("Mean training episode return")
    axes[0].set_title("PPO training trajectories")
    axes[0].legend(ncol=2, fontsize=8)

    seeds = [str(run["seed"]) for run in runs]
    scores = [run["j_score"] for run in runs]
    colors = ["#2563A6" if run["guardrail_pass"] else "#B9BDC5" for run in runs]
    axes[1].bar(seeds, scores, color=colors, edgecolor="#333333", linewidth=0.5)
    axes[1].axhline(
        r3["development_comparators"]["robust_fixed_rule"]["j_score"],
        color="#D38A1F",
        linestyle="--",
        linewidth=1.3,
        label="Robust fixed rule",
    )
    axes[1].set_xlabel("PPO random seed")
    axes[1].set_ylabel("Validation J")
    axes[1].set_title("PPO seed dispersion on chb21")
    axes[1].legend(fontsize=8)
    _save(fig, output)


def _single_artifact(directory: Path, subject: str):
    matches = sorted(directory.glob(f"predictions_{subject}_*.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one prediction artifact for {subject}")
    return load_prediction_artifact(matches[0])


def _plot_representative_timeline(
    r4: dict[str, Any],
    artifact_dir: Path,
    output: Path,
) -> None:
    subject = "chb22"
    artifact = _single_artifact(artifact_dir, subject)
    timeline = artifact.timeline
    event = next(value for value in timeline.events if "chb22_25.edf" in value.event_id)
    record_mask = timeline.record_indices == event.record_index
    seconds = timeline.start_samples / timeline.sampling_frequency_hz
    view = record_mask & (seconds >= event.start_seconds - 360) & (
        seconds <= event.end_seconds + 120
    )
    relative_minutes = (seconds[view] - event.start_seconds) / 60.0
    robust_all = np.load(
        r4["methods"]["robust_fixed_rule"]["actions"][subject]["path"],
        allow_pickle=False,
    )
    mlp_all = np.load(
        r4["methods"]["mlp_32x32"]["actions"][subject]["path"],
        allow_pickle=False,
    )
    robust_actions = robust_all[view]
    _, robust_accepted = _alarm_episodes(
        timeline,
        robust_all,
        refractory_seconds=r4["methods"]["robust_fixed_rule"]["rule"][
            "refractory_seconds"
        ],
    )
    _, mlp_accepted = _alarm_episodes(
        timeline,
        mlp_all,
        refractory_seconds=r4["methods"]["mlp_32x32"]["rule"][
            "refractory_seconds"
        ],
    )

    def accepted_minutes(episodes) -> np.ndarray:
        return np.asarray(
            [
                (alarm.start_seconds - event.start_seconds) / 60.0
                for alarm in episodes
                if alarm.record_index == event.record_index
                and event.start_seconds - 360
                <= alarm.start_seconds
                <= event.end_seconds + 120
            ],
            dtype=float,
        )

    robust_accepted_minutes = accepted_minutes(robust_accepted)
    mlp_accepted_minutes = accepted_minutes(mlp_accepted)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(
        relative_minutes,
        timeline.probabilities[view],
        color="#2563A6",
        linewidth=1.1,
        label="Frozen base probability",
    )
    event_end = (event.end_seconds - event.start_seconds) / 60.0
    ax.axvspan(0, event_end, color="#D38A1F", alpha=0.22, label="Labeled seizure")
    ax.scatter(
        relative_minutes[robust_actions == 1],
        np.full(int(robust_actions.sum()), 1.02),
        marker="|",
        s=60,
        color="#8CB3D9",
        label="Robust voted actions (pre-refractory)",
        clip_on=False,
    )
    ax.scatter(
        robust_accepted_minutes,
        np.full(robust_accepted_minutes.size, 1.07),
        marker="x",
        s=45,
        color=METHOD_COLORS["robust_fixed_rule"],
        label="Robust accepted alarms",
        clip_on=False,
    )
    ax.scatter(
        mlp_accepted_minutes,
        np.full(mlp_accepted_minutes.size, 1.11),
        marker="v",
        s=40,
        color=METHOD_COLORS["mlp_32x32"],
        label="MLP accepted alarms",
        clip_on=False,
    )
    for index, alarm_time in enumerate(robust_accepted_minutes):
        ax.axvspan(
            alarm_time,
            alarm_time + 5.0,
            color=METHOD_COLORS["robust_fixed_rule"],
            alpha=0.07,
            label="Robust 300 s refractory" if index == 0 else None,
        )
    ax.axhline(0.9, color="#333333", linestyle="--", linewidth=1.0, label="Rule threshold")
    ax.set_ylim(-0.02, 1.16)
    ax.set_xlim(float(relative_minutes.min()), float(relative_minutes.max()))
    ax.set_xlabel("Minutes relative to seizure onset")
    ax.set_ylabel("Ictal probability")
    ax.set_title("chb22_25: prior robust alarm suppresses seizure-period candidates")
    ax.legend(ncol=2, loc="upper left", fontsize=8.2)
    _save(fig, output)


def _metric_row(name: str, payload: dict[str, Any]) -> str:
    evaluation = payload["evaluation"]
    pooled = evaluation["pooled"]
    ranking = payload.get("ranking_metrics", {})
    auroc = f"{ranking['auroc']:.4f}" if "auroc" in ranking else "n/a"
    auprc = f"{ranking['auprc']:.4f}" if "auprc" in ranking else "n/a"
    return (
        f"| {name} | {auroc} | {auprc} | "
        f"{pooled['detected_events']}/{pooled['event_count']} | "
        f"{pooled['event_sensitivity']:.3f} | "
        f"{pooled['false_alarms_per_hour']:.3f} | "
        f"{pooled['mean_detection_latency_seconds']:.2f} | "
        f"{pooled['action_metrics']['f1']:.3f} | "
        f"{evaluation['j_score']:.4f} | "
        f"{'yes' if evaluation['guardrail_pass'] else 'no'} |"
    )


def _report_text(
    r1: dict[str, Any],
    r2: dict[str, Any],
    r3: dict[str, Any],
    r4: dict[str, Any],
) -> str:
    final_rows = [
        _metric_row(METHOD_LABELS[method], r4["methods"][method])
        for method in METHOD_ORDER
    ]
    patient_rows: list[str] = []
    for subject in r4["subjects"]:
        for method in METHOD_ORDER:
            metrics = r4["methods"][method]["evaluation"]["per_subject"][subject][
                "event_metrics"
            ]
            patient_rows.append(
                f"| {subject} | {METHOD_LABELS[method]} | "
                f"{metrics['detected_events']}/{metrics['event_count']} | "
                f"{metrics['event_sensitivity']:.3f} | "
                f"{metrics['false_alarms_per_hour']:.3f} | "
                f"{metrics['mean_detection_latency_seconds']:.2f} |"
            )

    robust_validation = r1["selected"]
    logistic_validation = r2["results"]["logistic_regression"]["selected"]
    mlp_validation = r2["results"]["mlp_32x32"]["selected"]
    selected_ppo = r3["ppo"]["selected"]
    return "\n".join(
        [
            "# EEG Alarm-Policy R1-R4 Evaluation Report",
            "",
            "## Technical summary",
            "",
            "The frozen Scheme C S1 base probabilities remain highly rank-discriminative on "
            f"chb22-chb23 (AUROC **{r4['base_probability_metrics']['auroc']:.4f}**, "
            f"AUPRC **{r4['base_probability_metrics']['auprc']:.4f}**). The primary robust "
            "fixed rule reduced false alarms to **0.471/hour**, but detected 9/10 events and "
            "failed the predeclared per-patient sensitivity guardrail because chb22 was 2/3.",
            "",
            "The frozen 32x32 MLP comparator detected **10/10 events** at **0.837 false "
            "alarms/hour** and passed the pooled and per-patient guardrail. The inherited rule "
            "also passed, but detected 9/10 events at 1.725 false alarms/hour. This is evidence "
            "that recent probability-trajectory shape can improve alarm decisions. It is not "
            "an independently confirmed promotion: the MLP became the apparent winner after "
            "observing the single final cohort.",
            "",
            "PPO was not evaluated on chb22-chb23 because it failed its frozen promotion gate. "
            f"Its median-seed policy achieved validation J={selected_ppo['j_score']:.4f} and "
            f"event sensitivity={selected_ppo['pooled']['event_sensitivity']:.2f}; the robust "
            "fixed rule remained stronger and substantially more stable. RL is therefore not "
            "justified by this experiment.",
            "",
            "## Fixed rules expose a sensitivity-false-alarm frontier",
            "",
            "![R1 validation Pareto frontier](results/validation_pareto.png)",
            "",
            "R1 searched deterministic threshold, voting, and refractory controls jointly on "
            "chb20-chb21. The selected rule (`threshold=0.90`, `2-of-5`, `300 s`) detected "
            f"{robust_validation['pooled_event_metrics']['detected_events']}/"
            f"{robust_validation['pooled_event_metrics']['event_count']} validation events at "
            f"{robust_validation['pooled_event_metrics']['false_alarms_per_hour']:.3f} false "
            "alarms/hour. The patient-level guardrail was added specifically to reject rules "
            "whose pooled score hides a weak patient.",
            "",
            "## Frozen final comparison favors the compact MLP",
            "",
            "![Final alarm-policy tradeoff](results/final_tradeoff.png)",
            "",
            "Lower-left is preferred in this latency-false-alarm view; point labels retain "
            "event sensitivity, and X markers indicate a failed sensitivity guardrail. The "
            "robust rule has the lowest false-alarm burden, but its missed chb22 event causes "
            "the guardrail failure. The inherited 60-second rule alarms much more often. "
            "The logistic control also misses a chb22 event. The MLP trades 0.366 additional "
            "false alarms/hour versus the robust rule for complete event detection in this "
            "cohort.",
            "",
            "| Method | AUROC | AUPRC | Events | Event sens. | FA/h | Latency s | "
            "Window F1 | J | Guardrail |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
            *final_rows,
            "",
            "AUROC/AUPRC are reported only for learned score-producing controls. The fixed-rule "
            "rows operate directly on the unchanged base probabilities; their ranking metrics "
            "are the base AUROC/AUPRC above rather than new model scores.",
            "",
            "## Patient-level results reveal the chb22 failure mode",
            "",
            "![Per-patient comparison](results/final_per_patient.png)",
            "",
            "| Patient | Method | Events | Event sens. | FA/h | Latency s |",
            "|---|---|---:|---:|---:|---:|",
            *patient_rows,
            "",
            "The final cohort contains only two patients and ten seizures, so one missed chb22 "
            "event moves that patient's sensitivity from 1.00 to 0.67. The apparent MLP "
            "advantage is meaningful for this cohort but has wide sampling uncertainty.",
            "",
            "## PPO is unstable and does not beat simpler controls",
            "",
            "![PPO seed stability](results/ppo_seed_stability.png)",
            "",
            "On chb21, the robust fixed rule reached "
            f"J={r3['development_comparators']['robust_fixed_rule']['j_score']:.4f}; "
            f"logistic reached J={logistic_validation['j_score']:.4f}; MLP reached "
            f"J={mlp_validation['j_score']:.4f}. PPO results varied sharply by seed, and the "
            "predeclared median-J seed failed the sensitivity guardrail. A tabular Q policy "
            "worked better than selected PPO but still did not beat the robust rule.",
            "",
            "## A missed event shows why temporal shape matters",
            "",
            "![Representative chb22 timeline](results/chb22_missed_event_timeline.png)",
            "",
            "For chb22_25, the base probability and voted actions rise around the labeled "
            "event, but a prior accepted robust alarm places the event inside the 300-second "
            "refractory interval. The MLP combines eight causal probability "
            "samples with enrollment distribution summaries, slope, variance, and a record-start "
            "flag. It emits an alarm without using future samples or labels as inputs.",
            "",
            "## Scope and metric definitions",
            "",
            "- Frozen base model: Qwen2.5-0.5B `visual_mean`, E1+E2+E3+E4, STFT 128/128/32.",
            "- Policy development: chb20 for supervised/RL fitting; chb21 for "
            "supervised/RL selection; "
            "R1 robust grid used chb20-chb21 jointly.",
            "- Final evaluation: chb22-chb23, exported once after the protocol freeze.",
            "- Event sensitivity: detected labeled seizure events divided by all labeled events.",
            "- False alarms/hour: alarm episodes overlapping no seizure divided by normal "
            "monitoring hours.",
            "- Latency: seconds from seizure onset to the first overlapping accepted alarm.",
            "- Guardrail: pooled event sensitivity >=0.80 and every patient event "
            "sensitivity >=0.80.",
            "- J: event sensitivity - 0.02 x FA/hour - 0.001 x normalized latency.",
            "",
            "## Experimental controls and audit trail",
            "",
            "The test export was locked until `r4_protocol_freeze_c24812a0fede.json` existed. "
            "The freeze fixed all rule thresholds, supervised checkpoints, objective weights, "
            "seed policy, and the exclusion of PPO. Final evaluation reloaded checkpoint SHA256 "
            "values and prediction artifact IDs and did not perform threshold search. The exported "
            "base metrics reproduce the authoritative S1 result exactly.",
            "",
            "## Limitations",
            "",
            "- The final sample is two patients and ten events; it is insufficient for a "
            "clinical claim.",
            "- chb01 and chb21 share a subject identity, so this remains the fast Tier A "
            "benchmark.",
            "- R1 used chb20-chb21 jointly, whereas supervised and RL methods fit chb20 and "
            "selected on chb21.",
            "- The MLP result is a frozen comparator result, but selecting it now would be "
            "post-test selection.",
            "- The alarm reward and J weights encode one operating preference, not a "
            "validated clinical utility.",
            "",
            "## Recommended next steps",
            "",
            "1. Freeze the MLP as the next candidate and confirm it on a new untouched "
            "patient cohort or dataset.",
            "2. Run Tier B with grouped chb01/chb21 and out-of-fold base probabilities for "
            "policy fitting.",
            "3. Add bootstrap confidence intervals over patients and seizure events; do not "
            "rely on point estimates alone.",
            "4. Diagnose chb22_25 and compare MLP calibration against a non-neural temporal "
            "model with monotonic constraints.",
            "5. Resume RL only after adding more policy-training patients and require it to "
            "beat both MLP and fixed-rule frontiers.",
            "",
            "## Further questions",
            "",
            "- Does the MLP advantage persist when every policy-training probability is "
            "out-of-fold?",
            "- Which of slope, variance, enrollment quantiles, and raw history produces the "
            "chb22 gain?",
            "- Can a calibrated sequence model retain 10/10 detection while reducing the "
            "MLP's 0.837 FA/hour?",
            "",
            "## Result identities",
            "",
            f"- R1 result SHA256: `{r1['result_sha256']}`",
            f"- R2 result SHA256: `{r2['result_sha256']}`",
            f"- R3 result SHA256: `{r3['result_sha256']}`",
            f"- R4 result SHA256: `{r4['result_sha256']}`",
            f"- Final chb22 artifact: `{r4['prediction_artifact_ids']['chb22']}`",
            f"- Final chb23 artifact: `{r4['prediction_artifact_ids']['chb23']}`",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1", type=Path, required=True)
    parser.add_argument("--r2", type=Path, required=True)
    parser.add_argument("--r3", type=Path, required=True)
    parser.add_argument("--r4", type=Path, required=True)
    parser.add_argument("--test-artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("EEG_RL_ALARM_POLICY_RESULTS.md"),
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    r1, r2, r3, r4 = (_load(path) for path in (args.r1, args.r2, args.r3, args.r4))
    _style()
    _plot_validation_pareto(r1, args.output_dir / "validation_pareto.png")
    _plot_final_tradeoff(r4, args.output_dir / "final_tradeoff.png")
    _plot_patient_comparison(r4, args.output_dir / "final_per_patient.png")
    _plot_ppo_stability(r3, args.output_dir / "ppo_seed_stability.png")
    _plot_representative_timeline(
        r4,
        args.test_artifact_dir,
        args.output_dir / "chb22_missed_event_timeline.png",
    )
    args.report.write_text(_report_text(r1, r2, r3, r4), encoding="utf-8")
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
