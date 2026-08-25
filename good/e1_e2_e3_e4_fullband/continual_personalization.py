"""Personalize the fixed Scheme C model on chronological CHB-MIT events.

The protocol keeps the feature extractor frozen. Each target patient receives
an E2 baseline from a short known-normal enrollment session, then a private
classifier head is updated event by event with replay. All reported primary
metrics are computed on future rows that are disjoint from enrollment and
adaptation.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import platform
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ai.chbmit.cache import ChbmitWindowCache
from ai.chbmit.eeg_continual_eval_cache import NaturalFoldImageCache
from ai.chbmit.eeg_continual_pretrain import compute_subject_log_spectral_baselines
from ai.chbmit.eeg_continual_pretrain_model import ServerSTFTConfig
from ai.chbmit.eegvl_enrollment_calibration import (
    _earliest_contiguous_normal_run,
    _load_matching_timelines,
)
from ai.chbmit.eegvl_multibranch_model import (
    checkpoint_sha256,
    load_portable_multibranch_state_dict,
)
from ai.chbmit.eegvl_s1_data import S1PreprocessedCache
from ai.chbmit.eegvl_training import _autocast_context, select_precision
from ai.chbmit.index import canonical_hash
from ai.v2.lightweight_dataset import write_content_addressed_json
from ai.v2.metrics import evaluate, find_optimal_threshold_exact

from .model import build_model
from .train_19_vs_chb10_14 import TRAINING_SUBJECTS, VALIDATION_TEST_SUBJECTS


SCHEMA_VERSION = "scheme_c_patient_continual_v1"


@dataclass(frozen=True)
class ContinualPersonalizationConfig:
    seed: int = 42
    enrollment_windows: int = 128
    adaptation_events: int = 2
    normal_to_positive_ratio: float = 7.0 / 3.0
    replay_batch_size: int = 16
    epochs_per_experience: int = 12
    classifier_learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    anchor_strength: float = 1e-3
    minimum_calibration_recall: float = 0.60
    prediction_batch_size: int = 128

    def validate(self) -> None:
        if self.enrollment_windows < 1 or self.adaptation_events < 1:
            raise ValueError("Enrollment windows and adaptation events must be positive")
        if self.normal_to_positive_ratio <= 0:
            raise ValueError("normal_to_positive_ratio must be positive")
        if self.replay_batch_size < 1 or self.epochs_per_experience < 1:
            raise ValueError("Replay batch size and epochs must be positive")
        if self.classifier_learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Classifier optimizer settings are invalid")
        if self.anchor_strength < 0:
            raise ValueError("anchor_strength must be non-negative")
        if not 0 <= self.minimum_calibration_recall <= 1:
            raise ValueError("minimum_calibration_recall must be in [0, 1]")
        if self.prediction_batch_size < 1:
            raise ValueError("prediction_batch_size must be positive")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _evenly_spaced(indices: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64)
    if count >= len(values):
        return values.copy()
    positions = np.linspace(0, len(values) - 1, num=count, dtype=np.int64)
    return values[positions]


def build_patient_continual_partition(
    cache: NaturalFoldImageCache,
    timelines: Mapping[str, Any],
    *,
    config: ContinualPersonalizationConfig,
) -> dict[str, Any]:
    """Create enrollment, event experiences and future-only holdouts."""
    config.validate()
    patients: dict[str, Any] = {}
    for subject_value in cache.metadata["subject_order"]:
        subject = str(subject_value)
        timeline = timelines[subject]
        subject_slice = cache.subject_slice(subject)
        row_start = int(subject_slice.start or 0)
        row_end = int(subject_slice.stop or 0)
        labels = np.asarray(timeline.labels, dtype=np.uint8)
        events = np.asarray(timeline.event_indices, dtype=np.int64)
        if len(labels) != row_end - row_start:
            raise ValueError(f"Timeline/cache row mismatch for {subject}")
        if not np.array_equal(labels, np.asarray(cache.labels[subject_slice])):
            raise ValueError(f"Timeline/cache label mismatch for {subject}")

        enrollment_local = _earliest_contiguous_normal_run(
            timeline,
            window_count=config.enrollment_windows,
        )
        enrollment_end = int(enrollment_local[-1])
        eligible_positive = np.flatnonzero(
            (labels == 1) & (np.arange(len(labels)) > enrollment_end)
        )
        eligible_events = [
            int(value) for value in np.unique(events[eligible_positive]) if int(value) > 0
        ]
        if len(eligible_events) <= config.adaptation_events:
            raise ValueError(f"{subject} needs at least one future event after adaptation")
        selected_events = eligible_events[: config.adaptation_events]
        experiences: list[dict[str, Any]] = []
        cursor = enrollment_end + 1
        for experience_index, event_index in enumerate(selected_events, start=1):
            positive = np.flatnonzero((events == event_index) & (labels == 1))
            positive = positive[positive > enrollment_end]
            if not len(positive):
                raise ValueError(f"{subject} event {event_index} has no positive rows")
            normal_candidates = np.flatnonzero((labels[cursor : int(positive[0])] == 0)) + cursor
            normal_count = min(
                len(normal_candidates),
                int(math.ceil(len(positive) * config.normal_to_positive_ratio)),
            )
            if normal_count < 1:
                raise ValueError(f"{subject} event {event_index} has no prior normals")
            normal = _evenly_spaced(normal_candidates, normal_count)
            local_rows = np.sort(np.concatenate([positive, normal]))
            experiences.append(
                {
                    "experience": experience_index,
                    "event_index": event_index,
                    "local_rows": local_rows,
                    "global_rows": local_rows + row_start,
                    "positive_rows": positive,
                    "normal_rows": normal,
                    "positive_count": int(len(positive)),
                    "normal_count": int(len(normal)),
                    "row_sha256": _sha256_array(local_rows),
                }
            )
            cursor = int(positive[-1]) + 1

        window_seconds = float(timeline.metadata["window_config"]["window_seconds"])
        stride_seconds = float(timeline.metadata["window_config"]["stride_seconds"])
        overlap_guard = int(math.ceil(window_seconds / stride_seconds))
        last_adaptation_row = int(experiences[-1]["positive_rows"][-1])
        holdout_start = last_adaptation_row + overlap_guard
        holdout_local = np.arange(holdout_start, len(labels), dtype=np.int64)
        holdout_labels = labels[holdout_local]
        if set(holdout_labels.tolist()) != {0, 1}:
            raise ValueError(f"{subject} future holdout must contain both classes")

        enrollment_global = enrollment_local + row_start
        holdout_global = holdout_local + row_start
        adaptation_global = np.concatenate(
            [np.asarray(item["global_rows"], dtype=np.int64) for item in experiences]
        )
        if np.intersect1d(enrollment_global, adaptation_global).size:
            raise RuntimeError("Enrollment/adaptation overlap")
        if np.intersect1d(adaptation_global, holdout_global).size:
            raise RuntimeError("Adaptation/holdout overlap")
        patients[subject] = {
            "row_start": row_start,
            "row_end": row_end,
            "enrollment_local": enrollment_local,
            "enrollment_global": enrollment_global,
            "experiences": experiences,
            "adaptation_local": adaptation_global - row_start,
            "holdout_local": holdout_local,
            "holdout_global": holdout_global,
            "selected_event_indices": selected_events,
            "future_event_indices": [
                int(value) for value in np.unique(events[holdout_local]) if int(value) > 0
            ],
            "overlap_guard_windows": overlap_guard,
            "holdout_positive_count": int(np.sum(holdout_labels == 1)),
            "holdout_normal_count": int(np.sum(holdout_labels == 0)),
        }
    return {"patients": patients}


class _NaturalRowsDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        cache: NaturalFoldImageCache,
        rows: np.ndarray,
        baseline: np.ndarray,
    ) -> None:
        self.cache = cache
        self.rows = np.asarray(rows, dtype=np.int64)
        self.baseline = np.asarray(baseline, dtype=np.float32)

    def __len__(self) -> int:
        return int(len(self.rows))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = int(self.rows[index])
        image = np.asarray(self.cache.images[row], dtype=np.float32)
        return {
            "image": torch.from_numpy(image.copy()),
            "label": torch.tensor(int(self.cache.labels[row]), dtype=torch.long),
            "baseline_log_magnitude": torch.from_numpy(self.baseline),
        }


@torch.no_grad()
def extract_fused_representations(
    model: nn.Module,
    dataset: Dataset[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    precision = select_precision(device)
    model.eval()
    features: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        baseline = batch["baseline_log_magnitude"].to(device, non_blocking=True)
        with _autocast_context(device, precision):
            fused = model.forward_fused_representation(
                images,
                baseline_log_magnitude=baseline,
            )
            logits = model.classifier(fused)
        features.append(fused.float().cpu().numpy())
        probabilities.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
        labels.append(np.asarray(batch["label"], dtype=np.int64))
    return (
        np.concatenate(features).astype(np.float32, copy=False),
        np.concatenate(probabilities).astype(np.float32, copy=False),
        np.concatenate(labels).astype(np.int64, copy=False),
    )


def _balanced_epoch_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64)
    negative = np.flatnonzero(values == 0)
    positive = np.flatnonzero(values == 1)
    if not len(negative) or not len(positive):
        raise ValueError("Replay training requires both classes")
    target = max(len(negative), len(positive))
    negative_balanced = rng.choice(negative, size=target, replace=len(negative) < target)
    positive_balanced = rng.choice(positive, size=target, replace=len(positive) < target)
    order = np.concatenate([negative_balanced, positive_balanced])
    rng.shuffle(order)
    return order.astype(np.int64, copy=False)


def train_replay_classifier(
    base_classifier: nn.Linear,
    features: np.ndarray,
    labels: np.ndarray,
    experiences: Sequence[np.ndarray],
    *,
    config: ContinualPersonalizationConfig,
    device: torch.device,
) -> tuple[nn.Linear, list[dict[str, Any]], np.ndarray]:
    classifier = copy.deepcopy(base_classifier).to(device)
    anchor = {name: value.detach().clone() for name, value in classifier.named_parameters()}
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=config.classifier_learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(config.seed)
    replay_rows = np.empty(0, dtype=np.int64)
    history: list[dict[str, Any]] = []
    for experience_index, current_value in enumerate(experiences, start=1):
        current = np.asarray(current_value, dtype=np.int64)
        replay_before = replay_rows.copy()
        training_rows = np.unique(np.concatenate([replay_before, current]))
        x = torch.from_numpy(np.asarray(features[training_rows], dtype=np.float32)).to(device)
        y = torch.from_numpy(np.asarray(labels[training_rows], dtype=np.int64)).to(device)
        losses: list[float] = []
        classifier.train()
        for _ in range(config.epochs_per_experience):
            order = _balanced_epoch_indices(y.cpu().numpy(), rng)
            for start in range(0, len(order), config.replay_batch_size):
                selected = torch.from_numpy(order[start : start + config.replay_batch_size]).to(
                    device
                )
                logits = classifier(x[selected])
                loss = nn.functional.cross_entropy(logits, y[selected])
                if config.anchor_strength:
                    anchor_loss = sum(
                        (parameter - anchor[name]).square().mean()
                        for name, parameter in classifier.named_parameters()
                    )
                    loss = loss + config.anchor_strength * anchor_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
        replay_rows = np.unique(np.concatenate([replay_rows, current]))
        history.append(
            {
                "experience": experience_index,
                "current_rows": int(len(current)),
                "replay_rows_before": int(len(replay_before)),
                "memory_rows_after": int(len(replay_rows)),
                "positive_rows": int(np.sum(labels[current] == 1)),
                "normal_rows": int(np.sum(labels[current] == 0)),
                "mean_loss": float(np.mean(losses)),
                "optimizer_steps": len(losses),
            }
        )
    return classifier.cpu(), history, replay_rows


def _classifier_probabilities(classifier: nn.Linear, features: np.ndarray) -> np.ndarray:
    classifier.eval()
    with torch.no_grad():
        logits = classifier(torch.from_numpy(np.asarray(features, dtype=np.float32)))
        return torch.softmax(logits, dim=1)[:, 1].numpy().astype(np.float32)


def _load_scheme_c_model(
    artifact: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[nn.Module, Mapping[str, Any], Path]:
    checkpoint_info = artifact["checkpoints"]["auroc"]
    checkpoint = Path(str(checkpoint_info["path"])).resolve()
    if checkpoint_sha256(checkpoint) != str(checkpoint_info["sha256"]):
        raise ValueError("Scheme C checkpoint SHA256 mismatch")
    contract = artifact["model_contract"]
    stft = contract["e1"]["stft"]
    model = build_model(
        qwen_model_name=str(contract["qwen"]["model_name"]),
        local_files_only=True,
        pretrained_visual_encoder=True,
        stft_config_override=ServerSTFTConfig(
            **{key: stft[key] for key in ServerSTFTConfig.__dataclass_fields__}
        ),
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if model.contract() != dict(payload["model_contract"]):
        raise ValueError("Scheme C model contract changed")
    load_portable_multibranch_state_dict(model, payload["state_dict"])
    model.to(device).eval()
    return model, payload, checkpoint


def _compute_population_baseline(
    model: nn.Module,
    artifact: Mapping[str, Any],
    *,
    config: ContinualPersonalizationConfig,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    manifest = _load_json(Path(str(artifact["source"]["window_manifest"])))
    raw_cache = ChbmitWindowCache(Path(str(artifact["source"]["raw_cache"])))
    preprocessed = S1PreprocessedCache(Path(str(artifact["source"]["preprocessed_cache"])))
    normal_indices: dict[str, np.ndarray] = {}
    for subject in TRAINING_SUBJECTS:
        indices = [
            index
            for index, row in enumerate(manifest["windows"])
            if str(row["subject_id"]) == subject and int(raw_cache.labels[index]) == 0
        ][: config.enrollment_windows]
        if len(indices) < config.enrollment_windows:
            raise ValueError(f"Insufficient source normals for {subject}")
        normal_indices[subject] = np.asarray(indices, dtype=np.int64)
    baselines, summary = compute_subject_log_spectral_baselines(
        model.visual_encoder,
        images=preprocessed.images,
        normal_indices_by_subject=normal_indices,
        device=device,
    )
    population = np.mean(
        np.stack([baselines[subject] for subject in TRAINING_SUBJECTS]),
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)
    return population, {
        "definition": "unweighted mean of 128-window source-patient baselines",
        "source_subjects": list(TRAINING_SUBJECTS),
        "source_baselines": summary,
        "population_baseline_sha256": _sha256_array(population),
    }


def _public_partition(partition: Mapping[str, Any]) -> dict[str, Any]:
    patients: dict[str, Any] = {}
    for subject, values in partition["patients"].items():
        patients[subject] = {
            "enrollment_windows": int(len(values["enrollment_global"])),
            "selected_event_indices": list(values["selected_event_indices"]),
            "future_event_indices": list(values["future_event_indices"]),
            "overlap_guard_windows": int(values["overlap_guard_windows"]),
            "adaptation_windows": int(len(values["adaptation_local"])),
            "holdout_windows": int(len(values["holdout_local"])),
            "holdout_positive_count": int(values["holdout_positive_count"]),
            "holdout_normal_count": int(values["holdout_normal_count"]),
            "experiences": [
                {
                    key: item[key]
                    for key in (
                        "experience",
                        "event_index",
                        "positive_count",
                        "normal_count",
                        "row_sha256",
                    )
                }
                for item in values["experiences"]
            ],
            "enrollment_sha256": _sha256_array(values["enrollment_global"]),
            "adaptation_sha256": _sha256_array(values["adaptation_local"]),
            "holdout_sha256": _sha256_array(values["holdout_global"]),
        }
    return {"patients": patients}


def _compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in (
            "observation_count",
            "threshold",
            "sensitivity",
            "specificity",
            "precision",
            "f1",
            "auroc",
            "auprc",
            "false_alarms_per_hour",
            "confusion_matrix",
        )
    }


def _macro_patient_metrics(patients: Mapping[str, Any], condition: str) -> dict[str, float]:
    return {
        metric: float(
            np.mean(
                [
                    float(values[condition][metric])
                    for values in patients.values()
                    if values[condition][metric] is not None
                ]
            )
        )
        for metric in ("auroc", "auprc", "f1")
    }


def _write_report(result: Mapping[str, Any], destination: Path) -> None:
    aggregate = result["aggregate"]
    lines = [
        "# Scheme C Patient Continual Personalization Results",
        "",
        "The AUROC-best Scheme C checkpoint is frozen as the shared base model. ",
        "Each patient uses 128 contiguous known-normal windows for E2 enrollment, ",
        "then adapts a private classifier on the first two future seizure events.",
        "",
        "Macro patient metrics are primary after personalization because each patient ",
        "has a different classifier head. Pooled raw-probability metrics are retained ",
        "only as diagnostics; their score scales are not calibrated across patients.",
        "",
        "## Aggregate future-holdout metrics",
        "",
        "| Condition | Macro AUROC | Macro AUPRC | Macro F1 | Pooled AUROC (diagnostic) | Pooled AUPRC (diagnostic) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in ("b0_population", "b1_patient_baseline", "b2_replay"):
        pooled = aggregate[condition]["pooled_fixed_threshold"]
        macro = aggregate[condition]["macro_patient"]
        lines.append(
            f"| {condition} | {macro['auroc']:.4f} | {macro['auprc']:.4f} | "
            f"{macro['f1']:.4f} | {pooled['auroc']:.4f} | {pooled['auprc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Per-patient deltas (B2 replay minus B1 patient baseline)",
            "",
            "| Patient | Adapt positives | Holdout positives | AUROC delta | AUPRC delta | F1 delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for subject, values in result["patients"].items():
        delta = values["delta_b2_minus_b1"]
        lines.append(
            f"| {subject} | {values['adaptation_positive_count']} | "
            f"{values['holdout_positive_count']} | {delta['auroc']:+.4f} | "
            f"{delta['auprc']:+.4f} | {delta['f1']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            "- Per-patient future-holdout AUROC/AUPRC and their macro averages are primary.",
            "- Pooled B2 probabilities are diagnostic only because private heads have different score scales.",
            "- The fixed threshold came from the original Scheme C development run; a second personalized threshold is selected only from adaptation rows.",
            "- chb10-chb14 were used to select the original checkpoint epoch, so this is a paired personalization experiment, not a pristine external test.",
            "- Only the 896-to-2 classifier is updated; the shared EEG/Qwen feature extractor remains unchanged.",
            "",
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(
    *,
    artifact_path: Path,
    output_dir: Path,
    config: ContinualPersonalizationConfig | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    settings = config or ContinualPersonalizationConfig()
    settings.validate()
    selected_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if selected_device.type != "cuda":
        raise RuntimeError("Continual personalization requires CUDA")
    started = time.perf_counter()
    artifact_path = Path(artifact_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = _load_json(artifact_path)
    model, checkpoint_payload, checkpoint_path = _load_scheme_c_model(
        artifact,
        device=selected_device,
    )
    cache_path = Path(str(artifact["source"]["development_natural_cache"])).resolve()
    cache = NaturalFoldImageCache(cache_path)
    if tuple(cache.metadata["subject_order"]) != VALIDATION_TEST_SUBJECTS:
        raise ValueError("Natural cache subjects do not match chb10-chb14")
    timelines = _load_matching_timelines(cache_path, cache)
    partition = build_patient_continual_partition(cache, timelines, config=settings)
    population_baseline, population_summary = _compute_population_baseline(
        model,
        artifact,
        config=settings,
        device=selected_device,
    )
    enrollment_indices = {
        subject: np.asarray(values["enrollment_global"], dtype=np.int64)
        for subject, values in partition["patients"].items()
    }
    patient_baselines, patient_baseline_summary = compute_subject_log_spectral_baselines(
        model.visual_encoder,
        images=cache.images,
        normal_indices_by_subject=enrollment_indices,
        device=selected_device,
    )

    base_threshold = float(checkpoint_payload["selected_threshold"])
    patients: dict[str, Any] = {}
    pooled: dict[str, dict[str, list[np.ndarray]]] = {
        condition: {"labels": [], "probabilities": []}
        for condition in ("b0_population", "b1_patient_baseline", "b2_replay")
    }
    classifier_dir = output_dir / "patient_classifiers"
    classifier_dir.mkdir(parents=True, exist_ok=True)
    for patient_index, subject in enumerate(VALIDATION_TEST_SUBJECTS):
        values = partition["patients"][subject]
        rows = np.arange(values["row_start"], values["row_end"], dtype=np.int64)
        print(f"extracting {subject} population-baseline representations", flush=True)
        _, b0_probabilities, labels = extract_fused_representations(
            model,
            _NaturalRowsDataset(cache, rows, population_baseline),
            device=selected_device,
            batch_size=settings.prediction_batch_size,
        )
        print(f"extracting {subject} patient-baseline representations", flush=True)
        features, b1_probabilities, target_labels = extract_fused_representations(
            model,
            _NaturalRowsDataset(cache, rows, patient_baselines[subject]),
            device=selected_device,
            batch_size=settings.prediction_batch_size,
        )
        if not np.array_equal(labels, target_labels):
            raise RuntimeError(f"Baseline conditions changed labels for {subject}")
        experiences = [
            np.asarray(item["local_rows"], dtype=np.int64) for item in values["experiences"]
        ]
        patient_config = ContinualPersonalizationConfig(
            **{**settings.to_dict(), "seed": settings.seed + patient_index}
        )
        adapted_classifier, history, replay_rows = train_replay_classifier(
            model.classifier.cpu(),
            features,
            labels,
            experiences,
            config=patient_config,
            device=selected_device,
        )
        model.classifier.to(selected_device)
        b2_probabilities = _classifier_probabilities(adapted_classifier, features)
        adaptation_probabilities = b2_probabilities[replay_rows]
        personalized_threshold = find_optimal_threshold_exact(
            labels[replay_rows],
            adaptation_probabilities,
            min_recall=settings.minimum_calibration_recall,
        )
        holdout = np.asarray(values["holdout_local"], dtype=np.int64)
        patient_conditions: dict[str, Any] = {}
        for condition, probabilities in (
            ("b0_population", b0_probabilities),
            ("b1_patient_baseline", b1_probabilities),
            ("b2_replay", b2_probabilities),
        ):
            metrics = evaluate(
                labels[holdout],
                probabilities[holdout],
                threshold=base_threshold,
                print_report=False,
                sample_duration_seconds=4.0,
            )
            patient_conditions[condition] = _compact_metrics(metrics)
            pooled[condition]["labels"].append(labels[holdout])
            pooled[condition]["probabilities"].append(probabilities[holdout])
        personalized_metrics = evaluate(
            labels[holdout],
            b2_probabilities[holdout],
            threshold=personalized_threshold,
            print_report=False,
            sample_duration_seconds=4.0,
        )
        b1 = patient_conditions["b1_patient_baseline"]
        b2 = patient_conditions["b2_replay"]
        classifier_path = classifier_dir / f"{subject}_replay_head.pt"
        temporary = classifier_path.with_name(f".{classifier_path.name}.{uuid.uuid4().hex}.tmp")
        torch.save(
            {
                "schema_version": SCHEMA_VERSION,
                "subject": subject,
                "base_checkpoint_sha256": artifact["checkpoints"]["auroc"]["sha256"],
                "baseline_log_magnitude": patient_baselines[subject],
                "classifier_state_dict": adapted_classifier.state_dict(),
                "personalized_threshold": personalized_threshold,
                "config": patient_config.to_dict(),
            },
            temporary,
        )
        os.replace(temporary, classifier_path)
        patients[subject] = {
            **patient_conditions,
            "b2_personalized_threshold": _compact_metrics(personalized_metrics),
            "personalized_threshold": float(personalized_threshold),
            "adaptation_positive_count": int(np.sum(labels[replay_rows] == 1)),
            "adaptation_normal_count": int(np.sum(labels[replay_rows] == 0)),
            "holdout_positive_count": int(values["holdout_positive_count"]),
            "holdout_normal_count": int(values["holdout_normal_count"]),
            "continual_history": history,
            "classifier": {
                "path": str(classifier_path),
                "sha256": checkpoint_sha256(classifier_path),
            },
            "delta_b2_minus_b1": {
                metric: float(b2[metric]) - float(b1[metric])
                for metric in ("auroc", "auprc", "f1")
                if b2[metric] is not None and b1[metric] is not None
            },
        }
        print(
            f"{subject}: B1 AUROC={b1['auroc']:.4f} AUPRC={b1['auprc']:.4f}; "
            f"B2 AUROC={b2['auroc']:.4f} AUPRC={b2['auprc']:.4f}",
            flush=True,
        )
        del features, b0_probabilities, b1_probabilities, b2_probabilities
        gc.collect()

    aggregate: dict[str, Any] = {}
    for condition, values in pooled.items():
        labels = np.concatenate(values["labels"])
        probabilities = np.concatenate(values["probabilities"])
        aggregate[condition] = {
            "pooled_fixed_threshold": _compact_metrics(
                evaluate(
                    labels,
                    probabilities,
                    threshold=base_threshold,
                    print_report=False,
                    sample_duration_seconds=4.0,
                )
            ),
            "macro_patient": _macro_patient_metrics(patients, condition),
        }
    duration_seconds = time.perf_counter() - started
    body = {
        "schema_version": SCHEMA_VERSION,
        "objective": "fixed Scheme C base -> patient rest E2 baseline -> event replay head adaptation",
        "config": settings.to_dict(),
        "methodology": {
            "shared_feature_extractor_frozen": True,
            "trainable_per_patient": "classifier Linear(896, 2) only",
            "experiences": "first two future seizure events in chronological order",
            "replay": "all prior target-patient feature rows plus current experience",
            "positive_handling": "balanced feature-level replication per epoch",
            "holdout": "all future rows after the second event plus raw-window overlap guard",
            "primary_metrics": ["future_holdout_auroc", "future_holdout_auprc"],
            "primary_aggregation": "macro over per-patient metrics",
            "pooled_probability_warning": (
                "After private-head adaptation, raw probability scales differ by patient; "
                "pooled AUROC/AUPRC are diagnostic only."
            ),
            "checkpoint_selection_warning": (
                "The fixed base epoch was selected on pooled chb10-chb14 labels; "
                "this measures paired personalization, not an untouched external test."
            ),
        },
        "partition": _public_partition(partition),
        "baselines": {
            "population": population_summary,
            "patient_enrollment": patient_baseline_summary,
        },
        "base_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": artifact["checkpoints"]["auroc"]["sha256"],
            "selected_epoch": artifact["checkpoints"]["auroc"]["epoch"],
            "fixed_threshold": base_threshold,
        },
        "patients": patients,
        "aggregate": aggregate,
        "runtime": {
            "duration_seconds": duration_seconds,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(selected_device),
        },
        "source": {
            "scheme_c_artifact": str(artifact_path),
            "natural_cache": str(cache_path),
        },
    }
    digest = canonical_hash(body)
    result = {**body, "artifact_sha256": digest}
    result_path = write_content_addressed_json(
        result,
        output_dir / f"continual_personalization_{digest[:12]}.json",
        hash_field="artifact_sha256",
    )
    report_path = output_dir / "SCHEME_C_CONTINUAL_PERSONALIZATION_RESULTS.md"
    _write_report(result, report_path)
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "artifact": str(result_path),
        "report": str(report_path),
        "aggregate": aggregate,
        "duration_seconds": duration_seconds,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(
            "artifacts/chbmit/good_multibranch_scheme_c_aligned/scheme_c_aligned_689d4192c374.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/chbmit/scheme_c_continual_personalization"),
    )
    parser.add_argument("--enrollment-windows", type=int, default=128)
    parser.add_argument("--adaptation-events", type=int, default=2)
    parser.add_argument("--epochs-per-experience", type=int, default=12)
    parser.add_argument("--replay-batch-size", type=int, default=16)
    parser.add_argument("--prediction-batch-size", type=int, default=128)
    args = parser.parse_args(argv)
    result = run_experiment(
        artifact_path=args.artifact,
        output_dir=args.output_dir,
        config=ContinualPersonalizationConfig(
            enrollment_windows=args.enrollment_windows,
            adaptation_events=args.adaptation_events,
            epochs_per_experience=args.epochs_per_experience,
            replay_batch_size=args.replay_batch_size,
            prediction_batch_size=args.prediction_batch_size,
        ),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
