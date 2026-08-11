"""Frozen Phase 2 evaluation and model-scope contracts."""
from __future__ import annotations

from typing import Any, Iterable, Mapping


EVALUATION_PROTOCOL_VERSION = "evaluation_protocol_v3"
DATA_MANIFEST_SCHEMA_VERSION = "eeg_dataset_manifest_v1"
SPLIT_MANIFEST_SCHEMA_VERSION = "eeg_split_manifest_v3"
BASE_MODEL_PROTOCOL = "cross_patient_base_v1"
PERSONALIZED_MODEL_PROTOCOL = "patient_personalization_v1"
VALID_PARTITIONS = frozenset({"train", "calibration", "test"})


def build_scope(
    records: Iterable[Mapping[str, Any]],
    *,
    purpose: str,
    partition: str,
    labels_used: bool,
    manifest_id: str,
) -> dict[str, Any]:
    """Summarize exactly which patients, labels and groups serve one purpose."""
    if partition not in VALID_PARTITIONS:
        raise ValueError(f"Unsupported scope partition: {partition}")
    rows = list(records)
    return {
        "purpose": purpose,
        "partition": partition,
        "manifest_id": manifest_id,
        "patient_ids": sorted({int(row["patient_id"]) for row in rows}),
        "labels": sorted({int(row["label"]) for row in rows}),
        "labels_used": labels_used,
        "label_use": (
            "parameter_or_threshold_selection"
            if labels_used
            else "final_scoring_only"
        ),
        "group_ids": sorted({str(row["group_id"]) for row in rows}),
        "record_count": len(rows),
    }


def validate_model_scopes(
    *,
    model_protocol: str,
    training_scope: Mapping[str, Any],
    calibration_scope: Mapping[str, Any],
    test_scope: Mapping[str, Any],
) -> None:
    """Reject leakage or ambiguous base/personalized model declarations."""
    if model_protocol not in {BASE_MODEL_PROTOCOL, PERSONALIZED_MODEL_PROTOCOL}:
        raise ValueError(f"Unsupported model protocol: {model_protocol}")
    scopes = {
        "training": training_scope,
        "calibration": calibration_scope,
        "test": test_scope,
    }
    for name, scope in scopes.items():
        required = {
            "purpose",
            "partition",
            "manifest_id",
            "patient_ids",
            "labels",
            "labels_used",
            "label_use",
            "group_ids",
            "record_count",
        }
        missing = required.difference(scope)
        if missing:
            raise ValueError(
                f"{name}_scope is missing fields: {', '.join(sorted(missing))}"
            )
        expected_partition = {
            "training": "train",
            "calibration": "calibration",
            "test": "test",
        }[name]
        if scope["partition"] != expected_partition:
            raise ValueError(
                f"{name}_scope partition must be {expected_partition!r}"
            )
        if not scope["manifest_id"]:
            raise ValueError(f"{name}_scope requires a manifest_id")
        if int(scope["record_count"]) < 1:
            raise ValueError(f"{name}_scope cannot be empty")
        expected_label_use = (
            "parameter_or_threshold_selection"
            if scope["labels_used"]
            else "final_scoring_only"
        )
        if scope["label_use"] != expected_label_use:
            raise ValueError(f"{name}_scope has inconsistent label_use")

    group_sets = {
        name: set(map(str, scope["group_ids"]))
        for name, scope in scopes.items()
    }
    for left, right in (
        ("training", "calibration"),
        ("training", "test"),
        ("calibration", "test"),
    ):
        overlap = group_sets[left] & group_sets[right]
        if overlap:
            raise ValueError(
                f"{left} and {right} scopes share groups: {sorted(overlap)[:3]}"
            )

    training_patients = set(map(int, training_scope["patient_ids"]))
    calibration_patients = set(map(int, calibration_scope["patient_ids"]))
    test_patients = set(map(int, test_scope["patient_ids"]))
    if model_protocol == BASE_MODEL_PROTOCOL:
        if test_patients & (training_patients | calibration_patients):
            raise ValueError("Base model target patients must be completely held out")
        if not calibration_scope["labels_used"]:
            raise ValueError("Base model calibration labels select epoch/threshold")
    else:
        if not test_patients:
            raise ValueError("Personalized model requires a frozen target holdout")
        if not test_patients.issubset(training_patients | calibration_patients):
            raise ValueError(
                "Personalized holdout patients must have enrollment/calibration data"
            )

    if test_scope["labels_used"]:
        raise ValueError(
            "Test labels cannot participate in training or threshold selection"
        )
    if not training_scope["labels_used"]:
        raise ValueError("Training scope must use labels")
    if not calibration_scope["labels_used"]:
        raise ValueError("Calibration scope must use labels")


def checkpoint_scope_metadata(
    *,
    model_protocol: str,
    training_scope: Mapping[str, Any],
    calibration_scope: Mapping[str, Any],
    test_scope: Mapping[str, Any],
) -> dict[str, Any]:
    validate_model_scopes(
        model_protocol=model_protocol,
        training_scope=training_scope,
        calibration_scope=calibration_scope,
        test_scope=test_scope,
    )
    return {
        "evaluation_protocol": EVALUATION_PROTOCOL_VERSION,
        "model_protocol": model_protocol,
        "training_scope": dict(training_scope),
        "calibration_scope": dict(calibration_scope),
        "test_scope": dict(test_scope),
    }


def validate_checkpoint_scopes(checkpoint: Mapping[str, Any]) -> bool:
    if checkpoint.get("evaluation_protocol") != EVALUATION_PROTOCOL_VERSION:
        return False
    try:
        validate_model_scopes(
            model_protocol=str(checkpoint["model_protocol"]),
            training_scope=checkpoint["training_scope"],
            calibration_scope=checkpoint["calibration_scope"],
            test_scope=checkpoint["test_scope"],
        )
    except (KeyError, TypeError, ValueError):
        return False
    return True
