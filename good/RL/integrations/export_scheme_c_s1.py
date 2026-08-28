"""Export frozen Scheme C S1 probabilities into eeg_alarm_policy artifacts.

Read-only bridge: loads the promoted S1 checkpoint, rebuilds each subject's
natural timeline identity arrays from the existing deep-timeline caches, runs
one deterministic inference pass over the natural-timeline image caches, and
writes one content-addressed probability artifact per subject. The exported
probabilities are cross-checked against the pooled AUROC/AUPRC recorded in the
authoritative S1 result artifact. Final-test subjects (chb22-chb23) are refused
unless ``--export-held-out`` is passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

_INTEGRATIONS_DIR = Path(__file__).resolve().parent
_RL_DIR = _INTEGRATIONS_DIR.parent
_ASTAR_ROOT = _RL_DIR.parent.parent
for _path in (str(_RL_DIR), str(_ASTAR_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from ai.chbmit.deep_timeline import DeepTargetTimeline  # noqa: E402
from ai.chbmit.direct20 import build_direct20_index  # noqa: E402
from ai.chbmit.eeg_continual_eval_cache import (  # noqa: E402
    NaturalFoldImageCache,
    NaturalFoldImageDataset,
)
from ai.chbmit.eeg_continual_pretrain import (  # noqa: E402
    compute_subject_log_spectral_baselines,
)
from ai.chbmit.eeg_continual_pretrain_model import ServerSTFTConfig  # noqa: E402
from ai.chbmit.eegmamba_b_experiment import _load_json, _seed_everything  # noqa: E402
from ai.chbmit.eegvl_multibranch_model import (  # noqa: E402
    load_portable_multibranch_state_dict,
)
from ai.chbmit.eegvl_training import predict_dataset  # noqa: E402
from ai.chbmit.windows import WindowConfig, load_chbmit_index  # noqa: E402
from good.e1_e2_e3_e4_qwen25_visual_mean.model import (  # noqa: E402
    DEFAULT_QWEN_MODEL,
    build_model,
)
from good.e1_e2_e3_e4_qwen25_visual_mean.train_stft_s1 import s1_stft_config  # noqa: E402

from eeg_alarm_policy.artifacts import (  # noqa: E402
    save_content_addressed_json,
    save_prediction_artifact,
)
from eeg_alarm_policy.contracts import EventInterval, ProbabilityTimeline  # noqa: E402
from eeg_alarm_policy.splits import (  # noqa: E402
    DevelopmentGate,
    base_model_role,
)

EXPORT_SCHEMA_VERSION = "eeg_rl_scheme_c_s1_probability_export_v1"
HISTORICAL_BASELINE_BATCH_SIZE = 64
NATURAL_WINDOW_CONFIG = WindowConfig(
    window_seconds=4.0,
    stride_seconds=4.0,
    ictal_overlap_fraction=0.5,
    seizure_guard_seconds=0.0,
    normal_to_ictal_ratio=0.0,
    sampling_frequency_hz=256,
    sampling_seed=0,
)
ROLE_SUBJECTS: dict[str, tuple[str, ...]] = {
    "validation": ("chb20", "chb21"),
    "test": ("chb22", "chb23"),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_existing(path: Path, description: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{description} is missing: {resolved}")
    return resolved


def _load_natural_cache(
    *,
    natural_cache_dir: Path | None,
    subjects: Sequence[str],
    shared_cache_dir: Path,
    build_if_missing: bool,
) -> NaturalFoldImageCache:
    if natural_cache_dir is not None:
        return NaturalFoldImageCache(_resolve_existing(natural_cache_dir, "natural cache"))
    if not build_if_missing:
        raise FileNotFoundError(
            "No natural-timeline image cache is recorded for this role. Re-run with "
            "--build-train-cache to materialize it (reads every EDF once; slow)."
        )
    from ai.chbmit.eegvl_multibranch_experiment import _build_fullband_natural_cache

    index = _loaded_index()
    _, dataset = _build_fullband_natural_cache(
        index=index,
        subjects=tuple(subjects),
        data_root=_data_root(),
        shared_dir=shared_cache_dir,
        preprocess=_loaded_preprocess(),
        seed=42,
    )
    return dataset.cache


# Module-level handles filled by prepare_assets so helper functions stay small.
_STATE: dict[str, Any] = {}


def _loaded_index() -> Mapping[str, Any]:
    return _STATE["index"]


def _data_root() -> Path:
    return _STATE["data_root"]


def _loaded_preprocess():
    return _STATE["preprocess"]


def _rebuild_partition_baselines(
    *,
    model: torch.nn.Module,
    cache: NaturalFoldImageCache,
    result: Mapping[str, Any],
    subjects: Sequence[str],
    partition: str,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Rebuild the exact S1 baseline and reject stale STFT-64 cache reuse."""
    expected = result["scheme_c_contract"]["e2_calibration"][partition]
    indices: dict[str, np.ndarray] = {}
    for subject in subjects:
        rows = cache.subject_slice(subject)
        normal_local = np.flatnonzero(np.asarray(cache.labels[rows]) == 0)
        count = int(expected["window_counts"][subject])
        if len(normal_local) < count:
            raise ValueError(
                f"{subject} has {len(normal_local)} natural normal windows; "
                f"{count} are required by the S1 baseline contract"
            )
        indices[subject] = normal_local[:count].astype(np.int64) + int(rows.start)
    baselines, summary = compute_subject_log_spectral_baselines(
        model.visual_encoder,
        images=cache.images,
        normal_indices_by_subject=indices,
        device=device,
        batch_size=HISTORICAL_BASELINE_BATCH_SIZE,
    )
    for subject in subjects:
        actual = summary["baseline_sha256"][subject]
        declared = expected["baseline_sha256"][subject]
        if actual != declared:
            raise ValueError(
                f"Rebuilt E2 baseline SHA256 mismatch for {subject}: "
                f"{actual} != {declared}"
            )
    summary["source"] = (
        "reconstructed from the selected natural-cache rows with the checkpoint "
        "STFT encoder; stale on-disk STFT-64 enrollment caches are not used"
    )
    return baselines, summary


def _timeline_for(subject: str, shared_cache_dir: Path) -> DeepTargetTimeline:
    from ai.chbmit.e2_enrollment import build_natural_enrollment_timelines

    timelines = build_natural_enrollment_timelines(
        _loaded_index(),
        subjects=[subject],
        output_dir=shared_cache_dir / "timelines",
        config=_STATE["enrollment_config"],
    )
    return timelines[str(subject)]


def _timeline_to_probability_timeline(
    timeline: DeepTargetTimeline,
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> ProbabilityTimeline:
    if not np.array_equal(np.asarray(labels), np.asarray(timeline.labels)):
        raise ValueError(
            f"Natural-cache labels disagree with deep timeline for "
            f"{timeline.metadata['target_subject']}"
        )
    window_config = timeline.metadata["window_config"]
    record_index_by_id = {
        str(record["record_id"]): int(record["record_index"])
        for record in timeline.metadata["records"]
    }
    events = tuple(
        EventInterval(
            event_index=int(event["event_index"]),
            event_id=str(event["event_id"]),
            record_index=record_index_by_id[str(event["record_id"])],
            start_seconds=float(event["start_seconds"]),
            end_seconds=float(event["end_seconds"]),
        )
        for event in timeline.metadata["events"]
    )
    return ProbabilityTimeline.create(
        subject_id=str(timeline.metadata["target_subject"]),
        probabilities=probabilities,
        labels=timeline.labels,
        record_indices=timeline.record_indices,
        start_samples=timeline.start_samples,
        event_indices=timeline.event_indices,
        records=tuple(
            str(record["record_id"]) for record in timeline.metadata["records"]
        ),
        events=events,
        sampling_frequency_hz=float(window_config["sampling_frequency_hz"]),
        window_seconds=float(window_config["window_seconds"]),
        stride_seconds=float(window_config["stride_seconds"]),
    )


def _cross_check(
    *,
    cache: NaturalFoldImageCache,
    probabilities: np.ndarray,
    result: Mapping[str, Any],
    checkpoint_role: str,
    subjects: Sequence[str],
    partition: str,
    tolerance: float,
) -> dict[str, Any]:
    window_count = int(cache.metadata["window_count"])
    rows: list[int] = []
    for subject in subjects:
        start, end, _ = cache.subject_slice(str(subject)).indices(window_count)
        rows.extend(range(start, end))
    selected = np.asarray(rows, dtype=np.int64)
    pooled_probabilities = probabilities[selected]
    pooled_labels = np.asarray(cache.labels[selected])
    from sklearn.metrics import average_precision_score, roc_auc_score

    auroc = float(roc_auc_score(pooled_labels, pooled_probabilities))
    auprc = float(average_precision_score(pooled_labels, pooled_probabilities))
    reference = result[f"{partition}_evaluations"][checkpoint_role]["pooled_metrics"]
    deltas = {
        "auroc_delta": abs(auroc - float(reference["auroc"])),
        "auprc_delta": abs(auprc - float(reference["auprc"])),
    }
    if max(deltas.values()) > tolerance:
        raise ValueError(
            "Exported probabilities fail the S1 cross-check: "
            f"exported auroc={auroc:.8f}, auprc={auprc:.8f}, "
            f"reference auroc={float(reference['auroc']):.8f}, "
            f"auprc={float(reference['auprc']):.8f}, deltas={deltas}"
        )
    return {
        "subjects": list(subjects),
        "exported_pooled": {"auroc": auroc, "auprc": auprc},
        "reference_pooled": {
            "auroc": float(reference["auroc"]),
            "auprc": float(reference["auprc"]),
        },
        **deltas,
    }


def run_export(
    *,
    result_json_path: Path,
    checkpoint_role: str,
    roles: Sequence[str],
    output_dir: Path,
    batch_size: int,
    allow_model_download: bool,
    export_held_out: bool,
    build_train_cache: bool,
    cross_check_tolerance: float,
) -> dict[str, Any]:
    gate = DevelopmentGate(unlocked=export_held_out)
    result_path = _resolve_existing(result_json_path, "S1 result artifact")
    result = _load_json(result_path)
    checkpoint_entry = result["checkpoints"][checkpoint_role]
    checkpoint_path = _resolve_existing(
        Path(checkpoint_entry["path"]), f"{checkpoint_role} checkpoint"
    )
    checkpoint_digest = _sha256_file(checkpoint_path)
    if checkpoint_digest != checkpoint_entry["sha256"]:
        raise ValueError(
            f"Checkpoint SHA256 mismatch: {checkpoint_digest} != "
            f"{checkpoint_entry['sha256']}"
        )

    if len(roles) != 1 or roles[0] not in {"validation", "test"}:
        raise ValueError(
            "This exporter supports one complete validation or test role per run. "
            "Training-timeline export needs a separate exact reconstruction of "
            "the historical per-patient training baselines."
        )
    partition = str(roles[0])
    subjects = list(ROLE_SUBJECTS[partition])
    natural_cache_dir = Path(result["source"][f"{partition}_natural_cache"])
    gate.require_export_allowed(subjects)

    shared_cache_dir = Path(result["source"]["raw_cache"]).parent.parent
    reference_path = _resolve_existing(
        Path(result["source"]["reference_artifact"]), "reference artifact"
    )
    reference = _load_json(reference_path)
    base_index = load_chbmit_index(Path(reference["source"]["index"]))
    index = build_direct20_index(base_index)
    if str(index["index_sha256"]) != str(result["source"]["direct20_index_sha256"]):
        raise ValueError("Direct-20 index hash does not match the S1 result artifact")

    from ai.chbmit.e2_enrollment import E2EnrollmentConfig
    from ai.chbmit.eegvl_s1_data import S1PreprocessConfig
    from good.e1_e2_e3_e4_fullband.train_scheme_c_aligned import (
        CALIBRATION_FRACTION,
        MAX_CALIBRATION_WINDOWS,
    )

    _STATE["index"] = index
    _STATE["data_root"] = _ASTAR_ROOT / "data" / "chbmit" / "1.0.0"
    _STATE["preprocess"] = S1PreprocessConfig(recipe_id="p0_clip_scale")
    _STATE["enrollment_config"] = E2EnrollmentConfig(
        fraction=CALIBRATION_FRACTION,
        maximum_windows=MAX_CALIBRATION_WINDOWS,
        stride_seconds=4.0,
        seed=42,
    )

    cache = _load_natural_cache(
        natural_cache_dir=natural_cache_dir,
        subjects=subjects,
        shared_cache_dir=shared_cache_dir,
        build_if_missing=build_train_cache,
    )
    missing_from_cache = sorted(set(subjects) - set(cache.metadata["subject_order"]))
    if missing_from_cache:
        raise ValueError(f"Natural cache lacks subjects {missing_from_cache}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Probability export requires CUDA for a faithful pass")
    _seed_everything(42)
    stft_override: ServerSTFTConfig = s1_stft_config()
    model = build_model(
        qwen_model_name=str(result["source"].get("qwen_model", DEFAULT_QWEN_MODEL)),
        local_files_only=not allow_model_download,
        pretrained_visual_encoder=False,
        stft_config_override=stft_override,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload["model_contract"] != result["model_contract"]:
        raise ValueError("Checkpoint model contract differs from the S1 result artifact")
    if model.contract() != payload["model_contract"]:
        raise ValueError("Current model factory no longer matches the S1 checkpoint")
    load_portable_multibranch_state_dict(model, payload["state_dict"])
    model.to(device)

    started = time.perf_counter()
    baselines, baseline_summary = _rebuild_partition_baselines(
        model=model,
        cache=cache,
        result=result,
        subjects=subjects,
        partition=partition,
        device=device,
    )
    dataset = NaturalFoldImageDataset(cache, subject_baselines=baselines)
    probabilities, labels = predict_dataset(
        model,
        dataset,
        device=device,
        batch_size=batch_size,
    )

    artifacts: dict[str, str] = {}
    artifact_paths: dict[str, str] = {}
    window_count = int(cache.metadata["window_count"])
    for subject in subjects:
        start, end, _ = cache.subject_slice(subject).indices(window_count)
        timeline = _timeline_for(subject, shared_cache_dir)
        probability_timeline = _timeline_to_probability_timeline(
            timeline,
            probabilities[start:end],
            np.asarray(labels[start:end]),
        )
        artifact = save_prediction_artifact(
            probability_timeline,
            output_dir,
            partition_role=base_model_role(subject),
            model_metadata={
                "checkpoint_role": checkpoint_role,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_digest,
                "checkpoint_epoch": checkpoint_entry.get("epoch"),
                "qwen_model": DEFAULT_QWEN_MODEL,
                "model_version": str(payload.get("model_version", "")),
                "stft": stft_override.to_dict(),
                "model_contract_sha256": hashlib.sha256(
                    json.dumps(
                        payload["model_contract"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "result_artifact": str(result_path),
                "result_artifact_sha256": result["artifact_sha256"],
            },
            source_metadata={
                "index_sha256": str(index["index_sha256"]),
                "window_manifest": result["source"]["window_manifest"],
                "window_manifest_sha256": result["source"]["window_manifest_sha256"],
                "natural_cache": str(cache.path),
                "natural_cache_key": cache.metadata["cache_key"],
                "enrollment_cache": None,
                "enrollment_baseline_summary": {
                    "definition": baseline_summary["definition"],
                    "window_count": baseline_summary["window_counts"][subject],
                    "baseline_sha256": baseline_summary["baseline_sha256"][subject],
                    "reconstruction_batch_size": HISTORICAL_BASELINE_BATCH_SIZE,
                    "source": baseline_summary["source"],
                },
                "timeline_metadata_sha256": timeline.metadata["metadata_sha256"],
                "inference": {
                    "device": torch.cuda.get_device_name(device),
                    "batch_size": int(batch_size),
                    "seed": 42,
                },
            },
        )
        artifacts[subject] = artifact.metadata["artifact_id"]
        artifact_paths[subject] = str(artifact.metadata_path)
        print(
            f"exported {subject}: {probability_timeline.row_count} windows "
            f"-> {artifact.metadata_path.name}",
            flush=True,
        )

    cross_check: dict[str, Any] | None = None
    if natural_cache_dir is not None:
        cross_check = _cross_check(
            cache=cache,
            probabilities=probabilities,
            result=result,
            checkpoint_role=checkpoint_role,
            subjects=ROLE_SUBJECTS[partition],
            partition=partition,
            tolerance=cross_check_tolerance,
        )

    summary = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "method": "read-only probability export of the promoted Scheme C S1 model",
        "checkpoint_role": checkpoint_role,
        "roles": list(roles),
        "subjects": subjects,
        "artifacts": artifacts,
        "artifact_paths": artifact_paths,
        "cross_check": cross_check,
        "cross_check_tolerance": cross_check_tolerance,
        "source": {
            "result_artifact": str(result_path),
            "reference_artifact": str(reference_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_digest,
            "natural_cache": str(cache.path),
            "shared_cache_dir": str(shared_cache_dir),
        },
        "duration_seconds": time.perf_counter() - started,
    }
    summary_path = save_content_addressed_json(
        summary,
        output_dir,
        hash_field="export_sha256",
        stem="export_summary",
    )
    print(f"summary: {summary_path}")
    if cross_check is not None:
        print(
            "cross-check deltas: "
            f"auroc={cross_check['auroc_delta']:.3e}, "
            f"auprc={cross_check['auprc_delta']:.3e}"
        )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-json",
        type=Path,
        default=_ASTAR_ROOT
        / "artifacts"
        / "chbmit"
        / "scheme_c_qwen25_05b_visual_mean_stft_s1_128_128_32"
        / "scheme_c_eegmamba_split_5d8a9a8fd8cf.json",
    )
    parser.add_argument(
        "--checkpoint",
        choices=("auroc", "auprc"),
        default="auprc",
        help="Which promoted S1 checkpoint to export (plan section 3)",
    )
    parser.add_argument(
        "--roles",
        type=str,
        default="validation",
        help="Exactly one role: validation or test; test also requires --export-held-out",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_RL_DIR / "artifacts" / "chbmit" / "eeg_rl" / "predictions",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--export-held-out", action="store_true")
    parser.add_argument(
        "--build-train-cache",
        action="store_true",
        help="Materialize the chb01-19 natural-timeline image cache if missing",
    )
    parser.add_argument("--cross-check-tolerance", type=float, default=1e-5)
    args = parser.parse_args(argv)
    roles = [role.strip() for role in args.roles.split(",") if role.strip()]
    if not roles:
        parser.error("--roles must name at least one role")
    summary = run_export(
        result_json_path=args.result_json,
        checkpoint_role=args.checkpoint,
        roles=roles,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        allow_model_download=args.allow_model_download,
        export_held_out=args.export_held_out,
        build_train_cache=args.build_train_cache,
        cross_check_tolerance=args.cross_check_tolerance,
    )
    print(json.dumps({key: summary[key] for key in ("artifacts", "cross_check")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
