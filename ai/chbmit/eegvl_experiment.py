"""Protocol-A orchestration for the EEG-VL S1 experiment."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyedflib
import torch
import torch.nn as nn

from ai.v2.lightweight_dataset import write_content_addressed_json

from .baseline import PILOT_TARGETS
from .cache import ChbmitWindowCache, _recipe_from_payload, load_window_manifest
from .dataset import partition_indices
from .eegvl_s1_data import (
    S1AugmentConfig,
    S1ImageDataset,
    S1PreprocessConfig,
    S1PreprocessedCache,
    augmentation_recipe,
    build_s1_preprocessed_cache,
    preprocess_s1_batch,
)
from .eegvl_s1_models import (
    EEGVL_S1_MODEL_VERSION,
    build_s1_model,
    describe_s1_model,
)
from .eegvl_training import (
    EEGVL_TRAINING_VERSION,
    EegvlTrainingConfig,
    predict_dataset,
    save_s1_checkpoint,
    select_precision,
    train_s1_model,
)
from .evaluation import AlarmConfig, evaluate_target_timeline
from .index import canonical_hash
from .montage import read_montage_window
from .timeline_cache import TargetTimelineCache
from .windows import load_chbmit_index


EEGVL_S1_SCHEMA_VERSION = "eegvl_s1_protocol_a_v1"
SCREENING_SCHEMA_VERSION = "eegvl_s1_screening_v1"
DEFAULT_PREPROCESS_RECIPES = (
    "p1_bandpass_clip_scale",
    "p0_clip_scale",
    "p2_bandpass_channel_zscore",
)
DEFAULT_INPUT_MODES = ("single_sum", "rgb_repeat")
DEFAULT_AUGMENTATION_RECIPES = (
    "a0_none",
    "a1_shift_amplitude",
    "a2_channel_cutout",
    "a3_random_filter",
    "a4_channel_mirror",
)


def _load_single(directory: Path, pattern: str) -> Path:
    matches = sorted(Path(directory).glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one artifact matching {pattern} in {directory}, "
            f"found {len(matches)}"
        )
    return matches[0]


def load_lopo_split(directory: Path, target: str) -> dict[str, Any]:
    path = _load_single(directory, f"lopo_{target}_*.json")
    split = json.loads(path.read_text(encoding="utf-8"))
    body = {
        key: value for key, value in split.items() if key != "split_sha256"
    }
    if split.get("split_sha256") != canonical_hash(body):
        raise ValueError(f"LOPO split hash is invalid: {path}")
    if str(split["target_subject"]) != target:
        raise ValueError("LOPO split target does not match requested target")
    return split


def load_target_timeline(directory: Path, target: str) -> TargetTimelineCache:
    return TargetTimelineCache(
        _load_single(directory, f"timeline_{target}_*")
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_metadata(device: torch.device) -> dict[str, Any]:
    cuda_device = None
    if device.type == "cuda":
        cuda_device = {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": int(
                torch.cuda.get_device_properties(device).total_memory
            ),
        }
    return {
        "python_packages": {
            name: _package_version(name)
            for name in (
                "numpy",
                "pyedflib",
                "scipy",
                "scikit-learn",
                "torch",
                "torchvision",
            )
        },
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
        "cuda_device": cuda_device,
    }


def _public_training_result(result: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        "model",
        "best_state_dict",
        "calibration_probabilities",
    }
    return {
        key: value for key, value in result.items() if key not in excluded
    }


def _calibration_reload_check(
    checkpoint: Path,
    *,
    model_name: str,
    input_mode: str,
    pretrained_encoder: bool,
    calibration_dataset: S1ImageDataset,
    reference_probabilities: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored = build_s1_model(
        model_name,
        pretrained_encoder=pretrained_encoder,
        input_mode=input_mode,
    )
    restored.load_state_dict(payload["state_dict"])
    restored.to(device)
    probabilities, labels = predict_dataset(
        restored,
        calibration_dataset,
        device=device,
        batch_size=batch_size,
    )
    maximum_difference = float(np.max(np.abs(
        probabilities - reference_probabilities
    )))
    if maximum_difference > 1e-6:
        raise RuntimeError(
            "Checkpoint reload changed calibration probabilities by "
            f"{maximum_difference:.3e}"
        )
    return restored, {
        "maximum_probability_difference": maximum_difference,
        "probability_count": int(len(probabilities)),
        "label_sha256": hashlib.sha256(
            labels.astype(np.int64, copy=False).tobytes()
        ).hexdigest(),
        "passed": True,
    }


def _train_with_oom_retry(
    *,
    model_name: str,
    input_mode: str,
    pretrained_encoder: bool,
    training_dataset: S1ImageDataset,
    calibration_dataset: S1ImageDataset,
    device: torch.device,
    config: EegvlTrainingConfig,
) -> tuple[nn.Module, dict[str, Any], EegvlTrainingConfig, list[int]]:
    attempted_batches: list[int] = []
    current = config
    while True:
        attempted_batches.append(current.micro_batch_size)
        model = build_s1_model(
            model_name,
            pretrained_encoder=pretrained_encoder,
            input_mode=input_mode,
        )
        try:
            result = train_s1_model(
                model,
                training_dataset,
                calibration_dataset,
                device=device,
                config=current,
            )
            return model, result, current, attempted_batches
        except torch.OutOfMemoryError:
            if device.type != "cuda" or current.micro_batch_size <= 1:
                raise
            del model
            gc.collect()
            torch.cuda.empty_cache()
            current = replace(
                current,
                micro_batch_size=max(1, current.micro_batch_size // 2),
            )
        except RuntimeError as exc:
            if (
                device.type != "cuda"
                or "out of memory" not in str(exc).lower()
                or current.micro_batch_size <= 1
            ):
                raise
            del model
            gc.collect()
            torch.cuda.empty_cache()
            current = replace(
                current,
                micro_batch_size=max(1, current.micro_batch_size // 2),
            )


def train_source_fold(
    *,
    source_cache: ChbmitWindowCache,
    image_cache: S1PreprocessedCache,
    window_manifest: Mapping[str, Any],
    split: Mapping[str, Any],
    model_name: str,
    input_mode: str,
    augmentation: S1AugmentConfig,
    pretrained_encoder: bool,
    config: EegvlTrainingConfig,
    device: torch.device,
    checkpoints_dir: Path,
) -> tuple[nn.Module, dict[str, Any]]:
    train_indices = partition_indices(
        window_manifest,
        split,
        "source_train",
    )
    calibration_indices = partition_indices(
        window_manifest,
        split,
        "source_calibration",
    )
    target_indices = partition_indices(
        window_manifest,
        split,
        "target_held_out",
    )
    if (
        np.intersect1d(train_indices, calibration_indices).size
        or np.intersect1d(train_indices, target_indices).size
        or np.intersect1d(calibration_indices, target_indices).size
    ):
        raise ValueError("S1 source/target partitions overlap")
    training_dataset = S1ImageDataset(
        image_cache,
        source_cache,
        window_manifest,
        train_indices,
        augmentation=augmentation,
        seed=config.seed,
    )
    calibration_dataset = S1ImageDataset(
        image_cache,
        source_cache,
        window_manifest,
        calibration_indices,
        augmentation=None,
        seed=config.seed,
    )
    model, training, used_config, attempts = _train_with_oom_retry(
        model_name=model_name,
        input_mode=input_mode,
        pretrained_encoder=pretrained_encoder,
        training_dataset=training_dataset,
        calibration_dataset=calibration_dataset,
        device=device,
        config=config,
    )
    model_description = describe_s1_model(model)
    model_config = {
        "model_name": model_name,
        "input_mode": input_mode,
        "pretrained_encoder": pretrained_encoder,
        "model_description": model_description,
    }
    checkpoint, checkpoint_sha256 = save_s1_checkpoint(
        training,
        model_name=model_name,
        model_config=model_config,
        output_dir=checkpoints_dir,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    restored, reload_check = _calibration_reload_check(
        checkpoint,
        model_name=model_name,
        input_mode=input_mode,
        pretrained_encoder=pretrained_encoder,
        calibration_dataset=calibration_dataset,
        reference_probabilities=training["calibration_probabilities"],
        device=device,
        batch_size=used_config.prediction_batch_size,
    )
    result = {
        "target_subject": str(split["target_subject"]),
        "source_calibration_subject": str(
            split["source_calibration_subject"]
        ),
        "model": model_description,
        "model_config": model_config,
        "training": _public_training_result(training),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha256,
            "size_bytes": int(checkpoint.stat().st_size),
            "reload_check": reload_check,
        },
        "partition_counts": {
            "source_train": int(len(train_indices)),
            "source_calibration": int(len(calibration_indices)),
            "target_held_out_not_used_for_training": int(len(target_indices)),
        },
        "target_isolation": {
            "target_windows_used_for_training": 0,
            "target_windows_used_for_epoch_selection": 0,
            "target_windows_used_for_threshold_selection": 0,
        },
        "oom_micro_batch_attempts": attempts,
        "training_config_used": used_config.to_dict(),
    }
    return restored, result


@torch.no_grad()
def predict_target_timeline_models(
    models: Mapping[str, nn.Module],
    *,
    timeline: TargetTimelineCache,
    index: Mapping[str, Any],
    data_root: Path,
    preprocess_config: S1PreprocessConfig,
    device: torch.device,
    batch_size: int,
    progress: bool = True,
) -> dict[str, np.ndarray]:
    if not models:
        raise ValueError("Target prediction requires at least one model")
    if batch_size < 1:
        raise ValueError("Target prediction batch size must be positive")
    if (
        timeline.metadata["dataset_index_sha256"]
        != index["index_sha256"]
    ):
        raise ValueError("Target timeline and dataset index do not match")
    records = {
        str(record["record_id"]): record for record in index["records"]
    }
    outputs = {
        name: np.empty(len(timeline.labels), dtype=np.float32)
        for name in models
    }
    precision = select_precision(device)
    for model in models.values():
        model.eval()
        model.to(device)
    root = Path(data_root).resolve()
    sampling_frequency = int(
        timeline.metadata["window_config"]["sampling_frequency_hz"]
    )
    sample_count = int(round(
        float(timeline.metadata["window_config"]["window_seconds"])
        * sampling_frequency
    ))
    if sample_count != 1024:
        raise ValueError("S1 Protocol A expects 1,024-sample target windows")

    for record_number, record_meta in enumerate(
        timeline.metadata["records"],
        start=1,
    ):
        record_id = str(record_meta["record_id"])
        record = records.get(record_id)
        if record is None:
            raise ValueError(f"Timeline references unknown record: {record_id}")
        path = root / Path(record_id)
        if not path.is_file():
            raise FileNotFoundError(f"EDF is missing: {path}")
        recipes = tuple(
            _recipe_from_payload(payload) for payload in record["montage"]
        )
        with pyedflib.EdfReader(str(path)) as reader:
            waveform = read_montage_window(
                reader,
                recipes,
                start_sample=0,
                sample_count=int(record["sample_count"]),
            )
        row_start = int(record_meta["row_start"])
        row_end = int(record_meta["row_end"])
        starts = np.asarray(
            timeline.start_samples[row_start:row_end],
            dtype=np.int64,
        )
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            raw = np.stack(
                [
                    waveform[:, start : start + sample_count]
                    for start in map(int, batch_starts)
                ],
                axis=0,
            )
            images, _ = preprocess_s1_batch(
                raw,
                config=preprocess_config,
            )
            tensor = torch.from_numpy(images).to(
                device,
                non_blocking=True,
            )
            begin = row_start + offset
            end = begin + len(batch_starts)
            for name, model in models.items():
                if device.type == "cuda":
                    dtype = (
                        torch.bfloat16
                        if precision == "bf16"
                        else torch.float16
                    )
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        logits = model(tensor)
                else:
                    logits = model(tensor)
                outputs[name][begin:end] = (
                    torch.softmax(logits.float(), dim=1)[:, 1]
                    .cpu()
                    .numpy()
                )
            del tensor, raw, images
        del waveform
        if progress:
            print(
                f"S1 target {timeline.metadata['target_subject']}: "
                f"{record_number}/{len(timeline.metadata['records'])} EDF files",
                flush=True,
            )
    if any(not np.isfinite(values).all() for values in outputs.values()):
        raise ValueError("S1 target inference produced non-finite probabilities")
    return outputs


def _save_prediction_artifact(
    probabilities: Mapping[str, np.ndarray],
    *,
    timeline: TargetTimelineCache,
    output_dir: Path,
) -> tuple[Path, str]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / f".predictions.{uuid.uuid4().hex}.tmp"
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            labels=np.asarray(timeline.labels, dtype=np.uint8),
            record_indices=np.asarray(
                timeline.record_indices,
                dtype=np.int16,
            ),
            start_samples=np.asarray(
                timeline.start_samples,
                dtype=np.int32,
            ),
            **{
                f"{name}_probabilities": np.asarray(values, dtype=np.float32)
                for name, values in probabilities.items()
            },
        )
    digest = _file_sha256(temporary)
    destination = output_dir / (
        f"{timeline.metadata['target_subject']}_{digest[:12]}.npz"
    )
    if destination.exists():
        temporary.unlink()
    else:
        os.replace(temporary, destination)
    return destination, digest


def select_screening_winner(
    rows: Sequence[Mapping[str, Any]],
    *,
    ordered_candidates: Sequence[str],
    candidate_field: str,
) -> str:
    if not rows:
        raise ValueError("Screening selection requires result rows")
    order = {name: index for index, name in enumerate(ordered_candidates)}
    eligible = [
        row for row in rows if str(row[candidate_field]) in order
    ]
    if len(eligible) != len(rows):
        raise ValueError("Screening row contains an unknown candidate")
    selected = max(
        eligible,
        key=lambda row: (
            float(row["calibration_auprc"]),
            -order[str(row[candidate_field])],
        ),
    )
    return str(selected[candidate_field])


def _screening_key(
    preprocess_recipe: str,
    input_mode: str,
    augmentation_recipe_id: str,
) -> str:
    return "|".join((
        preprocess_recipe,
        input_mode,
        augmentation_recipe_id,
    ))


def run_s1_screening(
    *,
    source_cache: ChbmitWindowCache,
    window_manifest: Mapping[str, Any],
    split: Mapping[str, Any],
    preprocess_cache_dir: Path,
    output_dir: Path,
    device: torch.device,
    config: EegvlTrainingConfig,
    pretrained_encoder: bool,
) -> tuple[dict[str, Any], Path]:
    if device.type != "cuda":
        raise RuntimeError("Scheduled S1 screening is CUDA-only")
    rows: dict[str, dict[str, Any]] = {}
    models: list[nn.Module] = []

    def run_candidate(
        preprocess_recipe: str,
        input_mode: str,
        augmentation_recipe_id: str,
        stage: str,
    ) -> dict[str, Any]:
        key = _screening_key(
            preprocess_recipe,
            input_mode,
            augmentation_recipe_id,
        )
        if key not in rows:
            print(f"S1 screening candidate {key} ({stage})", flush=True)
            preprocess = S1PreprocessConfig(recipe_id=preprocess_recipe)
            cache_path = build_s1_preprocessed_cache(
                source_cache,
                output_dir=preprocess_cache_dir,
                config=preprocess,
            )
            image_cache = S1PreprocessedCache(cache_path)
            model, result = train_source_fold(
                source_cache=source_cache,
                image_cache=image_cache,
                window_manifest=window_manifest,
                split=split,
                model_name="m3",
                input_mode=input_mode,
                augmentation=augmentation_recipe(
                    augmentation_recipe_id
                ),
                pretrained_encoder=pretrained_encoder,
                config=config,
                device=device,
                checkpoints_dir=output_dir / "checkpoints",
            )
            models.append(model)
            metric = result["training"]["calibration_metrics"]
            rows[key] = {
                "key": key,
                "stages": [stage],
                "preprocess_recipe": preprocess_recipe,
                "input_mode": input_mode,
                "augmentation_recipe": augmentation_recipe_id,
                "calibration_auprc": metric["auprc"],
                "calibration_auroc": metric["auroc"],
                "calibration_f1": metric["f1"],
                "best_epoch": result["training"]["best_epoch"],
                "checkpoint": result["checkpoint"],
                "training": result["training"],
                "preprocessed_cache": {
                    "path": str(cache_path),
                    "cache_key": image_cache.metadata["cache_key"],
                },
            }
            del models[:]
            gc.collect()
            torch.cuda.empty_cache()
        elif stage not in rows[key]["stages"]:
            rows[key]["stages"].append(stage)
        return rows[key]

    preprocess_rows = [
        run_candidate(
            recipe,
            "single_sum",
            "a0_none",
            "preprocessing",
        )
        for recipe in DEFAULT_PREPROCESS_RECIPES
    ]
    selected_preprocess = select_screening_winner(
        preprocess_rows,
        ordered_candidates=DEFAULT_PREPROCESS_RECIPES,
        candidate_field="preprocess_recipe",
    )
    input_rows = [
        run_candidate(
            selected_preprocess,
            mode,
            "a0_none",
            "input_conversion",
        )
        for mode in DEFAULT_INPUT_MODES
    ]
    selected_input = select_screening_winner(
        input_rows,
        ordered_candidates=DEFAULT_INPUT_MODES,
        candidate_field="input_mode",
    )
    augmentation_rows = [
        run_candidate(
            selected_preprocess,
            selected_input,
            recipe,
            "augmentation",
        )
        for recipe in DEFAULT_AUGMENTATION_RECIPES
    ]
    selected_augmentation = select_screening_winner(
        augmentation_rows,
        ordered_candidates=DEFAULT_AUGMENTATION_RECIPES,
        candidate_field="augmentation_recipe",
    )
    artifact_body = {
        "schema_version": SCREENING_SCHEMA_VERSION,
        "protocol": "A_source_validation_only",
        "selection_target_split": str(split["target_subject"]),
        "target_data_used": False,
        "selection_metric": "source_calibration_auprc",
        "tie_break": "earlier_ordered_candidate",
        "selected_recipe": {
            "preprocess_recipe": selected_preprocess,
            "input_mode": selected_input,
            "augmentation_recipe": selected_augmentation,
        },
        "candidate_rows": list(rows.values()),
        "training_config": config.to_dict(),
        "runtime": runtime_metadata(device),
    }
    artifact_sha256 = canonical_hash(artifact_body)
    artifact = {
        **artifact_body,
        "artifact_sha256": artifact_sha256,
    }
    path = write_content_addressed_json(
        artifact,
        output_dir / f"screening_{artifact_sha256[:12]}.json",
        hash_field="artifact_sha256",
    )
    return artifact, path


def _model_summary(
    folds: Sequence[Mapping[str, Any]],
    model_name: str,
) -> dict[str, Any]:
    selected = [
        fold["models"][model_name]
        for fold in folds
        if model_name in fold["models"]
    ]
    windows = [row["target_metrics"]["window_metrics"] for row in selected]
    events = [row["target_metrics"]["event_metrics"] for row in selected]

    def mean_defined(
        rows: Sequence[Mapping[str, Any]],
        key: str,
    ) -> tuple[float | None, int]:
        values = [
            float(row[key])
            for row in rows
            if row.get(key) is not None
        ]
        return (
            float(np.mean(values)) if values else None,
            len(values),
        )

    auprc, auprc_count = mean_defined(windows, "auprc")
    auroc, auroc_count = mean_defined(windows, "auroc")
    f1, f1_count = mean_defined(windows, "f1")
    sensitivity, sensitivity_count = mean_defined(
        events,
        "event_sensitivity",
    )
    false_alarms, false_alarm_count = mean_defined(
        events,
        "false_alarms_per_hour",
    )
    return {
        "fold_count": len(selected),
        "macro_target_auprc": auprc,
        "macro_target_auprc_defined_folds": auprc_count,
        "macro_target_auroc": auroc,
        "macro_target_auroc_defined_folds": auroc_count,
        "macro_target_f1": f1,
        "macro_target_f1_defined_folds": f1_count,
        "macro_event_sensitivity": sensitivity,
        "macro_event_sensitivity_defined_folds": sensitivity_count,
        "macro_false_alarms_per_hour": false_alarms,
        "macro_false_alarms_per_hour_defined_folds": false_alarm_count,
    }


def run_s1_final(
    *,
    source_cache: ChbmitWindowCache,
    window_manifest: Mapping[str, Any],
    index: Mapping[str, Any],
    data_root: Path,
    splits_dir: Path,
    timelines_dir: Path,
    preprocess_cache_dir: Path,
    output_dir: Path,
    targets: Sequence[str],
    selected_recipe: Mapping[str, str],
    device: torch.device,
    config: EegvlTrainingConfig,
    pretrained_encoder: bool,
    progress: bool = True,
) -> tuple[dict[str, Any], Path]:
    if device.type != "cuda":
        raise RuntimeError("Scheduled S1 final experiment is CUDA-only")
    preprocess_config = S1PreprocessConfig(
        recipe_id=str(selected_recipe["preprocess_recipe"])
    )
    cache_path = build_s1_preprocessed_cache(
        source_cache,
        output_dir=preprocess_cache_dir,
        config=preprocess_config,
    )
    image_cache = S1PreprocessedCache(cache_path)
    augmentation = augmentation_recipe(
        str(selected_recipe["augmentation_recipe"])
    )
    alarm_config = AlarmConfig()
    folds: list[dict[str, Any]] = []
    started = time.perf_counter()

    for target in targets:
        split = load_lopo_split(splits_dir, target)
        timeline = load_target_timeline(timelines_dir, target)
        fold_models: dict[str, nn.Module] = {}
        fold_results: dict[str, dict[str, Any]] = {}
        for model_name in ("m2", "m3"):
            print(
                f"S1 final {target}: training {model_name}",
                flush=True,
            )
            input_mode = (
                "single_sum"
                if model_name == "m2"
                else str(selected_recipe["input_mode"])
            )
            model, result = train_source_fold(
                source_cache=source_cache,
                image_cache=image_cache,
                window_manifest=window_manifest,
                split=split,
                model_name=model_name,
                input_mode=input_mode,
                augmentation=augmentation,
                pretrained_encoder=pretrained_encoder,
                config=config,
                device=device,
                checkpoints_dir=output_dir / "checkpoints",
            )
            fold_models[model_name] = model
            fold_results[model_name] = result
        probabilities = predict_target_timeline_models(
            fold_models,
            timeline=timeline,
            index=index,
            data_root=data_root,
            preprocess_config=preprocess_config,
            device=device,
            batch_size=config.prediction_batch_size,
            progress=progress,
        )
        prediction_path, prediction_sha = _save_prediction_artifact(
            probabilities,
            timeline=timeline,
            output_dir=output_dir / "predictions",
        )
        for model_name, values in probabilities.items():
            threshold = float(
                fold_results[model_name]["training"]["threshold"]
            )
            fold_results[model_name]["target_metrics"] = (
                evaluate_target_timeline(
                    timeline,
                    values,
                    threshold=threshold,
                    alarm_config=alarm_config,
                )
            )
            fold_results[model_name]["target_probability_sha256"] = (
                hashlib.sha256(values.tobytes()).hexdigest()
            )
        fold_body = {
            "target_subject": target,
            "split_sha256": split["split_sha256"],
            "timeline_cache_key": timeline.metadata["cache_key"],
            "prediction_artifact": {
                "path": str(prediction_path),
                "sha256": prediction_sha,
                "size_bytes": int(prediction_path.stat().st_size),
            },
            "models": fold_results,
        }
        fold = {
            **fold_body,
            "fold_sha256": canonical_hash(fold_body),
        }
        fold_path = output_dir / "folds" / (
            f"{target}_{fold['fold_sha256'][:12]}.json"
        )
        write_content_addressed_json(
            fold,
            fold_path,
            hash_field="fold_sha256",
        )
        folds.append({
            **fold,
            "fold_artifact": str(fold_path.resolve()),
        })
        del fold_models
        gc.collect()
        torch.cuda.empty_cache()

    summary = {
        model_name: _model_summary(folds, model_name)
        for model_name in ("m2", "m3")
    }
    artifact_body = {
        "schema_version": EEGVL_S1_SCHEMA_VERSION,
        "protocol": "A_current_project_screening",
        "seed": config.seed,
        "targets": list(targets),
        "selected_recipe": dict(selected_recipe),
        "preprocessed_cache": {
            "path": str(cache_path),
            "cache_key": image_cache.metadata["cache_key"],
            "metadata_sha256": image_cache.metadata["metadata_sha256"],
        },
        "folds": folds,
        "summary": summary,
        "gate": {
            "m3_macro_auprc_improves_over_m2": (
                summary["m3"]["macro_target_auprc"]
                > summary["m2"]["macro_target_auprc"]
            ),
            "note": (
                "The matched-event-sensitivity false-alarm comparison is "
                "reported separately from fixed source-calibrated thresholds."
            ),
        },
        "versions": {
            "experiment": EEGVL_S1_SCHEMA_VERSION,
            "model": EEGVL_S1_MODEL_VERSION,
            "training": EEGVL_TRAINING_VERSION,
        },
        "training_config": config.to_dict(),
        "runtime": runtime_metadata(device),
        "duration_seconds": time.perf_counter() - started,
    }
    artifact_sha256 = canonical_hash(artifact_body)
    artifact = {
        **artifact_body,
        "artifact_sha256": artifact_sha256,
    }
    path = write_content_addressed_json(
        artifact,
        output_dir / f"s1_protocol_a_{artifact_sha256[:12]}.json",
        hash_field="artifact_sha256",
    )
    return artifact, path


def run_s1(
    *,
    cache_path: Path,
    window_manifest_path: Path,
    index_path: Path,
    data_root: Path,
    splits_dir: Path,
    timelines_dir: Path,
    preprocess_cache_dir: Path,
    output_dir: Path,
    targets: Sequence[str] = PILOT_TARGETS,
    selection_target: str = "chb01",
    device: torch.device | None = None,
    config: EegvlTrainingConfig | None = None,
    pretrained_encoder: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    selected_device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if selected_device.type != "cuda":
        raise RuntimeError("S1 requires CUDA and will not fall back to CPU")
    settings = config or EegvlTrainingConfig()
    settings.validate()
    source_cache = ChbmitWindowCache(cache_path)
    window_manifest = load_window_manifest(window_manifest_path)
    index = load_chbmit_index(index_path)
    if (
        source_cache.metadata["window_manifest_sha256"]
        != window_manifest["window_manifest_sha256"]
    ):
        raise ValueError("S1 source cache and window manifest do not match")
    selection_split = load_lopo_split(splits_dir, selection_target)
    output_dir = Path(output_dir).resolve()
    screening, screening_path = run_s1_screening(
        source_cache=source_cache,
        window_manifest=window_manifest,
        split=selection_split,
        preprocess_cache_dir=preprocess_cache_dir,
        output_dir=output_dir / "screening",
        device=selected_device,
        config=settings,
        pretrained_encoder=pretrained_encoder,
    )
    final, final_path = run_s1_final(
        source_cache=source_cache,
        window_manifest=window_manifest,
        index=index,
        data_root=data_root,
        splits_dir=splits_dir,
        timelines_dir=timelines_dir,
        preprocess_cache_dir=preprocess_cache_dir,
        output_dir=output_dir / "final",
        targets=targets,
        selected_recipe=screening["selected_recipe"],
        device=selected_device,
        config=settings,
        pretrained_encoder=pretrained_encoder,
        progress=progress,
    )
    return {
        "screening_artifact": str(screening_path),
        "final_artifact": str(final_path),
        "selected_recipe": screening["selected_recipe"],
        "summary": final["summary"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--window-manifest", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--timelines-dir", type=Path, required=True)
    parser.add_argument("--preprocess-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(PILOT_TARGETS),
    )
    parser.add_argument("--selection-target", default="chb01")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=64)
    parser.add_argument("--prediction-batch-size", type=int, default=128)
    parser.add_argument("--no-pretrained-encoder", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    result = run_s1(
        cache_path=args.cache_path,
        window_manifest_path=args.window_manifest,
        index_path=args.index,
        data_root=args.data_root,
        splits_dir=args.splits_dir,
        timelines_dir=args.timelines_dir,
        preprocess_cache_dir=args.preprocess_cache_dir,
        output_dir=args.output_dir,
        targets=args.targets,
        selection_target=args.selection_target,
        config=EegvlTrainingConfig(
            max_epochs=args.max_epochs,
            patience=args.patience,
            micro_batch_size=args.micro_batch_size,
            prediction_batch_size=args.prediction_batch_size,
        ),
        pretrained_encoder=not args.no_pretrained_encoder,
        progress=not args.no_progress,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
