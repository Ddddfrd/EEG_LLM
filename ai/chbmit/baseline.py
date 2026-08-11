"""Train CHB-MIT LOPO Random Forest and FeatureMLP pilot baselines."""
from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

from ai.v2.lightweight_dataset import write_content_addressed_json
from ai.v2.metrics import evaluate, find_optimal_threshold_exact
from ai.v2.model import FeatureMLP

from .cache import ChbmitWindowCache, load_window_manifest
from .dataset import (
    RobustFeatureScaler,
    aggregate_channel_features,
    partition_indices,
)
from .evaluation import AlarmConfig, evaluate_target_timeline
from .index import canonical_hash
from .timeline_cache import TargetTimelineCache


BASELINE_SCHEMA_VERSION = "chbmit_phase3_baseline_v1"
PILOT_TARGETS = ("chb01", "chb06", "chb12", "chb15", "chb24")


@dataclass(frozen=True)
class BaselineConfig:
    seed: int = 42
    minimum_calibration_recall: float = 0.80
    rf_estimators: int = 400
    rf_min_samples_leaf: int = 2
    mlp_max_epochs: int = 30
    mlp_patience: int = 5
    mlp_batch_size: int = 256
    mlp_learning_rate: float = 1e-3
    mlp_weight_decay: float = 1e-4
    prediction_batch_size: int = 2048
    alarm_vote_k: int = 2
    alarm_vote_n: int = 3
    alarm_refractory_seconds: float = 60.0

    def validate(self) -> None:
        if not 0 <= self.minimum_calibration_recall <= 1:
            raise ValueError("Minimum calibration recall must be in [0, 1]")
        if self.rf_estimators < 1 or self.rf_min_samples_leaf < 1:
            raise ValueError("Random Forest limits must be positive")
        if (
            self.mlp_max_epochs < 1
            or self.mlp_patience < 1
            or self.mlp_batch_size < 2
        ):
            raise ValueError("FeatureMLP training limits must be positive")
        if self.mlp_learning_rate <= 0 or self.mlp_weight_decay < 0:
            raise ValueError("FeatureMLP optimizer settings are invalid")
        if self.prediction_batch_size < 1:
            raise ValueError("Prediction batch size must be positive")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_single(directory: Path, pattern: str) -> Path:
    matches = sorted(Path(directory).glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one artifact matching {pattern} in {directory}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _load_split(directory: Path, target: str) -> dict[str, Any]:
    path = _load_single(directory, f"lopo_{target}_*.json")
    split = json.loads(path.read_text(encoding="utf-8"))
    body = {
        key: value for key, value in split.items() if key != "split_sha256"
    }
    if split.get("split_sha256") != canonical_hash(body):
        raise ValueError(f"LOPO split hash is invalid: {path}")
    return split


def _load_timeline(directory: Path, target: str) -> TargetTimelineCache:
    return TargetTimelineCache(
        _load_single(directory, f"timeline_{target}_*")
    )


def _balanced_class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels.astype(np.int64), minlength=2)
    if np.any(counts == 0):
        raise ValueError("Training requires both classes")
    return len(labels) / (2.0 * counts.astype(np.float64))


def _transform_selected(
    features: np.ndarray,
    indices: np.ndarray,
    scaler: RobustFeatureScaler,
) -> np.ndarray:
    return scaler.transform(
        np.asarray(features[indices], dtype=np.float32)
    )


def _predict_rf_timeline(
    model: RandomForestClassifier,
    timeline: TargetTimelineCache,
    scaler: RobustFeatureScaler,
    *,
    batch_size: int,
) -> np.ndarray:
    probabilities = np.empty(len(timeline.labels), dtype=np.float64)
    for start in range(0, len(probabilities), batch_size):
        end = min(start + batch_size, len(probabilities))
        features = scaler.transform(
            np.asarray(timeline.features[start:end], dtype=np.float32)
        )
        aggregated = aggregate_channel_features(features)
        probabilities[start:end] = model.predict_proba(aggregated)[:, 1]
    return probabilities


@torch.no_grad()
def _predict_mlp(
    model: FeatureMLP,
    features: np.ndarray,
    scaler: RobustFeatureScaler,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    probabilities = np.empty(len(features), dtype=np.float64)
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        transformed = scaler.transform(
            np.asarray(features[start:end], dtype=np.float32)
        )
        batch = torch.from_numpy(transformed).to(device)
        probabilities[start:end] = (
            torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy()
        )
    return probabilities


def _selection_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    minimum_recall: float,
) -> tuple[float, dict[str, Any]]:
    threshold = find_optimal_threshold_exact(
        labels,
        probabilities,
        min_recall=minimum_recall,
    )
    metrics = evaluate(
        labels,
        probabilities,
        threshold=threshold,
        print_report=False,
        sample_duration_seconds=2.0,
    )
    return float(threshold), metrics


def _alarm_config(config: BaselineConfig) -> AlarmConfig:
    return AlarmConfig(
        vote_k=config.alarm_vote_k,
        vote_n=config.alarm_vote_n,
        refractory_seconds=config.alarm_refractory_seconds,
    )


def train_random_forest(
    cache: ChbmitWindowCache,
    timeline: TargetTimelineCache,
    train_indices: np.ndarray,
    calibration_indices: np.ndarray,
    scaler: RobustFeatureScaler,
    *,
    config: BaselineConfig,
) -> tuple[RandomForestClassifier, dict[str, Any]]:
    train_features = aggregate_channel_features(
        _transform_selected(cache.features, train_indices, scaler)
    )
    train_labels = np.asarray(
        cache.labels[train_indices], dtype=np.int64
    )
    calibration_features = aggregate_channel_features(
        _transform_selected(cache.features, calibration_indices, scaler)
    )
    calibration_labels = np.asarray(
        cache.labels[calibration_indices], dtype=np.int64
    )
    model = RandomForestClassifier(
        n_estimators=config.rf_estimators,
        min_samples_leaf=config.rf_min_samples_leaf,
        class_weight="balanced_subsample",
        max_features="sqrt",
        n_jobs=-1,
        random_state=config.seed,
    )
    model.fit(train_features, train_labels)
    calibration_probabilities = model.predict_proba(
        calibration_features
    )[:, 1]
    threshold, calibration_metrics = _selection_metrics(
        calibration_labels,
        calibration_probabilities,
        minimum_recall=config.minimum_calibration_recall,
    )
    target_probabilities = _predict_rf_timeline(
        model,
        timeline,
        scaler,
        batch_size=config.prediction_batch_size,
    )
    target_metrics = evaluate_target_timeline(
        timeline,
        target_probabilities,
        threshold=threshold,
        alarm_config=_alarm_config(config),
    )
    return model, {
        "threshold": threshold,
        "calibration_metrics": calibration_metrics,
        "target_metrics": target_metrics,
        "training_windows": len(train_indices),
        "calibration_windows": len(calibration_indices),
    }


def train_feature_mlp(
    cache: ChbmitWindowCache,
    timeline: TargetTimelineCache,
    train_indices: np.ndarray,
    calibration_indices: np.ndarray,
    scaler: RobustFeatureScaler,
    *,
    config: BaselineConfig,
    device: torch.device,
) -> tuple[FeatureMLP, dict[str, Any]]:
    _seed_everything(config.seed)
    train_features = _transform_selected(
        cache.features, train_indices, scaler
    )
    train_labels = np.asarray(
        cache.labels[train_indices], dtype=np.int64
    )
    calibration_features = _transform_selected(
        cache.features, calibration_indices, scaler
    )
    calibration_labels = np.asarray(
        cache.labels[calibration_indices], dtype=np.int64
    )
    model = FeatureMLP().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.mlp_learning_rate,
        weight_decay=config.mlp_weight_decay,
    )
    labels_tensor = torch.from_numpy(train_labels).to(device)
    features_tensor = torch.from_numpy(train_features).to(device)
    class_weights = torch.from_numpy(
        _balanced_class_weights(train_labels).astype(np.float32)
    ).to(device)
    generator = torch.Generator().manual_seed(config.seed)
    best_score: tuple[float, float, int] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_threshold = 0.5
    best_calibration: dict[str, Any] | None = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, config.mlp_max_epochs + 1):
        model.train()
        order = torch.randperm(len(train_labels), generator=generator)
        for start in range(0, len(order), config.mlp_batch_size):
            indices = order[start : start + config.mlp_batch_size].to(device)
            if len(indices) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            logits = model(features_tensor[indices])
            loss = nn.functional.cross_entropy(
                logits,
                labels_tensor[indices],
                weight=class_weights,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        calibration_probabilities = _predict_mlp(
            model,
            calibration_features,
            RobustFeatureScaler(
                median=tuple(0.0 for _ in scaler.median),
                scale=tuple(1.0 for _ in scaler.scale),
            ),
            device=device,
            batch_size=config.prediction_batch_size,
        )
        threshold, metrics = _selection_metrics(
            calibration_labels,
            calibration_probabilities,
            minimum_recall=config.minimum_calibration_recall,
        )
        score = (
            float(metrics["f1"] or 0.0),
            float(metrics["auprc"] or 0.0),
            -epoch,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_threshold = threshold
            best_calibration = metrics
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= config.mlp_patience:
            break
    if best_state is None or best_calibration is None:
        raise RuntimeError("FeatureMLP did not produce a selectable checkpoint")
    model.load_state_dict(best_state)
    target_probabilities = _predict_mlp(
        model,
        timeline.features,
        scaler,
        device=device,
        batch_size=config.prediction_batch_size,
    )
    target_metrics = evaluate_target_timeline(
        timeline,
        target_probabilities,
        threshold=best_threshold,
        alarm_config=_alarm_config(config),
    )
    return model, {
        "threshold": best_threshold,
        "selected_epoch": best_epoch,
        "calibration_metrics": best_calibration,
        "target_metrics": target_metrics,
        "training_windows": len(train_indices),
        "calibration_windows": len(calibration_indices),
        "device": str(device),
    }


def _summary(rows: Sequence[Mapping[str, Any]], model_name: str) -> dict[str, Any]:
    selected = [row["models"][model_name] for row in rows if model_name in row["models"]]
    event = [row["target_metrics"]["event_metrics"] for row in selected]
    window = [row["target_metrics"]["window_metrics"] for row in selected]
    return {
        "folds": len(selected),
        "macro_event_sensitivity": float(np.mean([
            float(metrics["event_sensitivity"]) for metrics in event
        ])),
        "macro_false_alarms_per_hour": float(np.mean([
            float(metrics["false_alarms_per_hour"]) for metrics in event
        ])),
        "macro_window_auprc": float(np.mean([
            float(metrics["auprc"]) for metrics in window
        ])),
        "macro_window_f1": float(np.mean([
            float(metrics["f1"] or 0.0) for metrics in window
        ])),
    }


def run_baselines(
    cache: ChbmitWindowCache,
    window_manifest: Mapping[str, Any],
    *,
    splits_dir: Path,
    timelines_dir: Path,
    output_dir: Path,
    targets: Sequence[str] = PILOT_TARGETS,
    models: Sequence[str] = ("rf", "mlp"),
    config: BaselineConfig | None = None,
    device: torch.device | None = None,
) -> Path:
    settings = config or BaselineConfig()
    settings.validate()
    unknown = set(models) - {"rf", "mlp"}
    if unknown:
        raise ValueError(f"Unknown baseline models: {sorted(unknown)}")
    selected_device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    rows: list[dict[str, Any]] = []
    model_root = Path(output_dir).resolve() / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    for fold_number, target in enumerate(targets, start=1):
        split = _load_split(splits_dir, target)
        timeline = _load_timeline(timelines_dir, target)
        train_indices = partition_indices(
            window_manifest, split, "source_train"
        )
        calibration_indices = partition_indices(
            window_manifest, split, "source_calibration"
        )
        scaler = RobustFeatureScaler.fit(
            cache.features, train_indices
        )
        fold_models: dict[str, Any] = {}
        target_model_dir = model_root / target
        target_model_dir.mkdir(parents=True, exist_ok=True)
        if "rf" in models:
            rf_model, rf_result = train_random_forest(
                cache,
                timeline,
                train_indices,
                calibration_indices,
                scaler,
                config=settings,
            )
            rf_path = target_model_dir / "random_forest.joblib"
            joblib.dump({
                "model": rf_model,
                "scaler": scaler.to_dict(),
                "split_sha256": split["split_sha256"],
                "timeline_cache_key": timeline.metadata["cache_key"],
                "config": asdict(settings),
            }, rf_path)
            fold_models["rf"] = {
                **rf_result,
                "checkpoint": str(rf_path),
            }
        if "mlp" in models:
            mlp_model, mlp_result = train_feature_mlp(
                cache,
                timeline,
                train_indices,
                calibration_indices,
                scaler,
                config=settings,
                device=selected_device,
            )
            mlp_path = target_model_dir / "feature_mlp.pt"
            torch.save({
                "model_state_dict": mlp_model.state_dict(),
                "scaler": scaler.to_dict(),
                "split_sha256": split["split_sha256"],
                "timeline_cache_key": timeline.metadata["cache_key"],
                "config": asdict(settings),
            }, mlp_path)
            fold_models["mlp"] = {
                **mlp_result,
                "checkpoint": str(mlp_path),
            }
        rows.append({
            "target_subject": target,
            "source_calibration_subject": split[
                "source_calibration_subject"
            ],
            "split_sha256": split["split_sha256"],
            "timeline_cache_key": timeline.metadata["cache_key"],
            "scaler": scaler.to_dict(),
            "models": fold_models,
        })
        print(
            f"completed baseline fold {fold_number}/{len(targets)}: {target}",
            flush=True,
        )
    summaries = {
        model_name: _summary(rows, model_name)
        for model_name in models
    }
    body = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "window_manifest_sha256": cache.metadata[
            "window_manifest_sha256"
        ],
        "config": asdict(settings),
        "device": str(selected_device),
        "targets": list(targets),
        "models": list(models),
        "folds": rows,
        "summary": summaries,
    }
    result = {**body, "result_sha256": canonical_hash(body)}
    output = Path(output_dir).resolve() / (
        f"baseline_{result['result_sha256'][:12]}.json"
    )
    write_content_addressed_json(
        result, output, hash_field="result_sha256"
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--timelines-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/chbmit/phase3"),
    )
    parser.add_argument("--targets", nargs="+", default=list(PILOT_TARGETS))
    parser.add_argument(
        "--models", nargs="+", choices=("rf", "mlp"), default=["rf", "mlp"]
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--rf-estimators", type=int, default=400)
    parser.add_argument("--mlp-max-epochs", type=int, default=30)
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    cache = ChbmitWindowCache(args.cache)
    windows = load_window_manifest(args.windows)
    output = run_baselines(
        cache,
        windows,
        splits_dir=args.splits_dir,
        timelines_dir=args.timelines_dir,
        output_dir=args.output_dir,
        targets=args.targets,
        models=args.models,
        config=BaselineConfig(
            rf_estimators=args.rf_estimators,
            mlp_max_epochs=args.mlp_max_epochs,
        ),
        device=device,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    print(json.dumps({
        "result": str(output),
        "device": result["device"],
        "summary": result["summary"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
