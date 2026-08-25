"""Run matched Qwen2.5-0.5B pooling ablations for Scheme C."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from functools import partial
from pathlib import Path
from typing import Any, Sequence

from ai.chbmit.eegvl_multibranch_experiment import MultibranchTrainingConfig
from .model import DEFAULT_QWEN_MODEL, build_model
from .train_scheme_c_aligned import MAX_CALIBRATION_WINDOWS
from .train_scheme_c_eegmamba_split import run_experiment


POOLING_MODES = ("visual_mean", "visual_attention", "summary_token")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-artifact",
        type=Path,
        default=Path(
            "artifacts/chbmit/eeg_continual_pretrain_strict_e2_smoke/"
            "fold0_pretrain_c27817a49668.json"
        ),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/chbmit/1.0.0"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "artifacts/chbmit/scheme_c_qwen25_05b_pooling_ablation"
        ),
    )
    parser.add_argument(
        "--shared-cache-dir",
        type=Path,
        default=Path("artifacts/chbmit/good_multibranch_scheme_c_aligned/cache"),
    )
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--effective-batch-size", type=int, default=32)
    parser.add_argument("--prediction-batch-size", type=int, default=32)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=POOLING_MODES,
        default=list(POOLING_MODES),
    )
    args = parser.parse_args(argv)

    progress_path = args.output_root / "pooling_ablation_progress.json"
    progress: dict[str, Any] = {
        "schema_version": "scheme_c_qwen25_pooling_ablation_v1",
        "status": "running",
        "pooling_modes": list(args.modes),
        "completed": {},
        "started_unix": time.time(),
    }
    _write_json_atomic(progress_path, progress)

    for pooling in args.modes:
        output_dir = args.output_root / pooling
        existing = sorted(output_dir.glob("scheme_c_eegmamba_split_*.json"))
        if existing:
            progress["completed"][pooling] = {
                "status": "skipped_existing",
                "artifact": str(existing[-1]),
            }
            _write_json_atomic(progress_path, progress)
            continue

        progress["active"] = pooling
        _write_json_atomic(progress_path, progress)
        result = run_experiment(
            reference_artifact_path=args.reference_artifact,
            data_root=args.data_root,
            output_dir=output_dir,
            shared_cache_dir=args.shared_cache_dir,
            qwen_model_name=args.qwen_model,
            local_files_only=not args.allow_model_download,
            config=MultibranchTrainingConfig(
                max_epochs=5,
                micro_batch_size=args.micro_batch_size,
                effective_batch_size=args.effective_batch_size,
                prediction_batch_size=args.prediction_batch_size,
                enrollment_baseline_windows=MAX_CALIBRATION_WINDOWS,
                checkpoint_metric="auroc",
            ),
            model_builder=partial(build_model, pooling=pooling),
        )
        progress["completed"][pooling] = {"status": "complete", **result}
        _write_json_atomic(progress_path, progress)

    progress.pop("active", None)
    progress["status"] = "complete"
    progress["finished_unix"] = time.time()
    _write_json_atomic(progress_path, progress)
    print(json.dumps(progress, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
