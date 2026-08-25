"""Train Scheme C with ModernBERT-base on the EEGMamba patient split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ai.chbmit.eegvl_multibranch_experiment import MultibranchTrainingConfig
from good.e1_e2_e3_e4_fullband.train_scheme_c_aligned import MAX_CALIBRATION_WINDOWS
from good.e1_e2_e3_e4_fullband.train_scheme_c_eegmamba_split import run_experiment

from .model import DEFAULT_MODERNBERT_MODEL, build_model


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
        "--output-dir",
        type=Path,
        default=Path("artifacts/chbmit/scheme_c_modernbert_base_eegmamba_split"),
    )
    parser.add_argument(
        "--shared-cache-dir",
        type=Path,
        default=Path("artifacts/chbmit/good_multibranch_scheme_c_aligned/cache"),
    )
    parser.add_argument("--model", default=DEFAULT_MODERNBERT_MODEL)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--prediction-batch-size", type=int, default=128)
    args = parser.parse_args(argv)

    result = run_experiment(
        reference_artifact_path=args.reference_artifact,
        data_root=args.data_root,
        output_dir=args.output_dir,
        shared_cache_dir=args.shared_cache_dir,
        qwen_model_name=args.model,
        local_files_only=not args.allow_model_download,
        config=MultibranchTrainingConfig(
            max_epochs=5,
            micro_batch_size=args.micro_batch_size,
            effective_batch_size=args.micro_batch_size,
            prediction_batch_size=args.prediction_batch_size,
            enrollment_baseline_windows=MAX_CALIBRATION_WINDOWS,
            checkpoint_metric="auroc",
        ),
        model_builder=build_model,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
