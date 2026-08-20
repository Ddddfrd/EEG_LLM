"""Reproduce the retained E1+E2+E3+E4 full-band training protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ai.chbmit.eegvl_models import DEFAULT_QWEN_MODEL
from ai.chbmit.eegvl_multibranch_experiment import (
    MultibranchTrainingConfig,
    run_multibranch_fold0,
)


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
        default=Path("artifacts/chbmit/good/e1_e2_e3_e4_fullband"),
    )
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--prediction-batch-size", type=int, default=128)
    args = parser.parse_args(argv)
    result = run_multibranch_fold0(
        reference_artifact_path=args.reference_artifact,
        data_root=args.data_root,
        output_dir=args.output_dir,
        qwen_model_name=args.qwen_model,
        local_files_only=not args.allow_model_download,
        config=MultibranchTrainingConfig(
            max_epochs=args.max_epochs,
            micro_batch_size=args.micro_batch_size,
            effective_batch_size=args.micro_batch_size,
            prediction_batch_size=args.prediction_batch_size,
        ),
    )
    print(json.dumps(result, indent=2, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

