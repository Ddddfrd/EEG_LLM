"""Reproduce the retained E1+E2 STFT-64 training protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ai.chbmit.eeg_continual_pretrain import ServerPretrainConfig, run_server_pretrain_fold

from .model import stft_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("artifacts/chbmit/chbmit_index_v1_1e1f0a81ecde.json"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/chbmit/1.0.0"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/chbmit/good/e1_e2_stft64"),
    )
    parser.add_argument(
        "--shared-data-dir",
        type=Path,
        default=Path("artifacts/chbmit/eeg_continual_pretrain"),
    )
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=64)
    parser.add_argument("--prediction-batch-size", type=int, default=128)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    result = run_server_pretrain_fold(
        index_path=args.index,
        data_root=args.data_root,
        output_dir=args.output_dir,
        fold=0,
        validation_fold=4,
        config=ServerPretrainConfig(
            max_epochs=args.max_epochs,
            micro_batch_size=args.micro_batch_size,
            effective_batch_size=128,
            prediction_batch_size=args.prediction_batch_size,
            enrollment_baseline_windows=128,
        ),
        stft_config=stft_config(),
        visual_bypass=True,
        relative_spectral_bypass=True,
        shared_data_dir=args.shared_data_dir,
        progress=not args.no_progress,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

