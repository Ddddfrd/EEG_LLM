"""Create the immutable R4 protocol freeze before held-out access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .artifacts import save_content_addressed_json

SCHEMA_VERSION = "eeg_rl_r4_protocol_freeze_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "artifact_hash": payload.get("result_sha256"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1", type=Path, required=True)
    parser.add_argument("--r2", type=Path, required=True)
    parser.add_argument("--r3", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    r1 = json.loads(args.r1.read_text(encoding="utf-8"))
    r2 = json.loads(args.r2.read_text(encoding="utf-8"))
    r3 = json.loads(args.r3.read_text(encoding="utf-8"))
    if r3["ppo"]["promoted"]:
        raise ValueError("Freeze contract expects the non-promoted PPO conclusion")
    selected_rule = r1["selected"]["rule"]
    if selected_rule != {
        "threshold": 0.9,
        "vote_k": 2,
        "vote_n": 5,
        "refractory_seconds": 300.0,
        "ema_alpha": None,
        "hysteresis_off_threshold": None,
    }:
        raise ValueError("R1 selected rule changed before protocol freeze")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_before_final_test_export",
        "development_subjects": ["chb20", "chb21"],
        "final_test_subjects": ["chb22", "chb23"],
        "primary_method": {
            "family": "robust_fixed_rule",
            "rule": selected_rule,
            "selection": "joint chb20-chb21 with pooled and per-patient sensitivity >= 0.8",
        },
        "final_comparators": {
            "inherited_rule": {
                "threshold": 0.5987815260887146,
                "vote_k": 2,
                "vote_n": 3,
                "refractory_seconds": 60.0,
                "ema_alpha": None,
                "hysteresis_off_threshold": None,
            },
            "logistic_regression": r2["results"]["logistic_regression"]["selected"]["rule"],
            "mlp_32x32": r2["results"]["mlp_32x32"]["selected"]["rule"],
        },
        "excluded_from_final_test": {
            "ppo": (
                "not promoted: median validation seed fails sensitivity guardrail "
                "and performance is unstable across five seeds"
            ),
            "tabular_q": "diagnostic only; does not beat the robust fixed rule",
        },
        "objective": {
            "lambda_fa": 0.02,
            "lambda_latency": 0.001,
            "latency_normalizer_seconds": 60.0,
            "minimum_event_sensitivity": 0.8,
        },
        "test_execution": (
            "export frozen S1 probabilities once, apply these frozen methods without "
            "retuning, save per-subject and pooled event metrics"
        ),
        "sources": {
            "r1": _source(args.r1),
            "r2": _source(args.r2),
            "r3": _source(args.r3),
        },
    }
    path = save_content_addressed_json(
        payload,
        args.output_dir,
        hash_field="freeze_sha256",
        stem="r4_protocol_freeze",
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
