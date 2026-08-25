from __future__ import annotations

from pathlib import Path
from typing import Any

from good.e1_e2_e3_e4_qwen25_visual_mean import model as model_module
from good.e1_e2_e3_e4_qwen25_visual_mean import train as train_module
from good.e1_e2_e3_e4_qwen25_visual_mean import train_stft_s1 as s1_module


def test_model_factory_fixes_visual_mean_pooling(monkeypatch: Any) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_build_fullband_model(**kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        model_module,
        "_build_fullband_model",
        fake_build_fullband_model,
    )

    result = model_module.build_model(
        qwen_model_name="local-qwen",
        local_files_only=True,
        pretrained_visual_encoder=False,
    )

    assert result is sentinel
    assert captured["pooling"] == "visual_mean"
    assert captured["qwen_model_name"] == "local-qwen"
    assert captured["local_files_only"] is True
    assert captured["pretrained_visual_encoder"] is False
    assert captured["stft_config_override"].n_fft == 64
    assert captured["stft_config_override"].win_length == 64
    assert captured["stft_config_override"].hop_length == 32


def test_training_entry_uses_matched_scheme_c_defaults(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_experiment(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(train_module, "run_experiment", fake_run_experiment)

    exit_code = train_module.main(
        [
            "--reference-artifact",
            str(tmp_path / "reference.json"),
            "--data-root",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "output"),
            "--shared-cache-dir",
            str(tmp_path / "cache"),
        ]
    )

    config = captured["config"]
    assert exit_code == 0
    assert captured["model_builder"] is model_module.build_model
    assert config.max_epochs == 5
    assert config.micro_batch_size == 8
    assert config.effective_batch_size == 32
    assert config.prediction_batch_size == 32
    assert captured["stft_config_override"].n_fft == 64


def test_s1_training_changes_only_stft_resolution(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_experiment(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(s1_module, "run_experiment", fake_run_experiment)

    exit_code = s1_module.main(
        [
            "--reference-artifact",
            str(tmp_path / "reference.json"),
            "--data-root",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "output"),
            "--shared-cache-dir",
            str(tmp_path / "cache"),
        ]
    )

    config = captured["config"]
    stft = captured["stft_config_override"]
    assert exit_code == 0
    assert captured["model_builder"] is model_module.build_model
    assert (stft.n_fft, stft.win_length, stft.hop_length) == (128, 128, 32)
    assert stft.zscore_input is False
    assert config.max_epochs == 5
    assert config.micro_batch_size == 8
    assert config.effective_batch_size == 32
