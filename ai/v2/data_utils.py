"""V2 数据加载：特征提取版本

每个 clip 提取以下逐通道特征（不做 z-score 归一化）：
  1. log_energy: log(1 + mean(x^2))
  2. log_abs_mean: log(1 + mean(|x|))  
  3. std: 标准差
  4. line_length: mean(|diff(x)|) - 波形复杂度
  5-10. 频段 log power: delta, theta, alpha, beta, gamma, HFO

= 10 features per channel; channel count is preserved for invariant aggregation.
"""
import os
import random
import re
from pathlib import Path
from typing import Any
import numpy as np
import scipy.io
from scipy.signal import welch
import torch
from torch.utils.data import Dataset
from .config import DATA_ROOT, TARGET_SR, TARGET_POINTS, PATIENT_IDS
from .feature_schema import (
    BAND_DEFS,
    BAND_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    PSD_METHOD,
)
from .evaluation_protocol import (
    EVALUATION_PROTOCOL_VERSION,
    SPLIT_MANIFEST_SCHEMA_VERSION,
)


# ============================================================
# 文件扫描
# ============================================================

CLIP_FILENAME_PATTERN = re.compile(
    r"^Patient_(?P<patient_id>\d+)_"
    r"(?P<label>ictal|interictal|test)_segment_(?P<segment>\d+)\.mat$",
    re.IGNORECASE,
)
LABEL_BY_NAME = {"interictal": 0, "ictal": 1}


def _parse_clip_filename(filename, expected_patient_id):
    match = CLIP_FILENAME_PATTERN.fullmatch(filename)
    if match is None:
        raise ValueError(
            "Unknown MAT clip filename. Expected "
            f"Patient_<id>_(ictal|interictal|test)_segment_<n>.mat; got {filename}"
        )
    file_patient_id = int(match.group("patient_id"))
    if file_patient_id != int(expected_patient_id):
        raise ValueError(
            f"Clip patient ID P{file_patient_id} does not match folder "
            f"Patient_{expected_patient_id}: {filename}"
        )
    return match.group("label").lower(), int(match.group("segment"))


def scan_patient_files(patient_id, return_exclusions=False, data_root=None):
    folder = os.path.join(data_root or DATA_ROOT, f"Patient_{patient_id}")
    results = []
    exclusions = []
    seen_segments = set()
    for fname in sorted(os.listdir(folder)):
        path = os.path.join(folder, fname)
        if not fname.lower().endswith(".mat"):
            exclusions.append({"path": path, "reason": "not_mat_file"})
            continue
        label_name, segment_number = _parse_clip_filename(fname, patient_id)
        if label_name == "test":
            exclusions.append({"path": path, "reason": "unlabeled_test_clip"})
            continue
        segment_key = (label_name, segment_number)
        if segment_key in seen_segments:
            raise ValueError(
                f"Duplicate {label_name} segment {segment_number} for P{patient_id}"
            )
        seen_segments.add(segment_key)
        results.append((path, LABEL_BY_NAME[label_name]))
    if return_exclusions:
        return results, exclusions
    return results


def scan_all_patients(exclude_ids=None, data_root=None):
    exclude_ids = set(exclude_ids or [])
    data = {}
    for pid in PATIENT_IDS:
        if pid in exclude_ids:
            continue
        files = scan_patient_files(pid, data_root=data_root)
        if files:
            data[pid] = files
    return data


# ============================================================
# 特征提取
# ============================================================

N_FEATURES_PER_CH = 4 + len(BAND_DEFS)  # 10
SEGMENT_PATTERN = re.compile(r"_segment_(\d+)\.mat$", re.IGNORECASE)
INTERICTAL_GROUP_SECONDS = 300


def _segment_number(path):
    match = SEGMENT_PATTERN.search(os.path.basename(path))
    if not match:
        raise ValueError(f"Cannot parse segment number from {path}")
    return int(match.group(1))


def build_patient_group_manifest(
    patient_id,
    interictal_group_seconds=INTERICTAL_GROUP_SECONDS,
    interictal_grouping="patient_supergroup",
    return_exclusions=False,
):
    """Describe leakage-resistant groups for one patient's one-second clips.

    Ictal episode boundaries come from latency resets. Interictal clips use the
    entire patient as a conservative supergroup unless a simulation explicitly
    requests proxy blocks. The supergroup can never be split across partitions.
    """
    records = []
    scanned_files, exclusions = scan_patient_files(
        patient_id, return_exclusions=True
    )
    items = sorted(scanned_files, key=lambda item: (
        item[1], _segment_number(item[0])
    ))
    ictal_episode = 0
    previous_latency = None
    for path, label in items:
        segment_number = _segment_number(path)
        if label == 1:
            metadata = scipy.io.loadmat(path, variable_names=["latency"])
            if "latency" not in metadata:
                raise ValueError(f"Ictal clip has no latency metadata: {path}")
            latency = float(metadata["latency"].flat[0])
            if not np.isfinite(latency):
                raise ValueError(f"Ictal clip has invalid latency metadata: {path}")
            if previous_latency is None or latency <= previous_latency:
                ictal_episode += 1
            previous_latency = latency
            group_id = f"P{patient_id}:ictal_episode:{ictal_episode:03d}"
            group_source = "mat_latency_reset"
        else:
            if interictal_grouping == "patient_supergroup":
                group_id = f"P{patient_id}:interictal_patient_supergroup"
                group_source = "patient_supergroup_recording_id_unavailable"
            elif interictal_grouping == "proxy_blocks":
                block = (segment_number - 1) // interictal_group_seconds + 1
                group_id = f"P{patient_id}:interictal_block:{block:04d}"
                group_source = (
                    f"simulation_proxy_contiguous_{interictal_group_seconds}s_"
                    "recording_id_unavailable"
                )
            else:
                raise ValueError(
                    "interictal_grouping must be 'patient_supergroup' or "
                    "'proxy_blocks'"
                )
            latency = None
        records.append({
            "path": path,
            "patient_id": int(patient_id),
            "label": int(label),
            "segment_number": segment_number,
            "latency": latency,
            "group_id": group_id,
            "group_source": group_source,
        })
    if return_exclusions:
        return records, exclusions
    return records


def patient_disjoint_train_validation_split(records, val_ratio=0.15, seed=42):
    """Split entire patients so no recording can cross train/validation."""
    by_patient: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_patient.setdefault(int(record["patient_id"]), []).append(record)
    patient_ids = sorted(by_patient)
    if len(patient_ids) < 2:
        raise ValueError("Patient-disjoint validation requires at least two patients")
    rng = random.Random(seed)
    rng.shuffle(patient_ids)
    target_samples = max(1, round(len(records) * val_ratio))
    validation_patients = set()
    selected_samples = 0
    for patient_id in patient_ids[:-1]:
        if selected_samples >= target_samples:
            break
        validation_patients.add(patient_id)
        selected_samples += len(by_patient[patient_id])
    train = [
        record for record in records
        if int(record["patient_id"]) not in validation_patients
    ]
    validation = [
        record for record in records
        if int(record["patient_id"]) in validation_patients
    ]
    for partition_name, partition in (("train", train), ("validation", validation)):
        if {record["label"] for record in partition} != {0, 1}:
            raise ValueError(f"{partition_name} partition must contain both labels")
    if (
        {int(record["patient_id"]) for record in train}
        & {int(record["patient_id"]) for record in validation}
    ):
        raise AssertionError("Training and validation patients overlap")
    return train, validation


def grouped_train_validation_split(records, val_ratio=0.15, seed=42):
    """Split whole groups while preserving both labels in train and validation."""
    rng = random.Random(seed)
    validation_groups = set()
    for label in (0, 1):
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            if record["label"] == label:
                groups.setdefault(record["group_id"], []).append(record)
        group_ids = list(groups)
        if len(group_ids) < 2:
            raise ValueError(f"Label {label} requires at least two independent groups")
        rng.shuffle(group_ids)
        target_samples = max(1, round(sum(map(len, groups.values())) * val_ratio))
        selected_samples = 0
        for group_id in group_ids[:-1]:
            if selected_samples >= target_samples:
                break
            validation_groups.add(group_id)
            selected_samples += len(groups[group_id])

    train = [record for record in records if record["group_id"] not in validation_groups]
    validation = [record for record in records if record["group_id"] in validation_groups]
    train_groups = {record["group_id"] for record in train}
    validation_group_ids = {record["group_id"] for record in validation}
    if train_groups & validation_group_ids:
        raise AssertionError("Training and validation groups overlap")
    return train, validation


def welch_band_powers(data, freq):
    """Estimate integrated band power from a detrended Hann-window Welch PSD."""
    data = np.asarray(data, dtype=np.float32)
    sample_count = data.shape[1]
    nperseg = min(int(PSD_METHOD["nperseg"]), sample_count)
    noverlap = min(int(PSD_METHOD["noverlap"]), nperseg - 1)
    frequencies, psd = welch(
        data,
        fs=float(freq),
        window=PSD_METHOD["window"],
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=PSD_METHOD["detrend"],
        scaling=PSD_METHOD["scaling"],
        average=PSD_METHOD["average"],
        axis=1,
    )
    if frequencies.size < 2:
        raise ValueError("Welch PSD requires at least two frequency bins")
    frequency_resolution = float(frequencies[1] - frequencies[0])
    powers = np.zeros((data.shape[0], len(BAND_DEFS)), dtype=np.float64)
    for index, (name, low, high) in enumerate(BAND_DEFS):
        mask = (frequencies >= low) & (frequencies < high)
        if not np.any(mask):
            raise ValueError(
                f"Frequency band {name} [{low}, {high}) Hz has no Welch bins "
                f"for {sample_count} samples at {freq:g} Hz"
            )
        powers[:, index] = psd[:, mask].sum(axis=1) * frequency_resolution
    return powers.astype(np.float32), frequencies, psd


def extract_features(data, freq):
    """从原始信号提取特征。
    
    Args:
        data: (C, T) numpy array, 原始信号（不做 z-score!）
        freq: 采样率
    
    Returns:
        features: (C, N_FEATURES) numpy array
    """
    data = np.asarray(data)
    if data.ndim != 2 or not np.issubdtype(data.dtype, np.number):
        raise ValueError("EEG data must be a numeric 2D array shaped (channels, time)")
    c, t = data.shape
    if c < 1 or t < 2:
        raise ValueError("EEG data requires at least one channel and two timepoints")
    if not np.isfinite(data).all():
        raise ValueError("EEG data contains NaN or infinite values")
    if not np.isfinite(freq) or freq <= 0:
        raise ValueError("Sampling frequency must be positive and finite")

    nyquist = float(freq) / 2.0
    highest_band_edge = max(hi for _, _, hi in BAND_DEFS)
    if nyquist < highest_band_edge:
        raise ValueError(
            f"Sampling rate {freq:g} Hz cannot represent all configured bands; "
            f"Nyquist must be at least {highest_band_edge:g} Hz"
        )

    data = data.astype(np.float32, copy=False)
    features = np.zeros((c, N_FEATURES_PER_CH), dtype=np.float32)

    # 1. log energy
    features[:, 0] = np.log1p((data ** 2).mean(axis=1))

    # 2. log abs mean
    features[:, 1] = np.log1p(np.abs(data).mean(axis=1))

    # 3. std
    features[:, 2] = np.log1p(data.std(axis=1))

    # 4. line length (波形复杂度)
    features[:, 3] = np.log1p(np.abs(np.diff(data, axis=1)).mean(axis=1))

    # 5-10. 频段 log power
    band_powers, _, _ = welch_band_powers(data, freq)
    features[:, 4:] = np.log1p(band_powers)

    if not np.isfinite(features).all():
        raise ValueError("Feature extraction produced NaN or infinite values")
    return features  # (C, 10)


def _matlab_text(value) -> str:
    current = value
    while isinstance(current, np.ndarray) and current.size == 1:
        current = current.flat[0]
    return str(current).strip()


def _extract_channel_names(mat, channel_count):
    raw = mat.get("channels")
    names = []
    if isinstance(raw, np.ndarray) and raw.dtype.names:
        record = raw.flat[0]
        for field_name in raw.dtype.names:
            text = _matlab_text(record[field_name])
            names.append(text or field_name.strip("X."))
    elif isinstance(raw, np.ndarray):
        names = [_matlab_text(value) for value in raw.flat]
    if len(names) != channel_count or len(set(names)) != len(names):
        names = [f"CH_{index + 1:03d}" for index in range(channel_count)]
    return names


def _optional_mat_text(mat, *keys):
    for key in keys:
        if key in mat:
            value = _matlab_text(mat[key])
            if value:
                return value
    return None


def _optional_mat_float(mat, *keys):
    for key in keys:
        if key in mat:
            values = np.asarray(mat[key]).reshape(-1)
            if values.size != 1:
                raise ValueError(f"MAT {key} must contain exactly one value")
            value = float(values[0])
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"MAT {key} must be positive and finite")
            return value
    return None


def _signal_contract(mat):
    source_unit = _optional_mat_text(mat, "unit", "units")
    reference = _optional_mat_text(mat, "reference", "reference_method")
    gain = _optional_mat_float(mat, "gain", "amplifier_gain")
    microvolts_per_count = _optional_mat_float(
        mat, "microvolts_per_count", "uv_per_count"
    )
    normalized_unit = (source_unit or "").lower().replace("μ", "u").replace("µ", "u")
    if normalized_unit in {"uv", "microvolt", "microvolts"}:
        model_unit = "uV"
        conversion_scale = 1.0
    elif normalized_unit in {"adc", "adc_count", "adc_counts", "count", "counts"}:
        if microvolts_per_count is None:
            raise ValueError(
                "ADC-count EEG requires microvolts_per_count metadata"
            )
        model_unit = "uV"
        conversion_scale = microvolts_per_count
    elif source_unit is None:
        model_unit = "source_native_unknown"
        conversion_scale = 1.0
    else:
        raise ValueError(f"Unsupported EEG amplitude unit: {source_unit}")

    missing = []
    if source_unit is None:
        missing.append("unit")
    if gain is None:
        missing.append("gain")
    if reference is None:
        missing.append("reference")
    return {
        "metadata_status": "complete" if not missing else "legacy_unspecified",
        "source_unit": source_unit or "unspecified",
        "model_unit": model_unit,
        "gain": gain,
        "microvolts_per_count": microvolts_per_count,
        "reference": reference or "unspecified",
        "missing_metadata": missing,
        "conversion_scale": conversion_scale,
    }


def assess_signal_quality(data, model_unit="source_native_unknown"):
    """Return deterministic clip-level QC without assuming unknown source units."""
    values = np.asarray(data, dtype=np.float64)
    channel_std = values.std(axis=1)
    channel_range = np.ptp(values, axis=1)
    flat = (channel_std <= 1e-8) | (channel_range <= 1e-7)

    minimum = values.min(axis=1, keepdims=True)
    maximum = values.max(axis=1, keepdims=True)
    at_rails = np.isclose(values, minimum) | np.isclose(values, maximum)
    clipped = (at_rails.mean(axis=1) >= 0.05) & ~flat

    differences = np.diff(values, axis=1)
    diff_center = np.median(differences, axis=1, keepdims=True)
    diff_mad = 1.4826 * np.median(
        np.abs(differences - diff_center), axis=1
    )
    largest_step = np.max(np.abs(differences), axis=1)
    abrupt = largest_step > np.maximum(100.0 * diff_mad, 1e-6)

    extreme = np.zeros(values.shape[0], dtype=bool)
    if model_unit == "uV":
        extreme = np.max(np.abs(values), axis=1) > 5000.0

    critical_fraction = max(float(flat.mean()), float(clipped.mean()))
    flags = []
    if flat.any():
        flags.append(f"flat_channels:{int(flat.sum())}")
    if clipped.any():
        flags.append(f"clipped_channels:{int(clipped.sum())}")
    if abrupt.any():
        flags.append(f"abrupt_step_channels:{int(abrupt.sum())}")
    if extreme.any():
        flags.append(f"extreme_amplitude_channels:{int(extreme.sum())}")

    passed = critical_fraction < 0.25
    status = "pass" if not flags else "warning"
    if not passed:
        status = "reject"
    return {
        "status": status,
        "passed": passed,
        "flags": flags,
        "channel_count": int(values.shape[0]),
        "flat_channel_count": int(flat.sum()),
        "clipped_channel_count": int(clipped.sum()),
        "abrupt_step_channel_count": int(abrupt.sum()),
        "extreme_amplitude_channel_count": int(extreme.sum()),
        "max_absolute_amplitude": float(np.max(np.abs(values))),
    }


def load_mat_raw(mat_path, return_channel_names=False, return_metadata=False):
    """加载 .mat 文件，返回原始数据、采样率和可选电极名称。"""
    mat = scipy.io.loadmat(mat_path)
    if "data" not in mat or "freq" not in mat:
        raise ValueError("MAT file must contain data and freq variables")
    data = mat["data"]
    if data.ndim != 2:
        raise ValueError("MAT data must be a two-dimensional matrix")
    if data.shape[0] > data.shape[1]:
        data = data.T
    contract = _signal_contract(mat)
    data = data.astype(np.float32) * float(contract.pop("conversion_scale"))
    freq = float(mat["freq"].flat[0])
    if not np.isfinite(data).all() or not np.isfinite(freq) or freq <= 0:
        raise ValueError("MAT data and freq must contain finite values")
    channel_names = _extract_channel_names(mat, data.shape[0])
    metadata = {
        **contract,
        "channel_count": int(data.shape[0]),
        "channel_names": channel_names,
    }
    if return_metadata:
        return data, freq, metadata
    if return_channel_names:
        return data, freq, channel_names
    return data, freq


def load_and_extract(mat_path, return_metadata=False):
    """按固定时间契约加载所有通道并提取逐通道特征。

    Returns: ``(C, 10)``，不截断、不补零、不依赖通道顺序。
    """
    data, freq, signal_metadata = load_mat_raw(mat_path, return_metadata=True)
    c, t = data.shape

    if not np.isclose(freq, TARGET_SR, atol=1.0):
        raise ValueError(f"Expected {TARGET_SR} Hz EEG, got {freq:g} Hz")
    if t != TARGET_POINTS:
        duration = t / freq
        raise ValueError(
            f"Expected a 1-second window with {TARGET_POINTS} timepoints; "
            f"got {t} points ({duration:.6g} seconds)"
        )

    quality = assess_signal_quality(data, signal_metadata["model_unit"])
    if not quality["passed"]:
        raise ValueError(
            "EEG signal quality check failed: " + ", ".join(quality["flags"])
        )

    features = extract_features(data, freq)
    band_power_unit = (
        "uV^2"
        if signal_metadata["model_unit"] == "uV"
        else "source_native_unit^2"
    )
    if return_metadata:
        return features, {
            **signal_metadata,
            "quality": quality,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "band_schema_version": BAND_SCHEMA_VERSION,
            "psd_method": PSD_METHOD,
            "band_power_unit": band_power_unit,
            "band_feature_transform": "log1p(power / 1 unit^2)",
            "channel_count": c,
            "channel_strategy": "all_channels_permutation_invariant",
        }
    return features


# ============================================================
# 数据增强（特征空间）
# ============================================================

def augment_features(features):
    """在特征空间做增强。"""
    features = features.copy()
    c = features.shape[0]

    # 1. 特征噪声（小幅扰动）
    noise = np.random.randn(*features.shape).astype(np.float32) * 0.05
    features += noise

    # 2. 通道 dropout（随机置零 1-2 个通道的所有特征）
    n_drop = np.random.randint(0, 3)
    if n_drop > 0:
        drop_idx = np.random.choice(c, n_drop, replace=False)
        features[drop_idx] = 0

    return features


# ============================================================
# Dataset
# ============================================================

class EEGFeatureDataset(Dataset):
    """特征数据集。每个样本为可变通道数的 ``(C, 10)`` 矩阵。"""

    def __init__(self, file_label_list, do_augment=False):
        self.samples = file_label_list
        self.do_augment = do_augment
        self._cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        if idx in self._cache:
            features = self._cache[idx].copy()
        else:
            features = load_and_extract(path)
            self._cache[idx] = features

        if self.do_augment:
            features = augment_features(features)

        x = torch.from_numpy(features)
        return x, label


def collate_channel_sets(batch):
    """Keep variable channel sets as a list instead of zero-padding."""
    features, labels = zip(*batch)
    return list(features), torch.tensor(labels, dtype=torch.long)


def attach_manifest_provenance(records, dataset_manifest, data_root=None):
    """Join split records to immutable file hashes and reject excluded clips."""
    root = Path(data_root or DATA_ROOT).resolve()
    provenance_by_path = {
        str((root / row["relative_path"]).resolve()): row
        for row in dataset_manifest["records"]
    }
    accepted: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for record in records:
        resolved_path = str(Path(record["path"]).resolve())
        provenance = provenance_by_path.get(resolved_path)
        if provenance is None:
            raise ValueError(f"Split record missing from dataset manifest: {resolved_path}")
        if provenance["exclusion_reasons"]:
            exclusions.extend({
                "patient_id": int(record["patient_id"]),
                "path": resolved_path,
                "reason": reason,
            } for reason in provenance["exclusion_reasons"])
            continue
        accepted.append({
            **record,
            "path": resolved_path,
            "relative_path": provenance["relative_path"],
            "file_sha256": provenance["sha256"],
            "session_id": provenance["session_id"],
            "episode_id": provenance["episode_id"],
            "quality_status": provenance["quality_status"],
            "quality_flags": provenance["quality_flags"],
        })
    return accepted, exclusions


# ============================================================
# LOPO 数据集构建
# ============================================================

def build_lopo_datasets(
    target_patient_id,
    val_ratio=0.15,
    augment_train=True,
    return_manifest=False,
    dataset_manifest=None,
):
    from .data_manifest import (  # Local import avoids the scanner/manifest cycle.
        build_dataset_manifest,
        validate_split_group_disjoint,
    )

    data_root = Path(DATA_ROOT).resolve()
    if dataset_manifest is None:
        dataset_manifest = build_dataset_manifest(data_root)
    target_records, target_exclusions = build_patient_group_manifest(
        target_patient_id, return_exclusions=True
    )
    source_records = []
    scan_exclusions = [
        {"patient_id": int(target_patient_id), **item}
        for item in target_exclusions
    ]
    for patient_id in PATIENT_IDS:
        if patient_id == target_patient_id:
            continue
        patient_records, patient_exclusions = build_patient_group_manifest(
            patient_id, return_exclusions=True
        )
        source_records.extend(patient_records)
        scan_exclusions.extend(
            {"patient_id": int(patient_id), **item}
            for item in patient_exclusions
        )

    source_records, source_manifest_exclusions = attach_manifest_provenance(
        source_records, dataset_manifest, data_root
    )
    target_records, target_manifest_exclusions = attach_manifest_provenance(
        target_records, dataset_manifest, data_root
    )
    scan_exclusions.extend(source_manifest_exclusions)
    scan_exclusions.extend(target_manifest_exclusions)
    train_records, val_records = patient_disjoint_train_validation_split(
        source_records, val_ratio=val_ratio, seed=42
    )
    train_files_final = [
        (record["path"], record["label"]) for record in train_records
    ]
    val_files = [(record["path"], record["label"]) for record in val_records]
    random.Random(42).shuffle(train_files_final)
    random.Random(43).shuffle(val_files)

    n_ictal = sum(1 for _, l in train_files_final if l == 1)
    n_interictal = sum(1 for _, l in train_files_final if l == 0)

    train_ds = EEGFeatureDataset(train_files_final, do_augment=augment_train)
    val_ds = EEGFeatureDataset(val_files, do_augment=False)
    test_files = [(record["path"], record["label"]) for record in target_records]
    test_ds = EEGFeatureDataset(test_files, do_augment=False)

    result = (train_ds, val_ds, test_ds, n_ictal, n_interictal)
    if return_manifest:
        manifest = {
            "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
            "evaluation_protocol": EVALUATION_PROTOCOL_VERSION,
            "dataset_manifest_schema": dataset_manifest["schema_version"],
            "dataset_manifest_id": dataset_manifest["manifest_sha256"],
            "filename_schema": (
                "Patient_<id>_(ictal|interictal|test)_segment_<n>.mat"
            ),
            "seed": 42,
            "validation_ratio": val_ratio,
            "target_patient_id": target_patient_id,
            "model_protocol": "cross_patient_base_v1",
            "validation_protocol": "source_patient_disjoint_calibration_v3",
            "group_provenance": "patient_supergroup_v1",
            "validation_patient_ids": sorted({
                record["patient_id"] for record in val_records
            }),
            "interictal_grouping": (
                "All interictal clips from a patient form one conservative "
                "supergroup because source recording IDs are unavailable."
            ),
            "scan_exclusions": scan_exclusions,
            "records": [
                {**record, "split": "train"} for record in train_records
            ] + [
                {**record, "split": "calibration"} for record in val_records
            ] + [
                {**record, "split": "test"} for record in target_records
            ],
        }
        manifest["group_sets"] = validate_split_group_disjoint(
            manifest["records"]
        )
        return (*result, manifest)
    return result
