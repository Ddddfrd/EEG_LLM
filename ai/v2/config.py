"""V2 配置：固定通道 EEGNet"""
import torch

from project_config import DATA_ROOT as PROJECT_DATA_ROOT, checkpoint_dir

# === 路径 ===
DATA_ROOT = PROJECT_DATA_ROOT
CHECKPOINT_DIR = checkpoint_dir("v2")

# === 设备 ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 病人 ===
PATIENT_IDS = [2, 3, 4, 5, 6, 7, 8]  # 排除 P1（500Hz 异类）

# === 数据参数 ===
TARGET_CHANNELS = 16   # 固定通道数（P8 = 16，最小值）
TARGET_SR = 5000       # 所有 P2-P8 都是 5000Hz
TARGET_POINTS = 5000   # 1 秒 × 5000Hz

# === Phase 1: 基座训练 ===
PHASE1 = {
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 80,
    "early_stop_patience": 15,
    "val_ratio": 0.15,
    "augment": True,
    "class_weight_seizure": 3.0,  # CE loss 中发作类权重
}

# === Phase 2: Setup + 在线学习 ===
PHASE2 = {
    "setup_normal_samples": 50,
    "setup_source_seizures": 100,
    "setup_epochs": 5,
    "setup_replay_per_batch": 8,
    "stream_lr": 1e-3,
    "stream_threshold": 0.7,
    "stream_replay_size": 16,
    "buffer_max_seizure": 200,
    "buffer_max_normal": 500,
}
