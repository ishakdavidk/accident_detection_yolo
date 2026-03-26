from ultralytics import YOLO
import os
from pathlib import Path

# ============================
# 1. FROZEN TRAINING (1024)
# ============================
def train_yolo11_frozen(epochs_frozen=20):
    model = YOLO('yolo11s.pt')  # Load pretrained YOLOv11 small model

    config = {
        'data': 'dataset/dataset.yaml',
        'epochs': epochs_frozen,
        'imgsz': 1024,                     # <<< UPDATED
        'batch': 8,                        # <<< Suggested for 1024; adjust if OOM
        'patience': epochs_frozen,
        'device': '0',
        'workers': 4,

        # Output folder now indicates 1024
        'project': 'runs/train',
        'name': 'accident_detection_yolo11_frozen_1024',    # <<< UPDATED
        'exist_ok': True,

        'pretrained': True,
        'optimizer': 'auto',
        'lr0': 5e-4,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 5e-4,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 1,
        'dfl': 1.5,
        'nbs': 64,
        'dropout': 0.,
        'val': True,
        'freeze': 8,   # Freeze backbone for warm-up stage
        'save_period': 1,
    }

    results = model.train(**config)

    # Path to last checkpoint
    out_folder = Path("runs/train/accident_detection_yolo11_frozen_1024/weights")
    last_model_path = out_folder / "last.pt"

    return last_model_path, results


# ============================
# 2. UNFROZEN TRAINING (1024)
# ============================
def train_yolo11_unfrozen(starting_model, epochs_unfrozen=80):
    model = YOLO(str(starting_model))

    config = {
        'data': 'dataset/dataset.yaml',
        'epochs': epochs_unfrozen,
        'imgsz': 1024,                   # <<< UPDATED
        'batch': 8,                      # <<< Suggested for 1024
        'patience': epochs_unfrozen,
        'device': '0',
        'workers': 4,

        'project': 'runs/train',
        'name': 'accident_detection_yolo11_finetune_1024',   # <<< UPDATED
        'exist_ok': True,

        'pretrained': False,
        'optimizer': 'auto',
        'lr0': 1e-4,      # Lower LR for fine-tuning
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 5e-4,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 1,
        'dfl': 1.5,
        'nbs': 64,
        'dropout': 0.,
        'val': True,
        'freeze': 0,        # Unfreeze all layers now
        'save_period': 1
    }

    results = model.train(**config)

    # Best checkpoint (if saved)
    best_model_path = Path("runs/train/accident_detection_yolo11_finetune_1024/weights/best.pt")
    if best_model_path.exists():
        print(f"Best model saved at: {best_model_path}")

    return results


# ============================
# MAIN
# ============================
if __name__ == '__main__':
    if not os.path.exists('dataset/dataset.yaml'):
        print("Please run prepare_dota.py first to prepare the dataset!")
        exit(1)

    frozen_epochs = 30
    unfrozen_epochs = 80

    # Step 1 — Frozen stage (warm-up)
    last_model_path, _ = train_yolo11_frozen(epochs_frozen=frozen_epochs)

    # Step 2 — Unfrozen full fine-tuning
    results = train_yolo11_unfrozen(starting_model=last_model_path, epochs_unfrozen=unfrozen_epochs)

    print("\nTraining completed!")
    print(f"Best mAP50: {results.box.map50:.4f}")
    print(f"Best mAP50-95: {results.box.map:.4f}")
