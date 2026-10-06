from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_YAML = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "npds_yolo"
    / "dataset.yaml"
)

RUN_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "plate_detector"
)


def main():

    print("=" * 60)
    print("INDIAN ANPR - NUMBER PLATE DETECTOR TRAINING")
    print("=" * 60)

    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found:\n{DATASET_YAML}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Training stopped."
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print("Dataset:", DATASET_YAML)

    model = YOLO("yolo11n.pt")

    model.train(
    data=str(DATASET_YAML),

    epochs=30,
    imgsz=640,

    batch=8,
    device=0,

    workers=4,

    patience=7,

    project=str(RUN_DIR),
    name="npds_yolo11n",

    exist_ok=True,

    amp=True,
    cache=False,

    plots=True
)


if __name__ == "__main__":
    main()