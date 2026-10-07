from pathlib import Path
import argparse
import csv
import json
import time

import torch
from ultralytics import YOLO, RTDETR


ROOT = Path(__file__).resolve().parents[2]

DATA_YAML = (
    ROOT
    / "data"
    / "processed"
    / "npds_leakage_safe"
    / "dataset.yaml"
)

FULL_PROJECT = (
    ROOT
    / "experiments"
    / "detector_benchmark"
)

SMOKE_PROJECT = (
    ROOT
    / "experiments"
    / "detector_benchmark_smoke"
)


MODELS = {
    "yolov8n": {
        "weights": "yolov8n.pt",
        "type": "yolo",
        "batch": 8,
    },

    "yolo11n": {
        "weights": "yolo11n.pt",
        "type": "yolo",
        "batch": 8,
    },

    "yolo11s": {
        "weights": "yolo11s.pt",
        "type": "yolo",
        "batch": 8,
    },

    "rtdetr_l": {
        "weights": "rtdetr-l.pt",
        "type": "rtdetr",
        "batch": 2,
    },
}


def create_model(config, weights=None):

    model_path = (
        weights
        if weights is not None
        else config["weights"]
    )

    if config["type"] == "rtdetr":
        return RTDETR(str(model_path))

    return YOLO(str(model_path))


def best_epoch_from_csv(csv_path):

    if not csv_path.exists():
        return -1

    with open(
        csv_path,
        "r",
        encoding="utf-8",
    ) as file:

        rows = list(
            csv.DictReader(file)
        )

    if not rows:
        return -1

    metric_column = None

    for column in rows[0]:

        if "mAP50-95" in column:
            metric_column = column
            break

    if metric_column is None:
        return -1

    best_index = max(
        range(len(rows)),
        key=lambda i: float(
            rows[i].get(
                metric_column,
                0
            )
            or 0
        )
    )

    return best_index + 1


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=MODELS.keys(),
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    config = MODELS[
        args.model
    ]

    project = (
        SMOKE_PROJECT
        if args.smoke
        else FULL_PROJECT
    )

    run_name = args.model

    epochs = (
        1
        if args.smoke
        else 50
    )

    fraction = (
        0.02
        if args.smoke
        else 1.0
    )

    print("=" * 72)
    print("INDIAN PLATE DETECTOR BENCHMARK")
    print("=" * 72)

    print("Model:", args.model)
    print("Weights:", config["weights"])
    print("Epochs:", epochs)
    print("Image size: 640")
    print("Batch:", config["batch"])
    print("Seed: 42")
    print("Dataset:", DATA_YAML)
    print("Smoke:", args.smoke)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = create_model(
        config
    )

    parameters = sum(
        p.numel()
        for p in model.model.parameters()
    )

    deterministic = (
        False
        if config["type"] == "rtdetr"
        else True
    )

    start_time = time.perf_counter()

    model.train(
        data=str(DATA_YAML),

        epochs=epochs,
        imgsz=640,

        batch=config["batch"],
        nbs=64,

        device=0,
        workers=4,

        seed=42,
        deterministic=deterministic,

        pretrained=True,
        amp=True,

        patience=10,

        fraction=fraction,

        cache=False,

        project=str(project),
        name=run_name,
        exist_ok=True,

        plots=True,
        verbose=True,
    )

    training_seconds = (
        time.perf_counter()
        - start_time
    )

    run_dir = (
        project
        / run_name
    )

    best_weights = (
        run_dir
        / "weights"
        / "best.pt"
    )

    if not best_weights.exists():

        raise FileNotFoundError(
            f"Best weights not found: "
            f"{best_weights}"
        )

    peak_vram_gb = (
        torch.cuda.max_memory_reserved()
        / (1024 ** 3)
    )

    print("\n" + "=" * 72)
    print("VALIDATING BEST MODEL")
    print("=" * 72)

    best_model = create_model(
        config,
        weights=best_weights,
    )

    validation_split = (
        "val"
        if args.smoke
        else "test"
    )

    metrics = best_model.val(
        data=str(DATA_YAML),
        split=validation_split,

        imgsz=640,
        batch=config["batch"],

        device=0,
        workers=4,

        plots=True,
        verbose=True,
    )

    precision = float(
        metrics.box.mp
    )

    recall = float(
        metrics.box.mr
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    map50 = float(
        metrics.box.map50
    )

    map5095 = float(
        metrics.box.map
    )

    speed = getattr(
        metrics,
        "speed",
        {}
    )

    inference_ms = float(
        speed.get(
            "inference",
            0.0
        )
    )

    fps = (
        1000.0 / inference_ms
        if inference_ms > 0
        else 0.0
    )

    model_size_mb = (
        best_weights.stat().st_size
        / (1024 ** 2)
    )

    best_epoch = (
        best_epoch_from_csv(
            run_dir
            / "results.csv"
        )
    )

    summary = {
        "model": args.model,
        "pretrained_weights": config[
            "weights"
        ],

        "split": validation_split,

        "epochs_requested": epochs,
        "best_epoch": best_epoch,

        "imgsz": 640,
        "batch": config["batch"],
        "seed": 42,

        "precision": round(
            precision,
            6
        ),

        "recall": round(
            recall,
            6
        ),

        "f1": round(
            f1,
            6
        ),

        "map50": round(
            map50,
            6
        ),

        "map50_95": round(
            map5095,
            6
        ),

        "inference_ms_per_image": round(
            inference_ms,
            4
        ),

        "fps_theoretical": round(
            fps,
            2
        ),

        "parameters": parameters,

        "model_size_mb": round(
            model_size_mb,
            2
        ),

        "peak_reserved_vram_gb": round(
            peak_vram_gb,
            3
        ),

        "training_seconds": round(
            training_seconds,
            2
        ),

        "training_minutes": round(
            training_seconds / 60,
            2
        ),

        "best_weights": str(
            best_weights
        ),
    }

    summary_path = (
        run_dir
        / "benchmark_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\n" + "=" * 72)
    print("BENCHMARK RESULT")
    print("=" * 72)

    for key, value in summary.items():
        print(
            f"{key}: {value}"
        )

    print(
        "\nSaved:",
        summary_path
    )


if __name__ == "__main__":
    main()