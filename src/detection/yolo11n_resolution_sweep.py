from pathlib import Path
import csv
import json

from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT
    / "experiments"
    / "detector_benchmark"
    / "yolo11n"
    / "weights"
    / "best.pt"
)

DATA_YAML = (
    ROOT
    / "data"
    / "processed"
    / "npds_leakage_safe"
    / "dataset.yaml"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "M17_yolo11n_resolution_sweep"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONTROLLED RESOLUTION EXPERIMENT
# ============================================================

RESOLUTIONS = [
    640,
    960,
    1280,
]

# Same batch for all resolutions so the comparison stays
# consistent. Batch 2 is conservative for the 6 GB RTX 4050.
BATCH = 2


def calculate_f1(
    precision,
    recall,
):

    if precision + recall == 0:
        return 0.0

    return (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
    )


def main():

    print("=" * 72)
    print("M17A — YOLO11N RESOLUTION SWEEP")
    print("=" * 72)

    print("\nModel:")
    print(MODEL_PATH)

    print("\nDataset:")
    print(DATA_YAML)

    print(
        "\nIMPORTANT:"
        "\nValidation split only."
        "\nNo test-set tuning."
        "\nNo retraining in this experiment."
    )

    results = []

    for imgsz in RESOLUTIONS:

        print(
            "\n"
            + "=" * 72
        )

        print(
            f"VALIDATING AT "
            f"{imgsz} × {imgsz}"
        )

        print(
            "=" * 72
        )

        model = YOLO(
            str(MODEL_PATH)
        )

        metrics = model.val(
            data=str(DATA_YAML),

            split="val",

            imgsz=imgsz,

            batch=BATCH,

            device=0,

            workers=4,

            plots=False,

            verbose=True,

            project=str(
                OUTPUT_DIR
                / "val_runs"
            ),

            name=(
                f"imgsz_{imgsz}"
            ),

            exist_ok=True,
        )

        precision = float(
            metrics.box.mp
        )

        recall = float(
            metrics.box.mr
        )

        f1 = calculate_f1(
            precision,
            recall,
        )

        map50 = float(
            metrics.box.map50
        )

        map50_95 = float(
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
                0.0,
            )
        )

        fps = (
            1000.0
            / inference_ms
            if inference_ms > 0
            else 0.0
        )

        result = {
            "imgsz": imgsz,

            "precision":
                round(
                    precision,
                    6,
                ),

            "recall":
                round(
                    recall,
                    6,
                ),

            "f1":
                round(
                    f1,
                    6,
                ),

            "map50":
                round(
                    map50,
                    6,
                ),

            "map50_95":
                round(
                    map50_95,
                    6,
                ),

            "inference_ms":
                round(
                    inference_ms,
                    4,
                ),

            "fps_theoretical":
                round(
                    fps,
                    2,
                ),
        }

        results.append(
            result
        )

        print(
            "\nRESULT"
        )

        for key, value in result.items():
            print(
                f"{key}: {value}"
            )


    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_path = (
        OUTPUT_DIR
        / "resolution_results.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "imgsz",
                "precision",
                "recall",
                "f1",
                "map50",
                "map50_95",
                "inference_ms",
                "fps_theoretical",
            ],
        )

        writer.writeheader()
        writer.writerows(
            results
        )


    # ========================================================
    # SAVE JSON
    # ========================================================

    json_path = (
        OUTPUT_DIR
        / "resolution_results.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )


    # ========================================================
    # AUTOMATIC COMPARISON
    # ========================================================

    baseline = next(
        row
        for row in results
        if row["imgsz"] == 640
    )

    best_map = max(
        results,
        key=lambda row:
            row["map50_95"],
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "M17A SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        "\nBaseline 640:"
    )

    print(
        f"mAP50-95: "
        f"{baseline['map50_95']:.4f}"
    )

    print(
        f"Inference: "
        f"{baseline['inference_ms']:.2f} ms"
    )

    print(
        "\nBest strict localization:"
    )

    print(
        f"{best_map['imgsz']} × "
        f"{best_map['imgsz']}"
    )

    print(
        f"mAP50-95: "
        f"{best_map['map50_95']:.4f}"
    )

    improvement = (
        best_map["map50_95"]
        - baseline["map50_95"]
    )

    print(
        f"\nAbsolute mAP50-95 gain "
        f"vs 640: "
        f"{improvement:+.4f}"
    )

    print(
        "\nSaved:"
    )

    print(
        csv_path
    )

    print(
        json_path
    )


if __name__ == "__main__":
    main()