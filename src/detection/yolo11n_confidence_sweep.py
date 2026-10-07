from pathlib import Path
import csv
import json

from ultralytics import YOLO


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
    / "M17_yolo11n_threshold_tuning"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONTROLLED VARIABLES
# ============================================================

IMAGE_SIZE = 640

# Keep NMS IoU fixed during confidence experiment.
NMS_IOU = 0.70

BATCH = 8

CONFIDENCE_THRESHOLDS = [
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
]


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
    print("M17B — YOLO11N CONFIDENCE THRESHOLD SWEEP")
    print("=" * 72)

    print("\nModel:")
    print(MODEL_PATH)

    print("\nDataset:")
    print(DATA_YAML)

    print("\nControlled settings:")
    print("Split       : validation")
    print("Image size  : 640")
    print("NMS IoU     : 0.70")
    print("Batch       : 8")
    print("Test set    : NOT USED")

    results = []

    for conf in CONFIDENCE_THRESHOLDS:

        print("\n" + "=" * 72)
        print(
            f"CONFIDENCE THRESHOLD = "
            f"{conf:.2f}"
        )
        print("=" * 72)

        model = YOLO(
            str(MODEL_PATH)
        )

        metrics = model.val(
            data=str(DATA_YAML),

            split="val",

            imgsz=IMAGE_SIZE,

            conf=conf,
            iou=NMS_IOU,

            batch=BATCH,

            device=0,

            workers=4,

            plots=False,
            verbose=False,

            project=str(
                OUTPUT_DIR
                / "confidence_runs"
            ),

            name=(
                f"conf_{conf:.2f}"
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
                0.0
            )
        )

        result = {
            "confidence":
                conf,

            "precision":
                round(
                    precision,
                    6
                ),

            "recall":
                round(
                    recall,
                    6
                ),

            "f1":
                round(
                    f1,
                    6
                ),

            "map50":
                round(
                    map50,
                    6
                ),

            "map50_95":
                round(
                    map50_95,
                    6
                ),

            "inference_ms":
                round(
                    inference_ms,
                    4
                ),
        }

        results.append(
            result
        )

        print(
            f"P={precision:.4f} | "
            f"R={recall:.4f} | "
            f"F1={f1:.4f} | "
            f"mAP50={map50:.4f} | "
            f"mAP50-95={map50_95:.4f}"
        )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    csv_path = (
        OUTPUT_DIR
        / "confidence_results.csv"
    )

    fields = [
        "confidence",
        "precision",
        "recall",
        "f1",
        "map50",
        "map50_95",
        "inference_ms",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            results
        )


    json_path = (
        OUTPUT_DIR
        / "confidence_results.json"
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
    # SELECT OPERATING POINT
    # ========================================================

    # ANPR priority:
    # keep recall >= 0.95 if possible,
    # then maximize F1.
    acceptable = [
        row
        for row in results
        if row["recall"] >= 0.95
    ]

    if acceptable:

        selected = max(
            acceptable,
            key=lambda row: (
                row["f1"],
                row["precision"],
            ),
        )

        selection_reason = (
            "Highest F1 among thresholds "
            "maintaining recall >= 0.95."
        )

    else:

        selected = max(
            results,
            key=lambda row:
                row["f1"],
        )

        selection_reason = (
            "Highest F1 because no threshold "
            "maintained recall >= 0.95."
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary_lines = [
        "M17B — YOLO11N CONFIDENCE THRESHOLD SWEEP",
        "=" * 60,
        "",
        "Objective:",
        (
            "Find a practical YOLO11n confidence threshold "
            "using the leakage-safe validation split."
        ),
        "",
        "Controlled settings:",
        "Image size: 640",
        "NMS IoU: 0.70",
        "Batch: 8",
        "Test set: NOT USED",
        "",
        "Results:",
    ]

    for row in results:

        summary_lines.append(
            (
                f"conf={row['confidence']:.2f} | "
                f"P={row['precision']:.4f} | "
                f"R={row['recall']:.4f} | "
                f"F1={row['f1']:.4f} | "
                f"mAP50={row['map50']:.4f} | "
                f"mAP50-95={row['map50_95']:.4f}"
            )
        )

    summary_lines += [
        "",
        "Selected operating point:",
        (
            f"Confidence = "
            f"{selected['confidence']:.2f}"
        ),
        (
            f"Precision = "
            f"{selected['precision']:.4f}"
        ),
        (
            f"Recall = "
            f"{selected['recall']:.4f}"
        ),
        (
            f"F1 = "
            f"{selected['f1']:.4f}"
        ),
        "",
        "Selection rule:",
        selection_reason,
        "",
        "Important methodology note:",
        (
            "Confidence tuning is performed on validation data only. "
            "The held-out test set is not used to choose the threshold."
        ),
    ]

    summary_path = (
        OUTPUT_DIR
        / "M17B_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(
                summary_lines
            )
        )


    print("\n" + "=" * 72)
    print("M17B SUMMARY")
    print("=" * 72)

    print(
        "\nSelected confidence:",
        selected["confidence"],
    )

    print(
        "Precision:",
        selected["precision"],
    )

    print(
        "Recall:",
        selected["recall"],
    )

    print(
        "F1:",
        selected["f1"],
    )

    print(
        "\nReason:",
        selection_reason,
    )

    print(
        "\nSaved:",
        csv_path,
    )

    print(
        summary_path
    )


if __name__ == "__main__":
    main()