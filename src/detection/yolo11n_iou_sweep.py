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


# ------------------------------------------------------------
# CONTROLLED SETTINGS
# ------------------------------------------------------------

IMAGE_SIZE = 640
BATCH = 8

# Low confidence floor so AP uses the full PR curve.
CONF_FLOOR = 0.001

NMS_IOUS = [
    0.30,
    0.50,
    0.70,
    0.80,
    0.90,
]


def main():

    print("=" * 72)
    print("M17C — YOLO11N NMS IoU SWEEP")
    print("=" * 72)

    print("\nValidation split only.")
    print("Image size       : 640")
    print("Confidence floor : 0.001")
    print("Batch            : 8")
    print("Test set         : NOT USED")

    results = []

    for nms_iou in NMS_IOUS:

        print("\n" + "=" * 72)

        print(
            f"NMS IoU = {nms_iou:.2f}"
        )

        print("=" * 72)

        model = YOLO(
            str(MODEL_PATH)
        )

        metrics = model.val(
            data=str(DATA_YAML),

            split="val",

            imgsz=IMAGE_SIZE,

            conf=CONF_FLOOR,
            iou=nms_iou,

            batch=BATCH,

            device=0,
            workers=4,

            plots=False,
            verbose=False,

            project=str(
                OUTPUT_DIR
                / "iou_runs"
            ),

            name=(
                f"iou_{nms_iou:.2f}"
            ),

            exist_ok=True,
        )

        precision = float(
            metrics.box.mp
        )

        recall = float(
            metrics.box.mr
        )

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
            if precision + recall > 0
            else 0.0
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

        preprocess_ms = float(
            speed.get(
                "preprocess",
                0.0
            )
        )

        inference_ms = float(
            speed.get(
                "inference",
                0.0
            )
        )

        postprocess_ms = float(
            speed.get(
                "postprocess",
                0.0
            )
        )

        result = {
            "nms_iou":
                nms_iou,

            "precision_at_best_f1":
                round(
                    precision,
                    6
                ),

            "recall_at_best_f1":
                round(
                    recall,
                    6
                ),

            "f1_best":
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

            "preprocess_ms":
                round(
                    preprocess_ms,
                    4
                ),

            "inference_ms":
                round(
                    inference_ms,
                    4
                ),

            "postprocess_ms":
                round(
                    postprocess_ms,
                    4
                ),
        }

        results.append(
            result
        )

        print(
            f"P*={precision:.4f} | "
            f"R*={recall:.4f} | "
            f"F1*={f1:.4f} | "
            f"mAP50={map50:.4f} | "
            f"mAP50-95={map50_95:.4f}"
        )


    # ========================================================
    # SELECT BEST NMS IoU
    # ========================================================

    selected = max(
        results,
        key=lambda row: (
            row["map50_95"],
            row["map50"],
            row["f1_best"],
        ),
    )


    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_path = (
        OUTPUT_DIR
        / "iou_results.csv"
    )

    fields = [
        "nms_iou",
        "precision_at_best_f1",
        "recall_at_best_f1",
        "f1_best",
        "map50",
        "map50_95",
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
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


    # ========================================================
    # SAVE JSON
    # ========================================================

    json_path = (
        OUTPUT_DIR
        / "iou_results.json"
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


    selected_path = (
        OUTPUT_DIR
        / "selected_nms_iou.json"
    )

    with open(
        selected_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "selected_nms_iou":
                    selected["nms_iou"],

                "selection_metric":
                    "validation mAP50-95",

                "map50":
                    selected["map50"],

                "map50_95":
                    selected["map50_95"],

                "note":
                    (
                        "Selected using validation data only. "
                        "Test set was not used."
                    ),
            },
            file,
            indent=2,
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary_lines = [
        "M17C — YOLO11N NMS IoU SWEEP",
        "=" * 60,
        "",
        "Objective:",
        (
            "Select the NMS IoU threshold using the "
            "leakage-safe validation split."
        ),
        "",
        "Controlled settings:",
        "Image size: 640",
        "Confidence floor: 0.001",
        "Batch: 8",
        "Test set: NOT USED",
        "",
        "Results:",
    ]

    for row in results:

        summary_lines.append(
            (
                f"IoU={row['nms_iou']:.2f} | "
                f"mAP50={row['map50']:.4f} | "
                f"mAP50-95={row['map50_95']:.4f} | "
                f"F1*={row['f1_best']:.4f}"
            )
        )

    summary_lines += [
        "",
        "Selected NMS IoU:",
        f"{selected['nms_iou']:.2f}",
        "",
        "Selection rule:",
        (
            "Highest validation mAP50-95; "
            "mAP50 and best-F1 used as tie-breakers."
        ),
        "",
        "Important:",
        (
            "P*, R* and F1* are curve-derived maximum-F1 "
            "summary values, not metrics at confidence 0.40."
        ),
        "",
        "Next:",
        (
            "Reconfirm the final confidence operating point "
            "using the selected NMS IoU."
        ),
    ]

    summary_path = (
        OUTPUT_DIR
        / "M17C_summary.txt"
    )

    summary_path.write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )


    # ========================================================
    # PRINT
    # ========================================================

    print("\n" + "=" * 72)
    print("M17C SUMMARY")
    print("=" * 72)

    for row in results:

        print(
            f"IoU {row['nms_iou']:.2f} | "
            f"mAP50={row['map50']:.4f} | "
            f"mAP50-95={row['map50_95']:.4f}"
        )

    print(
        "\nSelected NMS IoU:",
        selected["nms_iou"],
    )

    print(
        "Validation mAP50-95:",
        selected["map50_95"],
    )

    print(
        "\nSaved:",
        csv_path,
    )

    print(
        summary_path,
    )


if __name__ == "__main__":
    main()