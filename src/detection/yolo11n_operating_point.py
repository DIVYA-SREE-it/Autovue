from pathlib import Path
import csv
import json

import numpy as np
import matplotlib.pyplot as plt

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

MIN_RECALL = 0.95


def main():

    print("=" * 72)
    print("M17B2 — YOLO11N OPERATING-POINT SELECTION")
    print("=" * 72)

    print("\nValidation split only.")
    print("Image size: 640")
    print("NMS IoU: 0.70")
    print("Confidence floor: 0.001")
    print("Test set: NOT USED")

    model = YOLO(
        str(MODEL_PATH)
    )

    metrics = model.val(
        data=str(DATA_YAML),

        split="val",

        imgsz=640,

        conf=0.001,
        iou=0.70,

        batch=8,

        device=0,
        workers=4,

        plots=True,
        verbose=True,

        project=str(
            OUTPUT_DIR
            / "operating_point"
        ),

        name="curves",

        exist_ok=True,
    )

    print("\nAvailable curves:")

    for name in metrics.curves:
        print(" -", name)

    curve_data = {}

    for name, result in zip(
        metrics.curves,
        metrics.curves_results,
    ):

        x, y, x_label, y_label = result

        x = np.asarray(x)
        y = np.asarray(y)

        # Single class in this project.
        # Still handle multi-class arrays safely.
        if y.ndim == 2:
            y = y.mean(axis=0)

        curve_data[name] = {
            "x": x,
            "y": y,
            "x_label": x_label,
            "y_label": y_label,
        }

    f1_key = next(
        key
        for key in curve_data
        if "F1-Confidence" in key
    )

    p_key = next(
        key
        for key in curve_data
        if "Precision-Confidence" in key
    )

    r_key = next(
        key
        for key in curve_data
        if "Recall-Confidence" in key
    )

    confidence = curve_data[f1_key]["x"]

    f1_values = curve_data[f1_key]["y"]
    precision_values = curve_data[p_key]["y"]
    recall_values = curve_data[r_key]["y"]

    # Ensure all arrays align.
    length = min(
        len(confidence),
        len(f1_values),
        len(precision_values),
        len(recall_values),
    )

    confidence = confidence[:length]
    f1_values = f1_values[:length]
    precision_values = precision_values[:length]
    recall_values = recall_values[:length]

    acceptable = np.where(
        recall_values >= MIN_RECALL
    )[0]

    if len(acceptable) == 0:

        selected_index = int(
            np.argmax(f1_values)
        )

        reason = (
            "No confidence point maintained "
            f"recall >= {MIN_RECALL:.2f}; "
            "selected maximum F1."
        )

    else:

        acceptable_f1 = (
            f1_values[
                acceptable
            ]
        )

        max_f1 = (
            acceptable_f1.max()
        )

        f1_candidates = acceptable[
            np.isclose(
                acceptable_f1,
                max_f1,
            )
        ]

        if len(f1_candidates) > 1:

            selected_index = int(
                f1_candidates[
                    np.argmax(
                        precision_values[
                            f1_candidates
                        ]
                    )
                ]
            )

        else:

            selected_index = int(
                f1_candidates[0]
            )

        reason = (
            "Highest F1 while maintaining "
            f"recall >= {MIN_RECALL:.2f}."
        )

    selected = {
        "confidence":
            float(
                confidence[
                    selected_index
                ]
            ),

        "precision":
            float(
                precision_values[
                    selected_index
                ]
            ),

        "recall":
            float(
                recall_values[
                    selected_index
                ]
            ),

        "f1":
            float(
                f1_values[
                    selected_index
                ]
            ),
    }

    # ========================================================
    # SAVE FULL CURVE
    # ========================================================

    csv_path = (
        OUTPUT_DIR
        / "operating_point_curve.csv"
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
                "confidence",
                "precision",
                "recall",
                "f1",
            ],
        )

        writer.writeheader()

        for i in range(length):

            writer.writerow({
                "confidence":
                    float(
                        confidence[i]
                    ),

                "precision":
                    float(
                        precision_values[i]
                    ),

                "recall":
                    float(
                        recall_values[i]
                    ),

                "f1":
                    float(
                        f1_values[i]
                    ),
            })

    # ========================================================
    # SAVE SELECTED POINT
    # ========================================================

    json_path = (
        OUTPUT_DIR
        / "selected_operating_point.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                **selected,
                "minimum_recall_rule":
                    MIN_RECALL,
                "reason":
                    reason,
            },
            file,
            indent=2,
        )

    # ========================================================
    # PLOT
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        confidence,
        precision_values,
        label="Precision",
        linewidth=2,
    )

    plt.plot(
        confidence,
        recall_values,
        label="Recall",
        linewidth=2,
    )

    plt.plot(
        confidence,
        f1_values,
        label="F1",
        linewidth=2,
    )

    plt.axvline(
        selected["confidence"],
        linestyle="--",
        linewidth=1.5,
        label=(
            "Selected confidence "
            f"{selected['confidence']:.3f}"
        ),
    )

    plt.xlabel(
        "Confidence Threshold"
    )

    plt.ylabel(
        "Metric"
    )

    plt.ylim(
        0,
        1.02,
    )

    plt.title(
        "YOLO11n Precision / Recall / F1 "
        "vs Confidence"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "operating_point_curve.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # PRINT
    # ========================================================

    print("\n" + "=" * 72)
    print("SELECTED OPERATING POINT")
    print("=" * 72)

    print(
        f"Confidence : "
        f"{selected['confidence']:.4f}"
    )

    print(
        f"Precision  : "
        f"{selected['precision']:.4f}"
    )

    print(
        f"Recall     : "
        f"{selected['recall']:.4f}"
    )

    print(
        f"F1         : "
        f"{selected['f1']:.4f}"
    )

    print(
        "\nReason:",
        reason,
    )

    print(
        "\nSaved:",
        json_path,
    )


if __name__ == "__main__":
    main()