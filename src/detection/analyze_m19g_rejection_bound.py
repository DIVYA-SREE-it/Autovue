from pathlib import Path
import csv
import json


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "M19_road_validation"
)

FRAME_CSV = (
    OUTPUT_DIR
    / "M19G_frame_review.csv"
)


def to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def metrics(tp, fp, fn):

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return precision, recall, f1


def main():

    print("=" * 72)
    print("M19G — ADAPTED MODEL REJECTION ANALYSIS")
    print("=" * 72)

    with open(
        FRAME_CSV,
        newline="",
        encoding="utf-8",
    ) as file:

        rows = list(
            csv.DictReader(file)
        )

    total_visible = sum(
        to_int(
            row["visible_plates"]
        )
        for row in rows
    )

    total_detections = sum(
        to_int(
            row["detections"]
        )
        for row in rows
    )


    # ========================================================
    # GUARANTEED FALSE POSITIVES
    # ========================================================

    no_plate_detection_rows = [
        row
        for row in rows
        if (
            to_int(
                row["visible_plates"]
            ) == 0
            and
            to_int(
                row["detections"]
            ) > 0
        )
    ]

    guaranteed_fp = sum(
        to_int(
            row["detections"]
        )
        for row in no_plate_detection_rows
    )


    # ========================================================
    # ZERO-DETECTION FALSE NEGATIVES
    # ========================================================

    zero_detection_rows = [
        row
        for row in rows
        if to_int(
            row["detections"]
        ) == 0
    ]

    guaranteed_fn = sum(
        to_int(
            row["visible_plates"]
        )
        for row in zero_detection_rows
    )


    # ========================================================
    # OPTIMISTIC MAXIMUM TP
    #
    # Each remaining detection can produce at most one TP.
    # TP also cannot exceed visible plate count in its frame.
    # ========================================================

    max_tp = 0

    for row in rows:

        visible = to_int(
            row["visible_plates"]
        )

        detections = to_int(
            row["detections"]
        )

        max_tp += min(
            visible,
            detections,
        )


    min_fp = (
        total_detections
        - max_tp
    )

    min_fn = (
        total_visible
        - max_tp
    )


    best_p, best_r, best_f1 = (
        metrics(
            max_tp,
            min_fp,
            min_fn,
        )
    )


    # ========================================================
    # ORIGINAL ROAD-VALIDATION BASELINE
    # ========================================================

    baseline = {
        "tp": 16,
        "fp": 3,
        "fn": 17,
        "precision": 0.8421,
        "recall": 0.4848,
        "f1": 0.6154,
    }


    # ========================================================
    # DECISION
    # ========================================================

    reject = (
        best_p
        < baseline["precision"]
        and
        best_r
        < baseline["recall"]
        and
        best_f1
        < baseline["f1"]
    )


    result = {
        "frames":
            len(rows),

        "visible_plates":
            total_visible,

        "adapted_detector_boxes":
            total_detections,

        "guaranteed_false_positives":
            guaranteed_fp,

        "guaranteed_false_negatives_zero_detection":
            guaranteed_fn,

        "optimistic_upper_bound": {
            "tp":
                max_tp,

            "fp":
                min_fp,

            "fn":
                min_fn,

            "precision":
                round(
                    best_p,
                    6,
                ),

            "recall":
                round(
                    best_r,
                    6,
                ),

            "f1":
                round(
                    best_f1,
                    6,
                ),
        },

        "original_baseline":
            baseline,

        "adapted_model_rejected":
            reject,
    }


    json_path = (
        OUTPUT_DIR
        / "M19G_results.json"
    )

    json_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )


    summary = f"""M19G — DOMAIN-ADAPTATION ROAD VALIDATION
============================================================

Evaluation video
----------------
traffic_ocr.mp4

Training use:
NONE

Reviewed/sample frames:
{len(rows)}

Human-confirmed visible plates:
{total_visible}

Adapted-model detections:
{total_detections}

Guaranteed errors
-----------------
False positives in frames containing zero real plates:
{guaranteed_fp}

False negatives from zero-detection frames:
{guaranteed_fn}

Optimistic adapted-model upper bound
------------------------------------
This assumes every remaining detection that could possibly
correspond to a real plate is correct.

TP <= {max_tp}
FP >= {min_fp}
FN >= {min_fn}

Precision <= {best_p:.4f}
Recall    <= {best_r:.4f}
F1        <= {best_f1:.4f}

Original YOLO11n road-validation baseline
-----------------------------------------
TP = 16
FP = 3
FN = 17

Precision = 0.8421
Recall    = 0.4848
F1        = 0.6154

Decision
--------
Adapted model:
{"REJECTED" if reject else "REQUIRES MANUAL REVIEW"}

Reason:
Even its mathematically optimistic upper-bound performance
is below the original YOLO11n road-validation baseline.

NPDS result
-----------
The conservative adaptation preserved NPDS performance well:

Original:
P = 0.9654
R = 0.9672
F1 = 0.9663
mAP50-95 = 0.7109

Adapted:
P = 0.9613
R = 0.9687
F1 = 0.9650
mAP50-95 = 0.6942

Interpretation
--------------
Low-learning-rate fine-tuning prevented severe catastrophic
forgetting, but the small 12-frame road adaptation set was
insufficient for reliable cross-domain improvement.

Final detector decision
-----------------------
Retain the original leakage-safe YOLO11n detector as the
primary ANPR detector.

The M19 adapted checkpoint is retained only as an
experimental domain-adaptation result.

Important methodology
---------------------
traffic_ocr.mp4 has now been used for model-selection
evaluation and must NOT be described as the final road test.

A new unseen road video should be used for the final
end-to-end evaluation.
"""

    summary_path = (
        OUTPUT_DIR
        / "M19G_summary.txt"
    )

    summary_path.write_text(
        summary,
        encoding="utf-8",
    )


    print()
    print(
        "Visible plates          :",
        total_visible,
    )

    print(
        "Adapted detections      :",
        total_detections,
    )

    print(
        "Guaranteed FP           :",
        guaranteed_fp,
    )

    print(
        "Guaranteed zero-det FN  :",
        guaranteed_fn,
    )

    print("\nOptimistic upper bound:")

    print(
        f"TP <= {max_tp}"
    )

    print(
        f"FP >= {min_fp}"
    )

    print(
        f"FN >= {min_fn}"
    )

    print(
        f"Precision <= {best_p:.4f}"
    )

    print(
        f"Recall    <= {best_r:.4f}"
    )

    print(
        f"F1        <= {best_f1:.4f}"
    )

    print()

    print(
        "Decision:",
        (
            "REJECT ADAPTED MODEL"
            if reject
            else "MANUAL REVIEW REQUIRED"
        ),
    )

    print("\nSaved:")
    print(summary_path)
    print(json_path)


if __name__ == "__main__":
    main()