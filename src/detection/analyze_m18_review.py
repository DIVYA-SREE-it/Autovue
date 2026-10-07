from pathlib import Path
import csv
import json
from collections import defaultdict


ROOT = Path(__file__).resolve().parents[2]

M18_DIR = (
    ROOT
    / "outputs"
    / "M18_real_road_eval"
)

REVIEW_CSV = (
    M18_DIR
    / "manual_review.csv"
)


def to_int(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


def safe_div(a, b):
    return a / b if b else 0.0


def calculate_metrics(tp, fp, fn):

    precision = safe_div(
        tp,
        tp + fp,
    )

    recall = safe_div(
        tp,
        tp + fn,
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return (
        precision,
        recall,
        f1,
    )


def main():

    print("=" * 72)
    print("M18C — REAL-ROAD DETECTOR EVALUATION")
    print("=" * 72)

    with open(
        REVIEW_CSV,
        newline="",
        encoding="utf-8",
    ) as file:

        rows = list(
            csv.DictReader(file)
        )

    reviewed = [
        row
        for row in rows
        if row.get(
            "reviewed",
            ""
        ) == "1"
    ]

    usable = [
        row
        for row in reviewed
        if row.get(
            "exclude_from_metrics",
            ""
        ) != "1"
    ]

    if not usable:

        raise RuntimeError(
            "No reviewed rows available."
        )

    print(
        f"\nTotal sampled frames : "
        f"{len(rows)}"
    )

    print(
        f"Reviewed frames      : "
        f"{len(reviewed)}"
    )

    print(
        f"Used for metrics     : "
        f"{len(usable)}"
    )

    print(
        f"Coverage             : "
        f"{100 * len(reviewed) / len(rows):.1f}%"
    )


    # ========================================================
    # VALIDATE REVIEW DATA
    # ========================================================

    errors = []

    for row in usable:

        detections = to_int(
            row.get("detections")
        )

        visible = to_int(
            row.get(
                "visible_plate_count"
            )
        )

        tp = to_int(
            row.get(
                "true_positives"
            )
        )

        fp = to_int(
            row.get(
                "false_positives"
            )
        )

        fn = to_int(
            row.get(
                "false_negatives"
            )
        )

        if tp + fp != detections:

            errors.append(
                (
                    row["sample_id"],
                    "TP + FP != detections",
                )
            )

        if tp + fn != visible:

            errors.append(
                (
                    row["sample_id"],
                    "TP + FN != visible plates",
                )
            )

    if errors:

        print(
            "\nWARNING — inconsistent review rows:"
        )

        for sample_id, message in errors[:20]:

            print(
                sample_id,
                message,
            )

        raise RuntimeError(
            "Fix review inconsistencies "
            "before calculating metrics."
        )


    # ========================================================
    # GLOBAL METRICS
    # ========================================================

    total_tp = sum(
        to_int(
            row["true_positives"]
        )
        for row in usable
    )

    total_fp = sum(
        to_int(
            row["false_positives"]
        )
        for row in usable
    )

    total_fn = sum(
        to_int(
            row["false_negatives"]
        )
        for row in usable
    )

    visible_plates = sum(
        to_int(
            row.get(
                "visible_plate_count"
            )
        )
        for row in usable
    )

    detector_boxes = sum(
        to_int(
            row.get(
                "detections"
            )
        )
        for row in usable
    )

    precision, recall, f1 = (
        calculate_metrics(
            total_tp,
            total_fp,
            total_fn,
        )
    )


    # ========================================================
    # PER-VIDEO
    # ========================================================

    video_data = defaultdict(
        lambda: {
            "frames": 0,
            "visible": 0,
            "detections": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
    )

    for row in usable:

        video = row["video"]

        data = video_data[
            video
        ]

        data["frames"] += 1

        data["visible"] += to_int(
            row.get(
                "visible_plate_count"
            )
        )

        data["detections"] += to_int(
            row.get(
                "detections"
            )
        )

        data["tp"] += to_int(
            row.get(
                "true_positives"
            )
        )

        data["fp"] += to_int(
            row.get(
                "false_positives"
            )
        )

        data["fn"] += to_int(
            row.get(
                "false_negatives"
            )
        )


    # ========================================================
    # ERROR FRAME LIST
    # ========================================================

    error_rows = []

    for row in usable:

        fp = to_int(
            row.get(
                "false_positives"
            )
        )

        fn = to_int(
            row.get(
                "false_negatives"
            )
        )

        if fp > 0 or fn > 0:

            error_rows.append({
                "sample_id":
                    row["sample_id"],

                "video":
                    row["video"],

                "timestamp_sec":
                    row["timestamp_sec"],

                "visible_plates":
                    row.get(
                        "visible_plate_count",
                        ""
                    ),

                "detections":
                    row.get(
                        "detections",
                        ""
                    ),

                "true_positives":
                    row.get(
                        "true_positives",
                        ""
                    ),

                "false_positives":
                    fp,

                "false_negatives":
                    fn,
            })


    error_path = (
        M18_DIR
        / "M18_error_frames.csv"
    )

    with open(
        error_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fields = [
            "sample_id",
            "video",
            "timestamp_sec",
            "visible_plates",
            "detections",
            "true_positives",
            "false_positives",
            "false_negatives",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            error_rows
        )


    # ========================================================
    # PER-VIDEO CSV
    # ========================================================

    per_video_rows = []

    for video, data in video_data.items():

        p, r, video_f1 = (
            calculate_metrics(
                data["tp"],
                data["fp"],
                data["fn"],
            )
        )

        per_video_rows.append({
            "video":
                video,

            "reviewed_frames":
                data["frames"],

            "visible_plates":
                data["visible"],

            "detector_boxes":
                data["detections"],

            "tp":
                data["tp"],

            "fp":
                data["fp"],

            "fn":
                data["fn"],

            "precision":
                round(p, 6),

            "recall":
                round(r, 6),

            "f1":
                round(
                    video_f1,
                    6,
                ),
        })


    per_video_path = (
        M18_DIR
        / "M18_per_video_metrics.csv"
    )

    with open(
        per_video_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                per_video_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            per_video_rows
        )


    # ========================================================
    # GLOBAL JSON
    # ========================================================

    metrics = {
        "total_sampled_frames":
            len(rows),

        "reviewed_frames":
            len(reviewed),

        "metric_frames":
            len(usable),

        "review_coverage_percent":
            round(
                100
                * len(reviewed)
                / len(rows),
                2,
            ),

        "visible_real_plates":
            visible_plates,

        "detector_boxes":
            detector_boxes,

        "true_positives":
            total_tp,

        "false_positives":
            total_fp,

        "false_negatives":
            total_fn,

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

        "frames_with_errors":
            len(
                error_rows
            ),
    }


    metrics_path = (
        M18_DIR
        / "M18_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary_lines = [
        "M18 — REAL-ROAD YOLO11N EVALUATION",
        "=" * 60,
        "",
        "Evaluation configuration:",
        "Model: YOLO11n",
        "Input size: 640 x 640",
        "Confidence: 0.40",
        "NMS IoU: 0.70",
        "",
        "Human-review coverage:",
        (
            f"{len(reviewed)} / "
            f"{len(rows)} frames "
            f"({100 * len(reviewed) / len(rows):.1f}%)"
        ),
        "",
        "Ground-truth counts from reviewed frames:",
        f"Visible real plates : {visible_plates}",
        f"Detector boxes      : {detector_boxes}",
        "",
        "Detection outcomes:",
        f"True positives      : {total_tp}",
        f"False positives     : {total_fp}",
        f"False negatives     : {total_fn}",
        "",
        "Real-road metrics:",
        (
            f"Precision = "
            f"{precision:.4f}"
        ),
        (
            f"Recall    = "
            f"{recall:.4f}"
        ),
        (
            f"F1        = "
            f"{f1:.4f}"
        ),
        "",
        (
            f"Frames containing at least "
            f"one FP or FN: {len(error_rows)}"
        ),
        "",
        "Interpretation:",
        (
            "These results represent a human-reviewed "
            "real-road robustness audit, not NPDS test mAP."
        ),
        (
            "The road videos differ substantially from "
            "the training/benchmark domain and are used "
            "to expose deployment-domain failure modes."
        ),
        "",
        "Next milestone:",
        (
            "M19 — inspect FP/FN frames and perform "
            "hard-negative/domain-adaptation experiments."
        ),
    ]


    summary_path = (
        M18_DIR
        / "M18C_summary.txt"
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
    print("M18C RESULTS")
    print("=" * 72)

    print(
        f"\nVisible real plates : "
        f"{visible_plates}"
    )

    print(
        f"Detector boxes      : "
        f"{detector_boxes}"
    )

    print(
        f"\nTP : {total_tp}"
    )

    print(
        f"FP : {total_fp}"
    )

    print(
        f"FN : {total_fn}"
    )

    print(
        f"\nPrecision : "
        f"{precision:.4f}"
    )

    print(
        f"Recall    : "
        f"{recall:.4f}"
    )

    print(
        f"F1        : "
        f"{f1:.4f}"
    )

    print(
        f"\nError frames: "
        f"{len(error_rows)}"
    )

    print(
        "\nPer-video results:"
    )

    for row in per_video_rows:

        print(
            f"{row['video']:22s} | "
            f"P={row['precision']:.4f} | "
            f"R={row['recall']:.4f} | "
            f"F1={row['f1']:.4f}"
        )

    print(
        "\nSaved:"
    )

    print(
        metrics_path
    )

    print(
        per_video_path
    )

    print(
        error_path
    )

    print(
        summary_path
    )


if __name__ == "__main__":
    main()