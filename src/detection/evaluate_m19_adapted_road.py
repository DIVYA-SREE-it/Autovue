from pathlib import Path
import csv

import cv2
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT
    / "experiments"
    / "domain_adaptation"
    / "yolo11n_m19_safe"
    / "weights"
    / "best.pt"
)

M18_DIR = (
    ROOT
    / "outputs"
    / "M18_real_road_eval"
)

M18_REVIEW = (
    M18_DIR
    / "manual_review.csv"
)

RAW_DIR = (
    M18_DIR
    / "raw_frames"
    / "traffic_ocr"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "M19_road_validation"
)

DETECTED_DIR = (
    OUTPUT_DIR
    / "detected_frames"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DETECTED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DETECTOR CONFIGURATION
# Same operating point as M18 baseline.
# ============================================================

IMAGE_SIZE = 640
CONFIDENCE = 0.40
NMS_IOU = 0.70


def to_int(value):

    try:
        return int(value)

    except (ValueError, TypeError):
        return 0


def find_raw_image(sample_id):

    matches = list(
        RAW_DIR.glob(
            f"{sample_id}_*.jpg"
        )
    )

    if len(matches) != 1:

        raise RuntimeError(
            f"{sample_id}: "
            f"expected one raw frame, "
            f"found {len(matches)}"
        )

    return matches[0]


def main():

    print("=" * 72)
    print("M19G1 — ADAPTED MODEL ROAD VALIDATION")
    print("=" * 72)

    print("\nModel:")
    print(MODEL_PATH)

    print("\nOperating configuration:")
    print(f"Image size : {IMAGE_SIZE}")
    print(f"Confidence : {CONFIDENCE}")
    print(f"NMS IoU    : {NMS_IOU}")

    print(
        "\nRoad-validation source:"
    )

    print(
        "traffic_ocr.mp4 sampled frames ONLY"
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            MODEL_PATH
        )


    # ========================================================
    # LOAD EXISTING HUMAN GROUND-TRUTH COUNTS
    # ========================================================

    with open(
        M18_REVIEW,
        newline="",
        encoding="utf-8",
    ) as file:

        all_rows = list(
            csv.DictReader(file)
        )


    ground_truth_rows = [
        row
        for row in all_rows

        if (
            row["video"]
            == "traffic_ocr.mp4"

            and row.get(
                "reviewed",
                ""
            )
            == "1"
        )
    ]


    if len(ground_truth_rows) != 91:

        raise RuntimeError(
            f"Expected 91 reviewed traffic_ocr frames, "
            f"found {len(ground_truth_rows)}"
        )


    # ========================================================
    # MODEL
    # ========================================================

    model = YOLO(
        str(MODEL_PATH)
    )


    frame_rows = []
    box_rows = []

    total_boxes = 0
    zero_detection_frames = 0
    automatic_fn = 0


    # ========================================================
    # INFERENCE
    # ========================================================

    for index, row in enumerate(
        ground_truth_rows,
        start=1,
    ):

        sample_id = row[
            "sample_id"
        ]

        visible = to_int(
            row.get(
                "visible_plate_count"
            )
        )

        image_path = find_raw_image(
            sample_id
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            raise RuntimeError(
                f"Could not read "
                f"{image_path}"
            )


        result = model.predict(

            source=image,

            imgsz=IMAGE_SIZE,

            conf=CONFIDENCE,

            iou=NMS_IOU,

            device=0,

            verbose=False,

        )[0]


        detections = (
            len(result.boxes)
            if result.boxes is not None
            else 0
        )

        total_boxes += detections


        # ----------------------------------------------------
        # Frames with zero detections need NO manual review.
        # TP=0, FP=0 and FN=visible plates.
        # ----------------------------------------------------

        if detections == 0:

            zero_detection_frames += 1

            automatic_fn += visible

            tp = 0
            fp = 0
            fn = visible

            needs_review = 0

        else:

            tp = ""
            fp = ""
            fn = ""

            needs_review = 1


            annotated = (
                result.plot()
            )

            cv2.imwrite(
                str(
                    DETECTED_DIR
                    / f"{sample_id}.jpg"
                ),
                annotated,
            )


        frame_rows.append({

            "sample_id":
                sample_id,

            "video":
                "traffic_ocr.mp4",

            "timestamp_sec":
                row[
                    "timestamp_sec"
                ],

            "visible_plates":
                visible,

            "detections":
                detections,

            "true_positives":
                tp,

            "false_positives":
                fp,

            "false_negatives":
                fn,

            "needs_manual_review":
                needs_review,

            "reviewed":
                0
                if needs_review
                else 1,

        })


        # ----------------------------------------------------
        # SAVE BOX DETAILS
        # ----------------------------------------------------

        if (
            detections > 0
            and result.boxes is not None
        ):

            boxes = (
                result.boxes.xyxy
                .detach()
                .cpu()
                .numpy()
            )

            confidences = (
                result.boxes.conf
                .detach()
                .cpu()
                .numpy()
            )


            for detection_id, (
                box,
                confidence
            ) in enumerate(
                zip(
                    boxes,
                    confidences,
                ),
                start=1,
            ):

                x1, y1, x2, y2 = [
                    float(value)
                    for value in box
                ]

                box_rows.append({

                    "sample_id":
                        sample_id,

                    "detection_id":
                        detection_id,

                    "confidence":
                        round(
                            float(
                                confidence
                            ),
                            6,
                        ),

                    "x1":
                        round(x1, 2),

                    "y1":
                        round(y1, 2),

                    "x2":
                        round(x2, 2),

                    "y2":
                        round(y2, 2),

                })


        if (
            index % 20 == 0
            or index == 1
        ):

            print(
                f"Processed "
                f"{index}/"
                f"{len(ground_truth_rows)}"
            )


    # ========================================================
    # FRAME SUMMARY CSV
    # ========================================================

    frame_path = (
        OUTPUT_DIR
        / "M19G_frame_review.csv"
    )

    frame_fields = [
        "sample_id",
        "video",
        "timestamp_sec",
        "visible_plates",
        "detections",
        "true_positives",
        "false_positives",
        "false_negatives",
        "needs_manual_review",
        "reviewed",
    ]

    with open(
        frame_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=frame_fields,
        )

        writer.writeheader()
        writer.writerows(
            frame_rows
        )


    # ========================================================
    # BOX CSV
    # ========================================================

    box_path = (
        OUTPUT_DIR
        / "M19G_detection_boxes.csv"
    )

    box_fields = [
        "sample_id",
        "detection_id",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ]

    with open(
        box_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=box_fields,
        )

        writer.writeheader()
        writer.writerows(
            box_rows
        )


    # ========================================================
    # REVIEW-NEEDED CSV
    # ========================================================

    review_rows = [
        row
        for row in frame_rows
        if row[
            "needs_manual_review"
        ] == 1
    ]

    review_path = (
        OUTPUT_DIR
        / "M19G_review_needed.csv"
    )

    with open(
        review_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=frame_fields,
        )

        writer.writeheader()
        writer.writerows(
            review_rows
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    total_visible = sum(
        to_int(
            row[
                "visible_plates"
            ]
        )
        for row in frame_rows
    )

    summary = f"""M19G1 — ADAPTED MODEL ROAD VALIDATION
============================================================

Model:
{MODEL_PATH}

Evaluation source:
traffic_ocr.mp4

Training use:
NONE

Sampled frames:
{len(frame_rows)}

Previously human-confirmed visible plates:
{total_visible}

Detector configuration:
Image size = {IMAGE_SIZE}
Confidence = {CONFIDENCE}
NMS IoU = {NMS_IOU}

Adapted-model detector boxes:
{total_boxes}

Frames with zero detections:
{zero_detection_frames}

Frames requiring quick manual review:
{len(review_rows)}

False negatives already known from zero-detection frames:
{automatic_fn}

Important:
Only frames containing adapted-model detections require
manual correctness checking.

Frames with zero model detections are automatically:
TP = 0
FP = 0
FN = visible real plates.

Baseline traffic_ocr metrics:
Precision = 0.8421
Recall    = 0.4848
F1        = 0.6154
"""

    summary_path = (
        OUTPUT_DIR
        / "M19G1_summary.txt"
    )

    summary_path.write_text(
        summary,
        encoding="utf-8",
    )


    # ========================================================
    # PRINT
    # ========================================================

    print("\n" + "=" * 72)
    print("M19G1 COMPLETE")
    print("=" * 72)

    print(
        f"\nFrames evaluated       : "
        f"{len(frame_rows)}"
    )

    print(
        f"Visible real plates    : "
        f"{total_visible}"
    )

    print(
        f"Adapted detector boxes : "
        f"{total_boxes}"
    )

    print(
        f"Zero-detection frames  : "
        f"{zero_detection_frames}"
    )

    print(
        f"Need quick review      : "
        f"{len(review_rows)}"
    )

    print(
        "\nSaved:"
    )

    print(frame_path)
    print(box_path)
    print(review_path)
    print(summary_path)


if __name__ == "__main__":
    main()