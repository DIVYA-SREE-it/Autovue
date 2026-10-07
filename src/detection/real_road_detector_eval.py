from pathlib import Path
import csv
import json
import math

import cv2
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

VIDEO_DIR = (
    ROOT
    / "data"
    / "test_videos"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "M18_real_road_eval"
)

RAW_DIR = OUTPUT_DIR / "raw_frames"
DETECTED_DIR = OUTPUT_DIR / "detected_frames"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DETECTED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FINAL M17 DETECTOR SETTINGS
# ============================================================

IMAGE_SIZE = 640
CONFIDENCE = 0.40
NMS_IOU = 0.70

# Human-review sampling frequency.
SAMPLE_RATE_HZ = 2.0


VIDEOS = [
    "traffic_baseline.mp4",
    "traffic_ocr.mp4",
]


# ============================================================
# HELPERS
# ============================================================

def build_sample_indices(
    total_frames,
    fps,
):

    if fps <= 0:
        return []

    duration = (
        total_frames
        / fps
    )

    number_of_samples = int(
        math.floor(
            duration
            * SAMPLE_RATE_HZ
        )
    ) + 1

    indices = []

    for sample_number in range(
        number_of_samples
    ):

        timestamp = (
            sample_number
            / SAMPLE_RATE_HZ
        )

        frame_index = int(
            round(
                timestamp
                * fps
            )
        )

        if frame_index >= total_frames:
            continue

        indices.append(
            frame_index
        )

    # Remove any duplicate indices.
    return sorted(
        set(indices)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("M18A — REAL-ROAD YOLO11N DETECTOR AUDIT")
    print("=" * 72)

    print("\nFinal M17 configuration:")
    print(f"Model      : {MODEL_PATH}")
    print(f"Image size : {IMAGE_SIZE}")
    print(f"Confidence : {CONFIDENCE}")
    print(f"NMS IoU    : {NMS_IOU}")
    print(f"Sampling   : {SAMPLE_RATE_HZ} frames/sec")

    model = YOLO(
        str(MODEL_PATH)
    )

    prediction_rows = []
    review_rows = []
    video_summaries = []

    global_sample_number = 0

    for video_name in VIDEOS:

        video_path = (
            VIDEO_DIR
            / video_name
        )

        if not video_path.exists():

            print(
                f"\nWARNING: Missing video: "
                f"{video_path}"
            )

            continue

        print("\n" + "=" * 72)
        print(f"PROCESSING: {video_name}")
        print("=" * 72)

        cap = cv2.VideoCapture(
            str(video_path)
        )

        fps = float(
            cap.get(
                cv2.CAP_PROP_FPS
            )
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        sample_indices = (
            build_sample_indices(
                total_frames,
                fps,
            )
        )

        video_key = (
            Path(video_name).stem
        )

        raw_video_dir = (
            RAW_DIR
            / video_key
        )

        detected_video_dir = (
            DETECTED_DIR
            / video_key
        )

        raw_video_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        detected_video_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        total_detections = 0

        for local_number, frame_index in enumerate(
            sample_indices
        ):

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_index,
            )

            success, frame = cap.read()

            if not success:
                print(
                    f"Could not read frame "
                    f"{frame_index}"
                )
                continue

            global_sample_number += 1

            timestamp = (
                frame_index
                / fps
            )

            sample_id = (
                f"M18_"
                f"{global_sample_number:04d}"
            )

            filename = (
                f"{sample_id}"
                f"_f{frame_index:06d}.jpg"
            )

            raw_path = (
                raw_video_dir
                / filename
            )

            detected_path = (
                detected_video_dir
                / filename
            )

            cv2.imwrite(
                str(raw_path),
                frame,
            )

            result = model.predict(
                source=frame,

                imgsz=IMAGE_SIZE,

                conf=CONFIDENCE,

                iou=NMS_IOU,

                device=0,

                verbose=False,
            )[0]

            annotated = (
                result.plot()
            )

            cv2.imwrite(
                str(detected_path),
                annotated,
            )

            detections = 0

            if result.boxes is not None:

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

                detections = len(
                    boxes
                )

                total_detections += (
                    detections
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

                    box_width = max(
                        0.0,
                        x2 - x1,
                    )

                    box_height = max(
                        0.0,
                        y2 - y1,
                    )

                    box_area = (
                        box_width
                        * box_height
                    )

                    frame_area = (
                        width
                        * height
                    )

                    area_ratio = (
                        box_area
                        / frame_area
                        if frame_area > 0
                        else 0.0
                    )

                    prediction_rows.append({
                        "sample_id":
                            sample_id,

                        "video":
                            video_name,

                        "frame_index":
                            frame_index,

                        "timestamp_sec":
                            round(
                                timestamp,
                                3,
                            ),

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

                        "box_width":
                            round(
                                box_width,
                                2,
                            ),

                        "box_height":
                            round(
                                box_height,
                                2,
                            ),

                        "box_area_ratio":
                            round(
                                area_ratio,
                                8,
                            ),
                    })

            review_rows.append({
                "sample_id":
                    sample_id,

                "video":
                    video_name,

                "frame_index":
                    frame_index,

                "timestamp_sec":
                    round(
                        timestamp,
                        3,
                    ),

                "detections":
                    detections,

                "true_positives":
                    "",

                "false_positives":
                    "",

                "false_negatives":
                    "",

                "small_distant_misses":
                    "",

                "blur_misses":
                    "",

                "angle_misses":
                    "",

                "occlusion_misses":
                    "",

                "low_light_misses":
                    "",

                "text_sticker_fp":
                    "",

                "vehicle_body_fp":
                    "",

                "sign_watermark_fp":
                    "",

                "duplicate_detection_fp":
                    "",

                "notes":
                    "",
            })

            if (
                (local_number + 1) % 20 == 0
                or local_number == 0
            ):

                print(
                    f"Sampled "
                    f"{local_number + 1}/"
                    f"{len(sample_indices)} "
                    f"frames"
                )

        cap.release()

        video_summaries.append({
            "video":
                video_name,

            "resolution":
                f"{width}x{height}",

            "fps":
                round(
                    fps,
                    3,
                ),

            "total_frames":
                total_frames,

            "duration_sec":
                round(
                    total_frames
                    / fps,
                    3,
                )
                if fps > 0
                else 0,

            "sampled_frames":
                len(
                    sample_indices
                ),

            "detections":
                total_detections,
        })

        print(
            f"\nFinished {video_name}"
        )

        print(
            f"Sampled frames: "
            f"{len(sample_indices)}"
        )

        print(
            f"Detector boxes: "
            f"{total_detections}"
        )


    # ========================================================
    # PREDICTION CSV
    # ========================================================

    prediction_path = (
        OUTPUT_DIR
        / "predictions.csv"
    )

    prediction_fields = [
        "sample_id",
        "video",
        "frame_index",
        "timestamp_sec",
        "detection_id",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "box_width",
        "box_height",
        "box_area_ratio",
    ]

    with open(
        prediction_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=prediction_fields,
        )

        writer.writeheader()
        writer.writerows(
            prediction_rows
        )


    # ========================================================
    # HUMAN REVIEW TEMPLATE
    # ========================================================

    review_path = (
        OUTPUT_DIR
        / "manual_review.csv"
    )

    review_fields = list(
        review_rows[0].keys()
    )

    with open(
        review_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=review_fields,
        )

        writer.writeheader()
        writer.writerows(
            review_rows
        )


    # ========================================================
    # SUMMARY JSON
    # ========================================================

    summary_path = (
        OUTPUT_DIR
        / "sampling_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "configuration": {
                    "model":
                        "YOLO11n",

                    "imgsz":
                        IMAGE_SIZE,

                    "confidence":
                        CONFIDENCE,

                    "nms_iou":
                        NMS_IOU,

                    "sample_rate_hz":
                        SAMPLE_RATE_HZ,

                    "evaluation_type":
                        (
                            "human-reviewed "
                            "real-road robustness audit"
                        ),
                },

                "videos":
                    video_summaries,

                "total_sampled_frames":
                    len(
                        review_rows
                    ),

                "total_detector_boxes":
                    len(
                        prediction_rows
                    ),
            },
            file,
            indent=2,
        )


    # ========================================================
    # M18A SUMMARY
    # ========================================================

    summary_txt = (
        OUTPUT_DIR
        / "M18A_summary.txt"
    )

    lines = [
        "M18A — REAL-ROAD DETECTOR REVIEW SET",
        "=" * 60,
        "",
        "Purpose:",
        (
            "Evaluate the finalized YOLO11n detector "
            "outside the NPDS benchmark using actual "
            "1080p road traffic videos."
        ),
        "",
        "Detector configuration:",
        "YOLO11n",
        "Image size: 640",
        "Confidence: 0.40",
        "NMS IoU: 0.70",
        "",
        (
            f"Sampling frequency: "
            f"{SAMPLE_RATE_HZ} frames/second"
        ),
        "",
        "Videos:",
    ]

    for video in video_summaries:

        lines.append(
            (
                f"{video['video']} | "
                f"{video['duration_sec']:.2f}s | "
                f"{video['sampled_frames']} samples | "
                f"{video['detections']} detector boxes"
            )
        )

    lines += [
        "",
        (
            f"Total sampled frames: "
            f"{len(review_rows)}"
        ),

        (
            f"Total detector boxes: "
            f"{len(prediction_rows)}"
        ),

        "",
        "Important methodology:",
        (
            "These road videos do not yet have formal "
            "ground-truth bounding boxes."
        ),
        (
            "Therefore M18 uses human-reviewed TP/FP/FN "
            "rather than claiming mAP on unlabeled video."
        ),
        "",
        "Next:",
        (
            "Manually review raw and detected frames, "
            "record TP/FP/FN and categorize failure modes."
        ),
    ]

    summary_txt.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


    # ========================================================
    # FINAL PRINT
    # ========================================================

    print("\n" + "=" * 72)
    print("M18A COMPLETE")
    print("=" * 72)

    for video in video_summaries:

        print(
            f"{video['video']:22s} | "
            f"samples={video['sampled_frames']:3d} | "
            f"detections={video['detections']:3d}"
        )

    print(
        f"\nTotal sampled frames: "
        f"{len(review_rows)}"
    )

    print(
        f"Total detector boxes: "
        f"{len(prediction_rows)}"
    )

    print("\nReview file:")
    print(review_path)

    print("\nOutput directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()