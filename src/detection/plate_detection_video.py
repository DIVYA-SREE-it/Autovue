from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_VIDEO = (
    PROJECT_ROOT
    / "data"
    / "test_videos"
    / "traffic_baseline.mp4"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "plate"
    / "npds_yolo11n_best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "M04_plate_detection"
)

OUTPUT_VIDEO = OUTPUT_DIR / "plate_detection.mp4"
CROP_DIR = OUTPUT_DIR / "plate_crops"


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE = 0.25

# Larger inference size helps because plates
# are small in road videos.
IMAGE_SIZE = 960


def main():

    print("=" * 60)
    print("INDIAN ANPR - NUMBER PLATE DETECTION")
    print("=" * 60)

    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(
            f"Video not found:\n{INPUT_VIDEO}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CROP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    device = 0 if torch.cuda.is_available() else "cpu"

    print("Model :", MODEL_PATH)
    print("Video :", INPUT_VIDEO)

    if torch.cuda.is_available():
        print(
            "GPU   :",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = YOLO(str(MODEL_PATH))

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(INPUT_VIDEO)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open input video."
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print(
        f"Resolution: {width}x{height}"
    )

    print(
        f"FPS       : {fps:.2f}"
    )

    print(
        f"Frames    : {total_frames}"
    )

    # --------------------------------------------------------
    # OUTPUT VIDEO
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Could not create output video."
        )

    frame_number = 0
    total_detections = 0
    saved_crops = 0

    # --------------------------------------------------------
    # FRAME LOOP
    # --------------------------------------------------------

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        results = model.predict(
            source=frame,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device=device,
            verbose=False
        )

        result = results[0]

        annotated = result.plot()

        boxes = result.boxes

        current_detections = len(boxes)

        total_detections += current_detections

        # ----------------------------------------------------
        # SAVE PLATE CROPS
        # ----------------------------------------------------

        for detection_index, box in enumerate(boxes):

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            confidence = float(
                box.conf[0]
            )

            # Clamp coordinates
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            plate_crop = frame[
                y1:y2,
                x1:x2
            ]

            if plate_crop.size == 0:
                continue

            crop_name = (
                f"frame_{frame_number:05d}_"
                f"plate_{detection_index}_"
                f"conf_{confidence:.2f}.jpg"
            )

            crop_path = (
                CROP_DIR
                / crop_name
            )

            cv2.imwrite(
                str(crop_path),
                plate_crop
            )

            saved_crops += 1

        # ----------------------------------------------------
        # OVERLAY
        # ----------------------------------------------------

        cv2.rectangle(
            annotated,
            (15, 15),
            (470, 125),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            annotated,
            "Indian ANPR - Plate Detection",
            (30, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated,
            f"Frame: {frame_number}/{total_frames}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated,
            f"Plates: {current_detections}",
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )

        writer.write(
            annotated
        )

        if frame_number % 100 == 0:

            print(
                f"Processed "
                f"{frame_number}/{total_frames} "
                f"| detections={total_detections}"
            )

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    cap.release()
    writer.release()

    print()
    print("=" * 60)
    print("PLATE DETECTION COMPLETE")
    print("=" * 60)

    print(
        "Frames processed :",
        frame_number
    )

    print(
        "Total detections :",
        total_detections
    )

    print(
        "Plate crops saved:",
        saved_crops
    )

    print(
        "\nVideo:",
        OUTPUT_VIDEO
    )

    print(
        "\nCrops:",
        CROP_DIR
    )


if __name__ == "__main__":
    main()