from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_VIDEO = (
    PROJECT_ROOT
    / "data"
    / "test_videos"
    / "traffic_baseline.mp4"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "M02_vehicle_detection"
)

OUTPUT_VIDEO = OUTPUT_DIR / "vehicle_detection.mp4"


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "yolo11n.pt"

# COCO:
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
VEHICLE_CLASSES = [2, 3, 5, 7]

CONFIDENCE = 0.30
IMAGE_SIZE = 640


def main():

    print("=" * 60)
    print("INDIAN ANPR - VEHICLE DETECTION BASELINE")
    print("=" * 60)

    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(
            f"Video not found: {INPUT_VIDEO}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # GPU CHECK
    # --------------------------------------------------------

    device = 0 if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )
    else:
        print("WARNING: Running on CPU")

    # --------------------------------------------------------
    # LOAD YOLO
    # --------------------------------------------------------

    print("Loading model...")

    model = YOLO(MODEL_NAME)

    # --------------------------------------------------------
    # OPEN VIDEO
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
        f"Resolution : {width}x{height}"
    )

    print(
        f"FPS        : {fps:.2f}"
    )

    print(
        f"Frames     : {total_frames}"
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

    print("\nStarting detection...\n")

    # --------------------------------------------------------
    # PROCESS FRAMES
    # --------------------------------------------------------

    while True:

        success, frame = cap.read()

        if not success:
            break

        results = model.predict(
            source=frame,
            classes=VEHICLE_CLASSES,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device=device,
            verbose=False
        )

        result = results[0]

        annotated = result.plot()

        frame_number += 1

        # ----------------------------------------------------
        # DISPLAY INFO
        # ----------------------------------------------------

        detected = len(result.boxes)

        cv2.rectangle(
            annotated,
            (15, 15),
            (470, 125),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            annotated,
            "Indian ANPR - Vehicle Detection",
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
            f"Vehicles in frame: {detected}",
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )

        writer.write(annotated)

        if frame_number % 100 == 0:
            print(
                f"Processed "
                f"{frame_number}/{total_frames}"
            )

    cap.release()
    writer.release()

    print()
    print("=" * 60)
    print("VEHICLE DETECTION COMPLETE")
    print("=" * 60)

    print(
        f"Frames processed: {frame_number}"
    )

    print(
        f"Output: {OUTPUT_VIDEO}"
    )


if __name__ == "__main__":
    main()  