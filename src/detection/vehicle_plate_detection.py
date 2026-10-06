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

VEHICLE_MODEL_PATH = "yolo11n.pt"

PLATE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "plate"
    / "npds_yolo11n_best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "M05_vehicle_plate_pipeline"
)

OUTPUT_VIDEO = (
    OUTPUT_DIR
    / "vehicle_plate_detection.mp4"
)

CROP_DIR = (
    OUTPUT_DIR
    / "selected_plate_crops"
)


# ============================================================
# SETTINGS
# ============================================================

# COCO vehicle classes
# car = 2
# motorcycle = 3
# bus = 5
# truck = 7
VEHICLE_CLASSES = [2, 3, 5, 7]

VEHICLE_CONF = 0.30

# Slightly stricter than our previous 0.25
PLATE_CONF = 0.45

VEHICLE_IMGSZ = 640

# Plates are small, so use higher resolution
PLATE_IMGSZ = 960

# Save only every Nth frame candidate
SAVE_EVERY_N_FRAMES = 5

# Reject extremely tiny plate crops
MIN_PLATE_WIDTH = 40
MIN_PLATE_HEIGHT = 12


def clamp(value, minimum, maximum):

    return max(
        minimum,
        min(value, maximum)
    )


def main():

    print("=" * 65)
    print("INDIAN ANPR - VEHICLE + NUMBER PLATE PIPELINE")
    print("=" * 65)

    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(INPUT_VIDEO)

    if not PLATE_MODEL_PATH.exists():
        raise FileNotFoundError(PLATE_MODEL_PATH)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CROP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    device = 0 if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print("\nLoading vehicle model...")

    vehicle_model = YOLO(
        VEHICLE_MODEL_PATH
    )

    print("Loading plate model...")

    plate_model = YOLO(
        str(PLATE_MODEL_PATH)
    )

    # ========================================================
    # VIDEO
    # ========================================================

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

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (width, height)
    )

    frame_number = 0

    total_vehicles = 0
    total_plate_candidates = 0
    saved_crops = 0

    # ========================================================
    # FRAME LOOP
    # ========================================================

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        annotated = frame.copy()

        # ----------------------------------------------------
        # STEP 1: VEHICLE DETECTION
        # ----------------------------------------------------

        vehicle_results = vehicle_model.predict(
            frame,
            classes=VEHICLE_CLASSES,
            conf=VEHICLE_CONF,
            imgsz=VEHICLE_IMGSZ,
            device=device,
            verbose=False
        )[0]

        frame_vehicle_count = 0
        frame_plate_count = 0

        # ----------------------------------------------------
        # STEP 2: EACH VEHICLE
        # ----------------------------------------------------

        for vehicle_index, vehicle_box in enumerate(
            vehicle_results.boxes
        ):

            vx1, vy1, vx2, vy2 = map(
                int,
                vehicle_box.xyxy[0].tolist()
            )

            vehicle_conf = float(
                vehicle_box.conf[0]
            )

            vehicle_class = int(
                vehicle_box.cls[0]
            )

            vehicle_name = vehicle_model.names[
                vehicle_class
            ]

            # Expand vehicle box slightly
            vehicle_width = vx2 - vx1
            vehicle_height = vy2 - vy1

            pad_x = int(
                vehicle_width * 0.08
            )

            pad_y = int(
                vehicle_height * 0.08
            )

            vx1 = clamp(
                vx1 - pad_x,
                0,
                width
            )

            vy1 = clamp(
                vy1 - pad_y,
                0,
                height
            )

            vx2 = clamp(
                vx2 + pad_x,
                0,
                width
            )

            vy2 = clamp(
                vy2 + pad_y,
                0,
                height
            )

            vehicle_crop = frame[
                vy1:vy2,
                vx1:vx2
            ]

            if vehicle_crop.size == 0:
                continue

            frame_vehicle_count += 1
            total_vehicles += 1

            # ------------------------------------------------
            # DRAW VEHICLE BOX
            # ------------------------------------------------

            cv2.rectangle(
                annotated,
                (vx1, vy1),
                (vx2, vy2),
                (255, 180, 0),
                2
            )

            cv2.putText(
                annotated,
                f"{vehicle_name} {vehicle_conf:.2f}",
                (vx1, max(20, vy1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 180, 0),
                2
            )

            # ------------------------------------------------
            # STEP 3: PLATE DETECTION INSIDE VEHICLE
            # ------------------------------------------------

            plate_results = plate_model.predict(
                vehicle_crop,
                conf=PLATE_CONF,
                imgsz=PLATE_IMGSZ,
                device=device,
                verbose=False
            )[0]

            for plate_index, plate_box in enumerate(
                plate_results.boxes
            ):

                px1, py1, px2, py2 = map(
                    int,
                    plate_box.xyxy[0].tolist()
                )

                plate_conf = float(
                    plate_box.conf[0]
                )

                plate_width = px2 - px1
                plate_height = py2 - py1

                # --------------------------------------------
                # BASIC QUALITY FILTER
                # --------------------------------------------

                if (
                    plate_width < MIN_PLATE_WIDTH
                    or plate_height < MIN_PLATE_HEIGHT
                ):
                    continue

                # Convert plate coordinates back to
                # full-frame coordinates
                global_x1 = vx1 + px1
                global_y1 = vy1 + py1
                global_x2 = vx1 + px2
                global_y2 = vy1 + py2

                frame_plate_count += 1
                total_plate_candidates += 1

                # --------------------------------------------
                # DRAW PLATE BOX
                # --------------------------------------------

                cv2.rectangle(
                    annotated,
                    (global_x1, global_y1),
                    (global_x2, global_y2),
                    (0, 255, 0),
                    3
                )

                cv2.putText(
                    annotated,
                    f"PLATE {plate_conf:.2f}",
                    (
                        global_x1,
                        max(
                            20,
                            global_y1 - 8
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

                # --------------------------------------------
                # SAVE SOME CROPS, NOT EVERY FRAME
                # --------------------------------------------

                if (
                    frame_number
                    % SAVE_EVERY_N_FRAMES
                    == 0
                ):

                    plate_crop = vehicle_crop[
                        py1:py2,
                        px1:px2
                    ]

                    if plate_crop.size == 0:
                        continue

                    filename = (
                        f"frame_{frame_number:05d}_"
                        f"vehicle_{vehicle_index}_"
                        f"plate_{plate_index}_"
                        f"conf_{plate_conf:.2f}.jpg"
                    )

                    cv2.imwrite(
                        str(
                            CROP_DIR
                            / filename
                        ),
                        plate_crop
                    )

                    saved_crops += 1

        # ====================================================
        # HUD
        # ====================================================

        cv2.rectangle(
            annotated,
            (15, 15),
            (500, 145),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            annotated,
            "Indian ANPR - Vehicle + Plate",
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
            f"Vehicles: {frame_vehicle_count}",
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated,
            f"Plate candidates: {frame_plate_count}",
            (250, 110),
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
                f"Frame {frame_number}/{total_frames}"
                f" | plates={total_plate_candidates}"
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    writer.release()

    print("\n" + "=" * 65)
    print("PIPELINE COMPLETE")
    print("=" * 65)

    print(
        "Vehicle detections:",
        total_vehicles
    )

    print(
        "Accepted plate candidates:",
        total_plate_candidates
    )

    print(
        "Crops saved:",
        saved_crops
    )

    print(
        "\nOutput video:",
        OUTPUT_VIDEO
    )

    print(
        "Selected crops:",
        CROP_DIR
    )


if __name__ == "__main__":
    main()