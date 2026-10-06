from pathlib import Path
import csv

import cv2
import numpy as np
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

PLATE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "plate"
    / "npds_yolo11n_best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "M06_tracking_best_plate"
)

OUTPUT_VIDEO = OUTPUT_DIR / "tracking_best_plate.mp4"

BEST_CROP_DIR = OUTPUT_DIR / "best_plate_crops"

SUMMARY_CSV = OUTPUT_DIR / "track_summary.csv"


# ============================================================
# SETTINGS
# ============================================================

VEHICLE_MODEL = "yolo11n.pt"

# COCO
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
VEHICLE_CLASSES = [2, 3, 5, 7]

VEHICLE_CONF = 0.30
PLATE_CONF = 0.40

VEHICLE_IMGSZ = 640
PLATE_IMGSZ = 960

MIN_PLATE_WIDTH = 40
MIN_PLATE_HEIGHT = 12

MIN_ASPECT_RATIO = 1.2
MAX_ASPECT_RATIO = 6.5

TOP_K_CANDIDATES = 5
MIN_FRAME_GAP = 5


# ============================================================
# QUALITY FUNCTIONS
# ============================================================

def calculate_sharpness(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()


def calculate_contrast(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return float(
        np.std(gray)
    )


def calculate_quality_score(
    plate_crop,
    detector_conf
):

    height, width = plate_crop.shape[:2]

    sharpness = calculate_sharpness(
        plate_crop
    )

    contrast = calculate_contrast(
        plate_crop
    )

    area = width * height

    # Normalized components
    sharpness_score = min(
        sharpness / 500.0,
        1.0
    )

    contrast_score = min(
        contrast / 70.0,
        1.0
    )

    size_score = min(
        area / 15000.0,
        1.0
    )

    # Weighted quality score
    score = (
        detector_conf * 0.55
        + size_score * 0.25
        + sharpness_score * 0.10
        + contrast_score * 0.10
    )

    return {
        "score": score,
        "sharpness": sharpness,
        "contrast": contrast,
        "width": width,
        "height": height
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("INDIAN ANPR - TRACKING + BEST PLATE SELECTION")
    print("=" * 65)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    BEST_CROP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(INPUT_VIDEO)

    if not PLATE_MODEL_PATH.exists():
        raise FileNotFoundError(PLATE_MODEL_PATH)

    device = (
        0
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    print("Loading vehicle tracker...")

    vehicle_model = YOLO(
        VEHICLE_MODEL
    )

    print("Loading plate detector...")

    plate_model = YOLO(
        str(PLATE_MODEL_PATH)
    )

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(INPUT_VIDEO)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open video."
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

    # --------------------------------------------------------
    # TRACK MEMORY
    # --------------------------------------------------------

    plate_candidates = {}

    frame_number = 0

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
        # VEHICLE TRACKING WITH BYTETRACK
        # ----------------------------------------------------

        vehicle_result = vehicle_model.track(
            source=frame,
            persist=True,
            tracker="botsort.yaml",
            classes=VEHICLE_CLASSES,
            conf=VEHICLE_CONF,
            imgsz=VEHICLE_IMGSZ,
            device=device,
            verbose=False
        )[0]

        boxes = vehicle_result.boxes

        if (
            boxes is None
            or boxes.id is None
        ):
            writer.write(
                annotated
            )
            continue

        track_ids = (
            boxes.id
            .int()
            .cpu()
            .tolist()
        )

        # ----------------------------------------------------
        # EACH TRACKED VEHICLE
        # ----------------------------------------------------

        for box, track_id in zip(
            boxes,
            track_ids
        ):

            vx1, vy1, vx2, vy2 = map(
                int,
                box.xyxy[0].tolist()
            )

            vehicle_class = int(
                box.cls[0]
            )

            vehicle_conf = float(
                box.conf[0]
            )

            vehicle_name = (
                vehicle_model.names[
                    vehicle_class
                ]
            )

            vx1 = max(0, vx1)
            vy1 = max(0, vy1)

            vx2 = min(
                width,
                vx2
            )

            vy2 = min(
                height,
                vy2
            )

            vehicle_crop = frame[
                vy1:vy2,
                vx1:vx2
            ]

            if vehicle_crop.size == 0:
                continue

            # ------------------------------------------------
            # DRAW VEHICLE TRACK
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
                f"{vehicle_name} ID:{track_id}",
                (
                    vx1,
                    max(20, vy1 - 8)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 180, 0),
                2
            )

            # ------------------------------------------------
            # PLATE DETECTION INSIDE VEHICLE
            # ------------------------------------------------

            plate_result = plate_model.predict(
                source=vehicle_crop,
                conf=PLATE_CONF,
                imgsz=PLATE_IMGSZ,
                device=device,
                verbose=False
            )[0]

            for plate_box in plate_result.boxes:

                px1, py1, px2, py2 = map(
                    int,
                    plate_box.xyxy[0].tolist()
                )

                plate_conf = float(
                    plate_box.conf[0]
                )

                plate_width = (
                    px2 - px1
                )

                plate_height = (
                    py2 - py1
                )

                # --------------------------------------------
                # BASIC FILTERING
                # --------------------------------------------

                if (
                    plate_width
                    < MIN_PLATE_WIDTH
                    or plate_height
                    < MIN_PLATE_HEIGHT
                ):
                    continue

                aspect_ratio = (
                    plate_width
                    / max(
                        plate_height,
                        1
                    )
                )

                if not (
                    MIN_ASPECT_RATIO
                    <= aspect_ratio
                    <= MAX_ASPECT_RATIO
                ):
                    continue

                plate_crop = vehicle_crop[
                    py1:py2,
                    px1:px2
                ]

                if plate_crop.size == 0:
                    continue

                # --------------------------------------------
                # QUALITY SCORING
                # --------------------------------------------

                quality = (
                    calculate_quality_score(
                        plate_crop,
                        plate_conf
                    )
                )

                candidate_score = (
                    quality["score"]
                )

                # --------------------------------------------
                # KEEP ONLY BEST CROP PER TRACK
                # --------------------------------------------

                candidate = {
                    "crop": plate_crop.copy(),
                    "score": candidate_score,
                    "confidence": plate_conf,
                    "sharpness": quality["sharpness"],
                    "contrast": quality["contrast"],
                    "width": quality["width"],
                    "height": quality["height"],
                    "frame": frame_number,
                    "vehicle": vehicle_name
                }

                candidates = plate_candidates.setdefault(
                    track_id,
                    []
                )

                # ------------------------------------------------
                # FRAME-DIVERSITY FILTER
                # ------------------------------------------------
                # Check whether we already have a candidate from
                # nearly the same moment in the video.
                nearby_index = next(
                    (
                        index
                        for index, existing
                        in enumerate(candidates)
                        if abs(
                            candidate["frame"]
                            - existing["frame"]
                        ) < MIN_FRAME_GAP
                    ),
                    None
                )

                # No nearby candidate:
                # add this one.
                if nearby_index is None:

                    candidates.append(
                        candidate
                    )

                # Nearby candidate exists:
                # keep whichever has the better quality score.
                elif (
                    candidate["score"]
                    > candidates[
                        nearby_index
                    ]["score"]
                ):

                    candidates[
                        nearby_index
                    ] = candidate

                # Keep highest-quality candidates first.
                candidates.sort(
                    key=lambda item: item["score"],
                    reverse=True
                )

                # Maximum K candidates per tracked vehicle.
                del candidates[
                    TOP_K_CANDIDATES:
                ]

                # --------------------------------------------
                # DRAW PLATE
                # --------------------------------------------
                
        gx1 = vx1 + px1
        gy1 = vy1 + py1
        
        gx2 = vx1 + px2
        gy2 = vy1 + py2

        cv2.rectangle(
            annotated,
            (gx1, gy1),
            (gx2, gy2),
            (0, 255, 0),
            3
        )

        cv2.putText(
            annotated,
            (
                f"PLATE "
                f"{plate_conf:.2f}"
            ),
            (
                gx1,
                max(
                    20,
                    gy1 - 8
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # HUD
        # ----------------------------------------------------

        cv2.rectangle(
            annotated,
            (15, 15),
            (540, 125),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            annotated,
            "Indian ANPR - Tracking",
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
            (
                "Tracked vehicles with "
                f"plate candidate: {len(plate_candidates)}"
            ),
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
                f"Frame "
                f"{frame_number}/{total_frames}"
                f" | plate tracks="
                f"{len(plate_candidates)}"
            )

    # ========================================================
    # SAVE BEST PLATE PER TRACK
    # ========================================================

    print("\nSaving best plate crops...")

    summary_rows = []

    for track_id, candidates in plate_candidates.items():

        for rank, data in enumerate(
            candidates,
            start=1
        ):

            filename = (
                f"track_{track_id:04d}_"
                f"rank_{rank}_"
                f"frame_{data['frame']:05d}_"
                f"score_{data['score']:.3f}.jpg"
            )

            crop_path = (
                BEST_CROP_DIR
                / filename
            )

            cv2.imwrite(
                str(crop_path),
                data["crop"]
            )

            summary_rows.append({
                "track_id": track_id,
                "rank": rank,
                "vehicle": data["vehicle"],
                "frame": data["frame"],

                "quality_score": round(
                    data["score"],
                    4
                ),

                "plate_confidence": round(
                    data["confidence"],
                    4
                ),

                "sharpness": round(
                    data["sharpness"],
                    2
                ),

                "contrast": round(
                    data["contrast"],
                    2
                ),

                "width": data["width"],
                "height": data["height"],

                "filename": filename
            })

    # --------------------------------------------------------
    # CSV SUMMARY
    # --------------------------------------------------------

    if summary_rows:

        with open(
            SUMMARY_CSV,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer_csv = csv.DictWriter(
                file,
                fieldnames=summary_rows[0].keys()
            )

            writer_csv.writeheader()

            writer_csv.writerows(
                summary_rows
            )

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    cap.release()
    writer.release()

    print("\n" + "=" * 65)
    print("TRACKING COMPLETE")
    print("=" * 65)

    print(
        "Vehicle tracks with candidates:",
        len(plate_candidates)
    )

    print(
        "\nOutput video:",
        OUTPUT_VIDEO
    )

    print(
        "Best crops:",
        BEST_CROP_DIR
    )

    print(
        "Summary:",
        SUMMARY_CSV
    )


if __name__ == "__main__":
    main()