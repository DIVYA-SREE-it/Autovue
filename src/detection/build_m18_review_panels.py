from pathlib import Path
import csv

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

M18_DIR = (
    ROOT
    / "outputs"
    / "M18_real_road_eval"
)

RAW_DIR = (
    M18_DIR
    / "raw_frames"
)

DETECTED_DIR = (
    M18_DIR
    / "detected_frames"
)

REVIEW_CSV = (
    M18_DIR
    / "manual_review.csv"
)

PANEL_DIR = (
    M18_DIR
    / "review_panels"
)

PANEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def add_header(
    image,
    title,
):

    header_height = 70

    header = np.zeros(
        (
            header_height,
            image.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    cv2.putText(
        header,
        title,
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return np.vstack(
        [
            header,
            image,
        ]
    )


def add_footer(
    image,
    text,
):

    footer_height = 65

    footer = np.zeros(
        (
            footer_height,
            image.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    cv2.putText(
        footer,
        text,
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return np.vstack(
        [
            image,
            footer,
        ]
    )


def find_frame(
    base_dir,
    video_key,
    sample_id,
):

    folder = (
        base_dir
        / video_key
    )

    matches = list(
        folder.glob(
            f"{sample_id}_*.jpg"
        )
    )

    if not matches:
        return None

    return matches[0]


def main():

    print("=" * 72)
    print("M18B — BUILD HUMAN REVIEW PANELS")
    print("=" * 72)

    with open(
        REVIEW_CSV,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        rows = list(
            csv.DictReader(file)
        )

    created = 0
    missing = 0

    for row in rows:

        sample_id = row[
            "sample_id"
        ]

        video_name = row[
            "video"
        ]

        video_key = (
            Path(video_name).stem
        )

        raw_path = find_frame(
            RAW_DIR,
            video_key,
            sample_id,
        )

        detected_path = find_frame(
            DETECTED_DIR,
            video_key,
            sample_id,
        )

        if (
            raw_path is None
            or detected_path is None
        ):

            print(
                f"Missing image pair: "
                f"{sample_id}"
            )

            missing += 1
            continue

        raw = cv2.imread(
            str(raw_path)
        )

        detected = cv2.imread(
            str(detected_path)
        )

        if (
            raw is None
            or detected is None
        ):

            print(
                f"Could not read: "
                f"{sample_id}"
            )

            missing += 1
            continue

        # Keep review images manageable.
        target_width = 960

        scale = (
            target_width
            / raw.shape[1]
        )

        target_height = int(
            raw.shape[0]
            * scale
        )

        raw = cv2.resize(
            raw,
            (
                target_width,
                target_height,
            ),
        )

        detected = cv2.resize(
            detected,
            (
                target_width,
                target_height,
            ),
        )

        raw = add_header(
            raw,
            "RAW FRAME",
        )

        detected = add_header(
            detected,
            "YOLO11n DETECTIONS",
        )

        panel = np.hstack(
            [
                raw,
                detected,
            ]
        )

        footer_text = (
            f"{sample_id} | "
            f"{video_name} | "
            f"{float(row['timestamp_sec']):.2f}s | "
            f"Detector boxes: "
            f"{row['detections']}"
        )

        panel = add_footer(
            panel,
            footer_text,
        )

        output_path = (
            PANEL_DIR
            / f"{sample_id}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            panel,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                92,
            ],
        )

        created += 1

    print(
        f"\nReview panels created: "
        f"{created}"
    )

    print(
        f"Missing pairs: "
        f"{missing}"
    )

    print(
        "\nSaved to:"
    )

    print(
        PANEL_DIR
    )


if __name__ == "__main__":
    main()