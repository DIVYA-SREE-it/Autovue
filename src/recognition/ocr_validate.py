from pathlib import Path
import csv
import re

import cv2
import numpy as np
import torch
import easyocr


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "M06_tracking_best_plate"
    / "best_plate_crops"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "M07_ocr_validation"
)

VALID_DIR = OUTPUT_DIR / "format_valid"
REVIEW_DIR = OUTPUT_DIR / "needs_review"
REJECT_DIR = OUTPUT_DIR / "rejected"

CSV_PATH = OUTPUT_DIR / "ocr_results.csv"


# Standard Indian registration:
# TN04AE0711
# KA01AB1234
STANDARD_PATTERN = re.compile(
    r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$"
)

# Bharat Series:
# 22BH1234AA
BH_PATTERN = re.compile(
    r"^[0-9]{2}BH[0-9]{4}[A-Z]{2}$"
)

ALLOWED = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def clean_text(text):

    text = text.upper()

    return re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )


def format_valid(text):

    return bool(
        STANDARD_PATTERN.fullmatch(text)
        or BH_PATTERN.fullmatch(text)
    )


def plausible_plate(text):

    if not 6 <= len(text) <= 12:
        return False

    letters = sum(
        character.isalpha()
        for character in text
    )

    digits = sum(
        character.isdigit()
        for character in text
    )

    return (
        letters >= 2
        and digits >= 2
    )


def classify(text, confidence):

    if format_valid(text):

        if confidence >= 0.60:
            return "FORMAT_VALID"

        return "NEEDS_REVIEW"

    if plausible_plate(text):
        return "NEEDS_REVIEW"

    return "REJECTED"


def create_variants(image):

    height, width = image.shape[:2]

    # Enlarge small plate crop
    scale = max(
        2.0,
        min(
            6.0,
            320 / max(width, 1)
        )
    )

    resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # Mild sharpening
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpened = cv2.filter2D(
        enhanced,
        -1,
        sharpen_kernel
    )

    _, thresholded = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )

    return {
        "original": resized,
        "clahe": enhanced,
        "sharpened": sharpened,
        "threshold": thresholded
    }


def run_ocr(reader, image):

    results = reader.readtext(
        image,
        detail=1,
        paragraph=False,
        allowlist=ALLOWED
    )

    if not results:
        return "", 0.0

    pieces = []

    for box, text, confidence in results:

        cleaned = clean_text(text)

        if not cleaned:
            continue

        top = min(
            point[1]
            for point in box
        )

        left = min(
            point[0]
            for point in box
        )

        pieces.append(
            (
                top,
                left,
                cleaned,
                float(confidence)
            )
        )

    if not pieces:
        return "", 0.0

    # Supports two-line plates reasonably well
    pieces.sort(
        key=lambda item: (
            round(item[0] / 30),
            item[1]
        )
    )

    combined = "".join(
        piece[2]
        for piece in pieces
    )

    average_confidence = sum(
        piece[3]
        for piece in pieces
    ) / len(pieces)

    return combined, average_confidence


def choose_best_result(reader, image):

    variants = create_variants(image)

    candidates = []

    for variant_name, variant in variants.items():

        text, confidence = run_ocr(
            reader,
            variant
        )

        if not text:
            continue

        score = confidence

        # Strong preference for valid Indian format
        if format_valid(text):
            score += 0.40

        elif plausible_plate(text):
            score += 0.10

        candidates.append({
            "variant": variant_name,
            "text": text,
            "confidence": confidence,
            "score": score
        })

    if not candidates:

        return {
            "variant": "none",
            "text": "",
            "confidence": 0.0
        }

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return candidates[0]


def save_preview(
    original,
    filename,
    text,
    confidence,
    status
):

    preview = original.copy()

    preview = cv2.resize(
        preview,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    canvas = cv2.copyMakeBorder(
        preview,
        70,
        10,
        10,
        10,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    cv2.putText(
        canvas,
        text if text else "NO TEXT",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        canvas,
        f"{status} | OCR {confidence:.2f}",
        (10, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    if status == "FORMAT_VALID":
        destination = VALID_DIR

    elif status == "NEEDS_REVIEW":
        destination = REVIEW_DIR

    else:
        destination = REJECT_DIR

    cv2.imwrite(
        str(destination / filename),
        canvas
    )


def main():

    print("=" * 65)
    print("INDIAN ANPR - OCR + FORMAT VALIDATION")
    print("=" * 65)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(INPUT_DIR)

    for directory in [
        OUTPUT_DIR,
        VALID_DIR,
        REVIEW_DIR,
        REJECT_DIR
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    gpu = torch.cuda.is_available()

    print("OCR GPU:", gpu)
    print("Loading EasyOCR...")

    reader = easyocr.Reader(
        ["en"],
        gpu=gpu
    )

    image_files = sorted(
        list(INPUT_DIR.glob("*.jpg"))
        + list(INPUT_DIR.glob("*.png"))
    )

    rows = []

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        result = choose_best_result(
            reader,
            image
        )

        cleaned_text = clean_text(
            result["text"]
        )

        status = classify(
            cleaned_text,
            result["confidence"]
        )

        rows.append({
            "filename": image_path.name,
            "ocr_text": cleaned_text,
            "ocr_confidence": round(
                result["confidence"],
                4
            ),
            "preprocessing": result[
                "variant"
            ],
            "format_valid": format_valid(
                cleaned_text
            ),
            "status": status
        })

        save_preview(
            image,
            image_path.name,
            cleaned_text,
            result["confidence"],
            status
        )

        print(
            f"[{index}/{len(image_files)}] "
            f"{image_path.name} -> "
            f"{cleaned_text or 'NO_TEXT'} "
            f"({result['confidence']:.2f}) "
            f"=> {status}"
        )

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "filename",
            "ocr_text",
            "ocr_confidence",
            "preprocessing",
            "format_valid",
            "status"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 65)
    print("OCR COMPLETE")
    print("=" * 65)

    print("Results:", CSV_PATH)
    print("Valid :", VALID_DIR)
    print("Review:", REVIEW_DIR)
    print("Reject:", REJECT_DIR)


if __name__ == "__main__":
    main()