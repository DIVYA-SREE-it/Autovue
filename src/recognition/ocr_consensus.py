from pathlib import Path
from collections import defaultdict
import csv
import re

import cv2
import numpy as np
import torch
import easyocr


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRACK_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "M06_tracking_best_plate"
    / "track_summary.csv"
)

CROP_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "M06_tracking_best_plate"
    / "best_plate_crops"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "M08_ocr_consensus"
)

CANDIDATE_CSV = OUTPUT_DIR / "candidate_ocr.csv"
CONSENSUS_CSV = OUTPUT_DIR / "track_consensus.csv"


# ============================================================
# INDIAN REGISTRATION RULES
# ============================================================

STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR",
    "CG", "CH", "DD", "DL", "DN",
    "GA", "GJ", "HP", "HR", "JH",
    "JK", "KA", "KL", "LA", "LD",
    "MH", "ML", "MN", "MP", "MZ",
    "NL", "OD", "PB", "PY", "RJ",
    "SK", "TN", "TR", "TS", "UK",
    "UP", "WB"
}

ALLOWED = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

LETTER_TO_DIGIT = {
    "O": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6"
}

DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "8": "B",
    "6": "G"
}


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(text):

    text = text.upper()

    return re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )


def convert_expected_type(character, expected):

    if expected == "L":

        if character.isalpha():
            return character, 0

        if character in DIGIT_TO_LETTER:
            return DIGIT_TO_LETTER[character], 1

    if expected == "D":

        if character.isdigit():
            return character, 0

        if character in LETTER_TO_DIGIT:
            return LETTER_TO_DIGIT[character], 1

    return None, 99


def correct_standard_plate(text):

    text = clean_text(text)

    best = None

    # Standard form:
    # SS + district digits + series letters + number digits
    #
    # Example:
    # TN04AE0711

    # Standard Indian registration uses a two-digit RTO/district code.
    for district_len in [2]:

        for series_len in [1, 2, 3]:

            for number_len in [1, 2, 3, 4]:

                expected_length = (
                    2
                    + district_len
                    + series_len
                    + number_len
                )

                if len(text) != expected_length:
                    continue

                mask = (
                    "LL"
                    + "D" * district_len
                    + "L" * series_len
                    + "D" * number_len
                )

                corrected = []
                cost = 0
                valid = True

                for character, expected in zip(
                    text,
                    mask
                ):

                    converted, conversion_cost = (
                        convert_expected_type(
                            character,
                            expected
                        )
                    )

                    if converted is None:
                        valid = False
                        break

                    corrected.append(converted)
                    cost += conversion_cost

                if not valid:
                    continue

                corrected = "".join(corrected)

                state = corrected[:2]

                if state not in STATE_CODES:
                    continue

                candidate = {
                    "text": corrected,
                    "cost": cost,
                    "format": "STANDARD"
                }

                if (
                    best is None
                    or candidate["cost"] < best["cost"]
                ):
                    best = candidate

    return best


def correct_bh_plate(text):

    text = clean_text(text)

    # BH format:
    # 22BH1234AA

    if len(text) != 10:
        return None

    mask = "DDLLDDDDLL"

    corrected = []
    cost = 0

    for character, expected in zip(
        text,
        mask
    ):

        converted, conversion_cost = (
            convert_expected_type(
                character,
                expected
            )
        )

        if converted is None:
            return None

        corrected.append(converted)
        cost += conversion_cost

    corrected = "".join(corrected)

    if corrected[2:4] != "BH":
        return None

    return {
        "text": corrected,
        "cost": cost,
        "format": "BH"
    }


def correct_indian_plate(text):

    standard = correct_standard_plate(text)
    bh = correct_bh_plate(text)

    candidates = [
        candidate
        for candidate in [standard, bh]
        if candidate is not None
    ]

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item["cost"]
    )

    best = candidates[0]

    # Do not accept excessive OCR corrections.
    if best["cost"] > 2:
        return None

    return best


def plausible_plate(text):

    text = clean_text(text)

    # Standard Indian plates are normally around this range.
    if not 7 <= len(text) <= 11:
        return False

    # First two characters should identify the state/UT.
    state = text[:2]

    if state not in STATE_CODES:
        return False

    # Positions 3-4 should represent the two-digit RTO code.
    district = []

    for character in text[2:4]:

        converted, _ = convert_expected_type(
            character,
            "D"
        )

        if converted is None:
            return False

        district.append(converted)

    # Remaining characters should contain at least
    # one letter and one digit.
    remainder = text[4:]

    has_letter = any(
        character.isalpha()
        for character in remainder
    )

    has_digit = any(
        character.isdigit()
        for character in remainder
    )

    return (
        has_letter
        and has_digit
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def create_variants(image):

    height, width = image.shape[:2]

    scale = max(
        2.0,
        min(
            6.0,
            360 / max(width, 1)
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
        "gray": gray,
        "clahe": enhanced,
        "sharpened": sharpened,
        "threshold": thresholded
    }


# ============================================================
# OCR
# ============================================================

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

    confidence = sum(
        piece[3]
        for piece in pieces
    ) / len(pieces)

    return combined, confidence


def best_ocr_for_crop(reader, image):

    variants = create_variants(image)

    candidates = []

    for variant_name, variant in variants.items():

        raw_text, ocr_conf = run_ocr(
            reader,
            variant
        )

        if not raw_text:
            continue

        cleaned = clean_text(raw_text)

        correction = correct_indian_plate(
            cleaned
        )

        if correction:

            corrected_text = correction["text"]
            correction_cost = correction["cost"]

            score = (
                ocr_conf
                + 0.40
                - correction_cost * 0.08
            )

            valid_format = True

        else:

            corrected_text = cleaned
            correction_cost = -1

            score = ocr_conf

            if plausible_plate(cleaned):
                score += 0.10

            valid_format = False

        candidates.append({
            "variant": variant_name,
            "raw_text": cleaned,
            "text": corrected_text,
            "ocr_confidence": ocr_conf,
            "valid_format": valid_format,
            "correction_cost": correction_cost,
            "score": score
        })

    if not candidates:

        return {
            "variant": "none",
            "raw_text": "",
            "text": "",
            "ocr_confidence": 0.0,
            "valid_format": False,
            "correction_cost": -1,
            "score": 0.0
        }

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return candidates[0]


# ============================================================
# CONSENSUS
# ============================================================

def build_consensus(candidate_results):

    valid_results = [
        result
        for result in candidate_results
        if result["valid_format"]
    ]

    if valid_results:

        votes = defaultdict(float)
        counts = defaultdict(int)

        for result in valid_results:

            text = result["text"]

            weight = (
                result["ocr_confidence"] * 0.50
                + result["quality_score"] * 0.30
                + result["plate_confidence"] * 0.20
            )

            votes[text] += weight
            counts[text] += 1

        final_text = max(
            votes,
            key=votes.get
        )

        support = counts[final_text]

        total_valid = len(valid_results)

        consensus_ratio = (
            support / total_valid
            if total_valid
            else 0
        )

        if (
            support >= 2
            and consensus_ratio >= 0.50
        ):
            status = "VERIFIED"

        else:
            status = "NEEDS_REVIEW"

        return {
            "final_text": final_text,
            "status": status,
            "support": support,
            "valid_candidates": total_valid,
            "consensus_ratio": consensus_ratio
        }

    # No valid Indian-format candidate.
    plausible_results = [
        result
        for result in candidate_results
        if plausible_plate(
            result["text"]
        )
    ]

    if plausible_results:

        plausible_results.sort(
            key=lambda item: (
                item["ocr_confidence"]
                + item["quality_score"]
            ),
            reverse=True
        )

        best = plausible_results[0]

        return {
            "final_text": best["text"],
            "status": "NEEDS_REVIEW",
            "support": 1,
            "valid_candidates": 0,
            "consensus_ratio": 0.0
        }

    return {
        "final_text": "",
        "status": "REJECTED",
        "support": 0,
        "valid_candidates": 0,
        "consensus_ratio": 0.0
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("INDIAN ANPR - MULTI-FRAME OCR CONSENSUS")
    print("=" * 70)

    if not TRACK_CSV.exists():
        raise FileNotFoundError(TRACK_CSV)

    if not CROP_DIR.exists():
        raise FileNotFoundError(CROP_DIR)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "OCR GPU:",
        torch.cuda.is_available()
    )

    reader = easyocr.Reader(
        ["en"],
        gpu=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # READ TRACK METADATA
    # --------------------------------------------------------

    rows = []

    with open(
        TRACK_CSV,
        "r",
        encoding="utf-8"
    ) as file:

        reader_csv = csv.DictReader(file)

        rows = list(reader_csv)

    grouped = defaultdict(list)

    for row in rows:

        grouped[
            int(row["track_id"])
        ].append(row)

    candidate_output = []
    consensus_output = []

    # --------------------------------------------------------
    # PROCESS EACH TRACK
    # --------------------------------------------------------

    for track_id in sorted(grouped):

        track_rows = grouped[track_id]

        track_results = []

        print(
            f"\nTrack {track_id} "
            f"({len(track_rows)} candidates)"
        )

        for row in track_rows:

            image_path = (
                CROP_DIR
                / row["filename"]
            )

            image = cv2.imread(
                str(image_path)
            )

            if image is None:
                continue

            result = best_ocr_for_crop(
                reader,
                image
            )

            result["track_id"] = track_id
            result["rank"] = int(row["rank"])
            result["filename"] = row["filename"]

            result["quality_score"] = float(
                row["quality_score"]
            )

            result["plate_confidence"] = float(
                row["plate_confidence"]
            )

            result["frame"] = int(
                row["frame"]
            )

            result["vehicle"] = row["vehicle"]

            track_results.append(
                result
            )

            candidate_output.append(
                result.copy()
            )

            print(
                f"  rank {row['rank']} "
                f"frame {row['frame']} -> "
                f"{result['raw_text'] or 'NO_TEXT'}"
                f" => "
                f"{result['text'] or 'NO_TEXT'} "
                f"({result['ocr_confidence']:.2f})"
            )

        consensus = build_consensus(
            track_results
        )

        consensus_output.append({
            "track_id": track_id,
            "vehicle": (
                track_rows[0]["vehicle"]
                if track_rows
                else ""
            ),
            "candidate_count": len(
                track_results
            ),
            "final_plate": consensus[
                "final_text"
            ],
            "status": consensus[
                "status"
            ],
            "support": consensus[
                "support"
            ],
            "valid_candidates": consensus[
                "valid_candidates"
            ],
            "consensus_ratio": round(
                consensus[
                    "consensus_ratio"
                ],
                4
            )
        })

        print(
            "  FINAL:",
            consensus["final_text"]
            or "NO PLATE",
            "=>",
            consensus["status"]
        )

    # --------------------------------------------------------
    # SAVE CANDIDATE CSV
    # --------------------------------------------------------

    if candidate_output:

        fields = [
            "track_id",
            "rank",
            "frame",
            "vehicle",
            "filename",
            "raw_text",
            "text",
            "ocr_confidence",
            "variant",
            "valid_format",
            "correction_cost",
            "quality_score",
            "plate_confidence",
            "score"
        ]

        with open(
            CANDIDATE_CSV,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fields
            )

            writer.writeheader()

            for row in candidate_output:

                writer.writerow({
                    key: row.get(key, "")
                    for key in fields
                })

    # --------------------------------------------------------
    # SAVE TRACK CONSENSUS
    # --------------------------------------------------------

    if consensus_output:

        fields = [
            "track_id",
            "vehicle",
            "candidate_count",
            "final_plate",
            "status",
            "support",
            "valid_candidates",
            "consensus_ratio"
        ]

        with open(
            CONSENSUS_CSV,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fields
            )

            writer.writeheader()
            writer.writerows(
                consensus_output
            )

    print("\n" + "=" * 70)
    print("CONSENSUS COMPLETE")
    print("=" * 70)

    print(
        "Candidate OCR:",
        CANDIDATE_CSV
    )

    print(
        "Track results:",
        CONSENSUS_CSV
    )


if __name__ == "__main__":
    main()