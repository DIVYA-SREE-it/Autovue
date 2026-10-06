from pathlib import Path
from collections import defaultdict
import csv
import re

import cv2
import numpy as np
from paddleocr import PaddleOCR


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRACK_CSV = (
    ROOT
    / "outputs"
    / "M09_ocr_video_tracking"
    / "track_summary.csv"
)

CROP_DIR = (
    ROOT
    / "outputs"
    / "M09_ocr_video_tracking"
    / "best_plate_crops"
)

GROUND_TRUTH_CSV = (
    ROOT
    / "outputs"
    / "M12_manual_ground_truth"
    / "verified_plates.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "M13_two_line_temporal_fusion"
)

VARIANT_CSV = OUTPUT_DIR / "variant_ocr.csv"
FUSION_CSV = OUTPUT_DIR / "track_fusion.csv"
EVAL_CSV = OUTPUT_DIR / "evaluation.csv"


# ============================================================
# INDIAN REGISTRATION KNOWLEDGE
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


DIGIT_MAP = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8",
}


LETTER_MAP = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
}


# ============================================================
# HELPERS
# ============================================================

def clean_text(text):
    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(text).upper()
    )


def convert_char(character, expected):
    """
    expected:
        L -> letter
        D -> digit

    Returns:
        converted_character, correction_cost
    """

    character = character.upper()

    if expected == "L":

        if character.isalpha():
            return character, 0

        if character in LETTER_MAP:
            return LETTER_MAP[character], 1

        return None, 99

    if expected == "D":

        if character.isdigit():
            return character, 0

        if character in DIGIT_MAP:
            return DIGIT_MAP[character], 1

        return None, 99

    return None, 99


def apply_mask(text, mask):
    if len(text) != len(mask):
        return None

    output = []
    total_cost = 0

    for character, expected in zip(text, mask):

        converted, cost = convert_char(
            character,
            expected
        )

        if converted is None:
            return None

        output.append(converted)
        total_cost += cost

    return "".join(output), total_cost


def normalize_indian_plate(text):
    """
    Searches the OCR result for a plausible standard Indian
    registration and performs position-aware character repair.

    Example:
        APIIZ3837
             ↓
        AP11Z3837
    """

    text = clean_text(text)

    if not text:
        return "", -1

    candidates = []

    # Standard format:
    # SS DD SERIES NUMBER
    #
    # Example:
    # AP 31 AE 7144
    # AP 11 Z 3837

    for series_len in [1, 2, 3]:

        for number_len in [1, 2, 3, 4]:

            mask = (
                "LL"
                + "DD"
                + "L" * series_len
                + "D" * number_len
            )

            target_len = len(mask)

            if len(text) < target_len:
                continue

            for start in range(
                len(text) - target_len + 1
            ):

                substring = text[
                    start:start + target_len
                ]

                converted = apply_mask(
                    substring,
                    mask
                )

                if converted is None:
                    continue

                plate, correction_cost = (
                    converted
                )

                state = plate[:2]

                if state not in STATE_CODES:
                    continue

                removed_characters = (
                    len(text)
                    - target_len
                )

                # Prefer:
                # 1. fewer OCR corrections
                # 2. less removed noise
                # 3. longer complete plates
                candidate_score = (
                    correction_cost
                    + removed_characters * 0.35
                    - target_len * 0.01
                )

                candidates.append({
                    "plate": plate,
                    "cost": correction_cost,
                    "score": candidate_score
                })

    # Bharat-series direct support.
    if re.fullmatch(
        r"[0-9]{2}BH[0-9]{4}[A-Z]{2}",
        text
    ):
        candidates.append({
            "plate": text,
            "cost": 0,
            "score": -0.1
        })

    if not candidates:
        return "", -1

    candidates.sort(
        key=lambda item: item["score"]
    )

    best = candidates[0]

    return (
        best["plate"],
        best["cost"]
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocessing_variants(image):

    variants = {}

    variants["original"] = image

    # --------------------------------------------------------
    # UPSCALE
    # --------------------------------------------------------

    upscaled = cv2.resize(
        image,
        None,
        fx=3.0,
        fy=3.0,
        interpolation=cv2.INTER_CUBIC
    )

    variants["upscaled"] = upscaled

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        upscaled,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    clahe_gray = clahe.apply(gray)

    variants["clahe"] = cv2.cvtColor(
        clahe_gray,
        cv2.COLOR_GRAY2BGR
    )

    # --------------------------------------------------------
    # SHARPEN
    # --------------------------------------------------------

    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpened = cv2.filter2D(
        upscaled,
        -1,
        sharpen_kernel
    )

    variants["sharpened"] = sharpened

    # --------------------------------------------------------
    # OTSU THRESHOLD
    # --------------------------------------------------------

    _, threshold = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )

    variants["threshold"] = cv2.cvtColor(
        threshold,
        cv2.COLOR_GRAY2BGR
    )

    return variants


# ============================================================
# PADDLE RESULT HANDLING
# ============================================================

def get_result_data(result):

    data = result.json

    if callable(data):
        data = data()

    if "res" in data:
        data = data["res"]

    return data


def extract_lines(ocr, image):

    predictions = ocr.predict(image)

    regions = []

    for prediction in predictions:

        data = get_result_data(
            prediction
        )

        texts = data.get(
            "rec_texts",
            []
        )

        scores = data.get(
            "rec_scores",
            []
        )

        boxes = data.get(
            "rec_boxes",
            []
        )

        for index, text in enumerate(texts):

            cleaned = clean_text(text)

            if not cleaned:
                continue

            confidence = (
                float(scores[index])
                if index < len(scores)
                else 0.0
            )

            if (
                boxes is not None
                and len(boxes) > index
            ):

                box = boxes[index]

                x1 = float(box[0])
                y1 = float(box[1])
                x2 = float(box[2])
                y2 = float(box[3])

            else:

                x1 = float(index)
                y1 = 0.0
                x2 = float(index + 1)
                y2 = 1.0

            regions.append({
                "text": cleaned,
                "confidence": confidence,
                "x": (x1 + x2) / 2.0,
                "y": (y1 + y2) / 2.0,
                "height": max(
                    y2 - y1,
                    1.0
                )
            })

    if not regions:
        return [], "", 0.0

    # --------------------------------------------------------
    # GROUP OCR REGIONS INTO TEXT ROWS
    # --------------------------------------------------------

    regions.sort(
        key=lambda item: (
            item["y"],
            item["x"]
        )
    )

    median_height = np.median([
        item["height"]
        for item in regions
    ])

    row_tolerance = max(
        10.0,
        median_height * 0.6
    )

    rows = []

    for region in regions:

        assigned = False

        for row in rows:

            mean_y = np.mean([
                item["y"]
                for item in row
            ])

            if abs(
                region["y"] - mean_y
            ) <= row_tolerance:

                row.append(region)
                assigned = True
                break

        if not assigned:
            rows.append([region])

    # Top → bottom
    rows.sort(
        key=lambda row: np.mean([
            item["y"]
            for item in row
        ])
    )

    line_texts = []
    weighted_confidence_sum = 0.0
    character_count = 0

    for row in rows:

        # Left → right inside each row
        row.sort(
            key=lambda item: item["x"]
        )

        line = "".join(
            item["text"]
            for item in row
        )

        line_texts.append(line)

        for item in row:

            length = len(
                item["text"]
            )

            weighted_confidence_sum += (
                item["confidence"]
                * length
            )

            character_count += length

    combined = "".join(
        line_texts
    )

    confidence = (
        weighted_confidence_sum
        / max(character_count, 1)
    )

    return (
        line_texts,
        combined,
        confidence
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 74)
    print(
        "M13 - TWO-LINE + "
        "GRAMMAR-AWARE TEMPORAL FUSION"
    )
    print("=" * 74)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD MANUAL EVALUATION ANCHORS
    # --------------------------------------------------------

    with open(
        GROUND_TRUTH_CSV,
        "r",
        encoding="utf-8"
    ) as file:

        ground_truth_rows = list(
            csv.DictReader(file)
        )

    target_tracks = {
        row["track_id"]
        for row in ground_truth_rows
    }

    ground_truth = {
        row["track_id"]:
        clean_text(
            row["ground_truth"]
        )
        for row in ground_truth_rows
    }

    print(
        "Evaluation tracks:",
        sorted(target_tracks)
    )

    # --------------------------------------------------------
    # LOAD ONLY VERIFIED TRACK CANDIDATES
    # --------------------------------------------------------

    with open(
        TRACK_CSV,
        "r",
        encoding="utf-8"
    ) as file:

        all_rows = list(
            csv.DictReader(file)
        )

    track_rows = [
        row
        for row in all_rows
        if row["track_id"]
        in target_tracks
    ]

    print(
        "Candidate crops:",
        len(track_rows)
    )

    # --------------------------------------------------------
    # LOAD PADDLEOCR
    # --------------------------------------------------------

    print(
        "\nLoading PaddleOCR..."
    )

    ocr = PaddleOCR(
        text_detection_model_name=(
            "PP-OCRv5_server_det"
        ),
        text_recognition_model_name=(
            "PP-OCRv5_server_rec"
        ),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="gpu:0"
    )

    variant_rows = []

    # plate → data
    votes = defaultdict(
        lambda: defaultdict(
            lambda: {
                "frames": set(),
                "variants": 0,
                "total_score": 0.0,
                "ocr_confidences": [],
                "correction_costs": []
            }
        )
    )

    # --------------------------------------------------------
    # OCR ALL PREPROCESSING VARIANTS
    # --------------------------------------------------------

    for crop_index, row in enumerate(
        track_rows,
        start=1
    ):

        image_path = (
            CROP_DIR
            / row["filename"]
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                "Could not read:",
                image_path
            )
            continue

        variants = preprocessing_variants(
            image
        )

        print(
            f"\n[{crop_index}/"
            f"{len(track_rows)}] "
            f"Track {row['track_id']} "
            f"frame {row['frame']}"
        )

        for variant_name, variant_image in (
            variants.items()
        ):

            (
                lines,
                raw_text,
                ocr_confidence
            ) = extract_lines(
                ocr,
                variant_image
            )

            (
                normalized_plate,
                correction_cost
            ) = normalize_indian_plate(
                raw_text
            )

            print(
                f"  {variant_name:10s} "
                f"lines={lines} "
                f"=> {raw_text or 'NO_TEXT'} "
                f"=> "
                f"{normalized_plate or '-'} "
                f"({ocr_confidence:.3f})"
            )

            variant_rows.append({
                "track_id": row[
                    "track_id"
                ],
                "frame": row[
                    "frame"
                ],
                "rank": row[
                    "rank"
                ],
                "vehicle": row[
                    "vehicle"
                ],
                "filename": row[
                    "filename"
                ],
                "variant": variant_name,
                "ocr_lines": (
                    " | ".join(lines)
                ),
                "raw_text": raw_text,
                "ocr_confidence": round(
                    ocr_confidence,
                    4
                ),
                "normalized_plate": (
                    normalized_plate
                ),
                "correction_cost": (
                    correction_cost
                ),
                "quality_score": row[
                    "quality_score"
                ],
                "plate_confidence": row[
                    "plate_confidence"
                ]
            })

            if not normalized_plate:
                continue

            quality_score = float(
                row["quality_score"]
            )

            plate_confidence = float(
                row["plate_confidence"]
            )

            # Weighted evidence.
            score = (
                ocr_confidence * 0.55
                + quality_score * 0.25
                + plate_confidence * 0.20
                - max(
                    correction_cost,
                    0
                ) * 0.05
            )

            entry = votes[
                row["track_id"]
            ][normalized_plate]

            entry["frames"].add(
                int(row["frame"])
            )

            entry["variants"] += 1

            entry["total_score"] += score

            entry[
                "ocr_confidences"
            ].append(
                ocr_confidence
            )

            entry[
                "correction_costs"
            ].append(
                correction_cost
            )

    # --------------------------------------------------------
    # SAVE VARIANT OCR
    # --------------------------------------------------------

    if variant_rows:

        with open(
            VARIANT_CSV,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=(
                    variant_rows[
                        0
                    ].keys()
                )
            )

            writer.writeheader()
            writer.writerows(
                variant_rows
            )

    # --------------------------------------------------------
    # TRACK-LEVEL TEMPORAL CONSENSUS
    # --------------------------------------------------------

    fusion_rows = []

    for track_id in sorted(
        target_tracks,
        key=int
    ):

        candidates = votes.get(
            track_id,
            {}
        )

        if not candidates:

            fusion_rows.append({
                "track_id": track_id,
                "final_plate": "",
                "status": "REJECTED",
                "frame_support": 0,
                "variant_support": 0,
                "mean_ocr_confidence": 0.0,
                "mean_correction_cost": 0.0,
                "fusion_score": 0.0
            })

            continue

        ranked = []

        for plate, data in (
            candidates.items()
        ):

            frame_support = len(
                data["frames"]
            )

            mean_ocr_confidence = (
                sum(
                    data[
                        "ocr_confidences"
                    ]
                )
                / len(
                    data[
                        "ocr_confidences"
                    ]
                )
            )

            mean_cost = (
                sum(
                    data[
                        "correction_costs"
                    ]
                )
                / len(
                    data[
                        "correction_costs"
                    ]
                )
            )

            # Distinct frames matter much more
            # than preprocessing variants.
            fusion_score = (
                frame_support * 2.0
                + data["variants"] * 0.15
                + data["total_score"]
            )

            ranked.append({
                "plate": plate,
                "frame_support": (
                    frame_support
                ),
                "variant_support": (
                    data["variants"]
                ),
                "mean_ocr_confidence": (
                    mean_ocr_confidence
                ),
                "mean_correction_cost": (
                    mean_cost
                ),
                "fusion_score": (
                    fusion_score
                )
            })

        ranked.sort(
            key=lambda item: (
                item[
                    "frame_support"
                ],
                item[
                    "fusion_score"
                ]
            ),
            reverse=True
        )

        best = ranked[0]

        if (
            best["frame_support"] >= 2
            and best[
                "mean_ocr_confidence"
            ] >= 0.60
        ):
            status = "VERIFIED"

        elif (
            best["frame_support"] >= 1
            and best[
                "mean_ocr_confidence"
            ] >= 0.50
        ):
            status = "NEEDS_REVIEW"

        else:
            status = "REJECTED"

        fusion_rows.append({
            "track_id": track_id,
            "final_plate": best[
                "plate"
            ],
            "status": status,
            "frame_support": best[
                "frame_support"
            ],
            "variant_support": best[
                "variant_support"
            ],
            "mean_ocr_confidence": round(
                best[
                    "mean_ocr_confidence"
                ],
                4
            ),
            "mean_correction_cost": round(
                best[
                    "mean_correction_cost"
                ],
                2
            ),
            "fusion_score": round(
                best[
                    "fusion_score"
                ],
                4
            )
        })

    # --------------------------------------------------------
    # SAVE FUSION RESULTS
    # --------------------------------------------------------

    with open(
        FUSION_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=(
                fusion_rows[
                    0
                ].keys()
            )
        )

        writer.writeheader()
        writer.writerows(
            fusion_rows
        )

    # --------------------------------------------------------
    # EVALUATION AGAINST MANUAL LABELS
    # Ground truth is used ONLY HERE.
    # --------------------------------------------------------

    evaluation_rows = []

    correct = 0

    for row in fusion_rows:

        track_id = row[
            "track_id"
        ]

        gt = ground_truth[
            track_id
        ]

        prediction = clean_text(
            row["final_plate"]
        )

        exact_match = (
            prediction == gt
        )

        if exact_match:
            correct += 1

        evaluation_rows.append({
            "track_id": track_id,
            "ground_truth": gt,
            "prediction": prediction,
            "status": row[
                "status"
            ],
            "exact_match": (
                exact_match
            )
        })

    with open(
        EVAL_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=(
                evaluation_rows[
                    0
                ].keys()
            )
        )

        writer.writeheader()
        writer.writerows(
            evaluation_rows
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 74)
    print("M13 COMPLETE")
    print("=" * 74)

    for row in evaluation_rows:

        symbol = (
            "MATCH"
            if row[
                "exact_match"
            ]
            else "MISS"
        )

        print(
            f"Track {row['track_id']}: "
            f"GT={row['ground_truth']} | "
            f"PRED="
            f"{row['prediction'] or 'NONE'} "
            f"| {row['status']} "
            f"| {symbol}"
        )

    print(
        f"\nExact matches: "
        f"{correct}/"
        f"{len(evaluation_rows)}"
    )

    print(
        "\nVariant OCR:",
        VARIANT_CSV
    )

    print(
        "Fusion:",
        FUSION_CSV
    )

    print(
        "Evaluation:",
        EVAL_CSV
    )


if __name__ == "__main__":
    main()
