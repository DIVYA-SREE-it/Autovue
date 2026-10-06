from pathlib import Path
import csv
import re

import numpy as np
from paddleocr import PaddleOCR


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRACK_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "M09_ocr_video_tracking"
    / "track_summary.csv"
)

CROP_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "M09_ocr_video_tracking"
    / "best_plate_crops"
)

EASYOCR_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "M10_ocr_video_consensus"
    / "candidate_ocr.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "M11_paddleocr_comparison"
)

PADDLE_CSV = (
    OUTPUT_DIR
    / "paddle_candidate_ocr.csv"
)

COMPARISON_CSV = (
    OUTPUT_DIR
    / "easy_vs_paddle.csv"
)


# ============================================================
# HELPERS
# ============================================================

def clean_text(text):

    return re.sub(
        r"[^A-Z0-9]",
        "",
        text.upper()
    )


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


def valid_indian_format(text):

    text = clean_text(text)

    standard = re.fullmatch(
        r"([A-Z]{2})([0-9]{2})([A-Z]{1,3})([0-9]{1,4})",
        text
    )

    if standard:

        state = standard.group(1)

        return state in STATE_CODES

    bharat = re.fullmatch(
        r"[0-9]{2}BH[0-9]{4}[A-Z]{2}",
        text
    )

    return bool(bharat)


def get_result_data(result):

    data = result.json

    # Defensive compatibility
    if callable(data):
        data = data()

    if "res" in data:
        data = data["res"]

    return data


def run_paddle_ocr(ocr, image_path):

    predictions = ocr.predict(
        str(image_path)
    )

    pieces = []

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

            cleaned = clean_text(
                text
            )

            if not cleaned:
                continue

            score = (
                float(scores[index])
                if index < len(scores)
                else 0.0
            )

            # Use detected box position so two-line
            # plates are assembled top-to-bottom.
            if (
                boxes is not None
                and len(boxes) > index
            ):

                box = boxes[index]

                x1 = float(box[0])
                y1 = float(box[1])

            else:

                x1 = index
                y1 = 0

            pieces.append({
                "text": cleaned,
                "score": score,
                "x": x1,
                "y": y1
            })

    if not pieces:

        return "", 0.0, 0

    # Reading order
    pieces.sort(
        key=lambda item: (
            round(
                item["y"] / 20
            ),
            item["x"]
        )
    )

    combined_text = "".join(
        item["text"]
        for item in pieces
    )

    # Character-length weighted confidence
    total_characters = sum(
        len(item["text"])
        for item in pieces
    )

    confidence = sum(
        item["score"]
        * len(item["text"])
        for item in pieces
    ) / max(
        total_characters,
        1
    )

    return (
        combined_text,
        confidence,
        len(pieces)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("M11 - EASYOCR VS PADDLEOCR")
    print("=" * 70)

    if not TRACK_CSV.exists():
        raise FileNotFoundError(
            TRACK_CSV
        )

    if not CROP_DIR.exists():
        raise FileNotFoundError(
            CROP_DIR
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD PADDLEOCR
    # --------------------------------------------------------

    print(
        "\nLoading PaddleOCR "
        "server detection + recognition..."
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

    # --------------------------------------------------------
    # READ TRACK METADATA
    # --------------------------------------------------------

    with open(
        TRACK_CSV,
        "r",
        encoding="utf-8"
    ) as file:

        track_rows = list(
            csv.DictReader(file)
        )

    print(
        "Candidate crops:",
        len(track_rows)
    )

    # --------------------------------------------------------
    # LOAD EXISTING EASYOCR RESULTS
    # --------------------------------------------------------

    easy_results = {}

    if EASYOCR_CSV.exists():

        with open(
            EASYOCR_CSV,
            "r",
            encoding="utf-8"
        ) as file:

            for row in csv.DictReader(file):

                easy_results[
                    row["filename"]
                ] = row

    # --------------------------------------------------------
    # RUN PADDLEOCR
    # --------------------------------------------------------

    paddle_rows = []
    comparison_rows = []

    detected_count = 0
    exact_agreement = 0
    paddle_valid_count = 0

    for index, row in enumerate(
        track_rows,
        start=1
    ):

        filename = row["filename"]

        image_path = (
            CROP_DIR
            / filename
        )

        text, confidence, regions = (
            run_paddle_ocr(
                ocr,
                image_path
            )
        )

        is_valid = (
            valid_indian_format(
                text
            )
        )

        if text:
            detected_count += 1

        if is_valid:
            paddle_valid_count += 1

        paddle_row = {
            "track_id": row[
                "track_id"
            ],
            "rank": row["rank"],
            "frame": row["frame"],
            "vehicle": row[
                "vehicle"
            ],
            "filename": filename,
            "paddle_text": text,
            "paddle_confidence": round(
                confidence,
                4
            ),
            "text_regions": regions,
            "indian_format": is_valid,
            "quality_score": row[
                "quality_score"
            ],
            "plate_confidence": row[
                "plate_confidence"
            ]
        }

        paddle_rows.append(
            paddle_row
        )

        easy = easy_results.get(
            filename,
            {}
        )

        easy_text = clean_text(
            easy.get(
                "raw_text",
                ""
            )
        )

        agreement = (
            bool(text)
            and text == easy_text
        )

        if agreement:
            exact_agreement += 1

        comparison_rows.append({
            "track_id": row[
                "track_id"
            ],
            "rank": row["rank"],
            "filename": filename,

            "easy_text": easy_text,
            "easy_confidence": easy.get(
                "ocr_confidence",
                ""
            ),

            "paddle_text": text,
            "paddle_confidence": round(
                confidence,
                4
            ),

            "exact_agreement": agreement,

            "easy_indian_format": (
                valid_indian_format(
                    easy_text
                )
            ),

            "paddle_indian_format": (
                is_valid
            )
        })

        print(
            f"[{index:02d}/"
            f"{len(track_rows)}] "
            f"Track {row['track_id']} "
            f"rank {row['rank']} "
            f"-> "
            f"{text or 'NO_TEXT'} "
            f"({confidence:.3f})"
        )

    # --------------------------------------------------------
    # SAVE PADDLE RESULTS
    # --------------------------------------------------------

    if paddle_rows:

        with open(
            PADDLE_CSV,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=(
                    paddle_rows[
                        0
                    ].keys()
                )
            )

            writer.writeheader()
            writer.writerows(
                paddle_rows
            )

    # --------------------------------------------------------
    # SAVE DIRECT COMPARISON
    # --------------------------------------------------------

    if comparison_rows:

        with open(
            COMPARISON_CSV,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=(
                    comparison_rows[
                        0
                    ].keys()
                )
            )

            writer.writeheader()
            writer.writerows(
                comparison_rows
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("M11 COMPLETE")
    print("=" * 70)

    print(
        "Total crops:",
        len(track_rows)
    )

    print(
        "Paddle detected text:",
        detected_count
    )

    print(
        "Paddle strict-format candidates:",
        paddle_valid_count
    )

    print(
        "Exact Easy/Paddle agreements:",
        exact_agreement
    )

    print(
        "\nPaddle results:",
        PADDLE_CSV
    )

    print(
        "Comparison:",
        COMPARISON_CSV
    )


if __name__ == "__main__":
    main()