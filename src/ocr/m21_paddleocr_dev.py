from pathlib import Path
import csv
import re
import time

import cv2
import paddle

from paddleocr import PaddleOCR


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

GT_CSV = (
    ROOT
    / "outputs"
    / "M20_ocr_dataset"
    / "M20E_ground_truth.csv"
)

IMAGE_DIR = (
    ROOT
    / "data"
    / "processed"
    / "m20_ocr_benchmark"
    / "dev"
    / "images"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "M21_ocr_benchmark"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_CSV = (
    OUTPUT_DIR
    / "M21C_paddleocr_dev_results.csv"
)

SUMMARY_TXT = (
    OUTPUT_DIR
    / "M21C_paddleocr_dev_summary.txt"
)


# ============================================================
# HELPERS
# ============================================================

def clean_text(text):

    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(text).upper(),
    )


def levenshtein(a, b):

    previous = list(
        range(
            len(b) + 1
        )
    )

    for i, char_a in enumerate(
        a,
        start=1,
    ):

        current = [i]

        for j, char_b in enumerate(
            b,
            start=1,
        ):

            insert_cost = (
                current[j - 1]
                + 1
            )

            delete_cost = (
                previous[j]
                + 1
            )

            substitute_cost = (
                previous[j - 1]
                + (
                    char_a
                    != char_b
                )
            )

            current.append(
                min(
                    insert_cost,
                    delete_cost,
                    substitute_cost,
                )
            )

        previous = current

    return previous[-1]


def synchronize_gpu():

    if paddle.device.is_compiled_with_cuda():

        try:
            paddle.device.cuda.synchronize()

        except Exception:
            pass


def get_result_data(result):

    data = result.json

    if callable(data):
        data = data()

    if "res" in data:
        data = data["res"]

    return data


def extract_prediction(
    ocr,
    image,
):

    predictions = ocr.predict(
        image
    )

    regions = []

    for prediction in predictions:

        data = get_result_data(
            prediction
        )

        texts = data.get(
            "rec_texts",
            [],
        )

        scores = data.get(
            "rec_scores",
            [],
        )

        boxes = data.get(
            "rec_boxes",
            [],
        )


        for index, text in enumerate(
            texts
        ):

            cleaned = clean_text(
                text
            )

            if not cleaned:
                continue


            confidence = (
                float(
                    scores[index]
                )
                if index < len(scores)
                else 0.0
            )


            if (
                boxes is not None
                and index < len(boxes)
            ):

                box = boxes[
                    index
                ]

                x1 = float(
                    box[0]
                )

                y1 = float(
                    box[1]
                )

                x2 = float(
                    box[2]
                )

                y2 = float(
                    box[3]
                )

            else:

                x1 = float(
                    index
                )

                y1 = 0.0

                x2 = float(
                    index + 1
                )

                y2 = 1.0


            regions.append({
                "text":
                    cleaned,

                "confidence":
                    confidence,

                "x":
                    (
                        x1 + x2
                    ) / 2.0,

                "y":
                    (
                        y1 + y2
                    ) / 2.0,

                "height":
                    max(
                        y2 - y1,
                        1.0,
                    ),
            })


    if not regions:

        return (
            "",
            0.0,
            0,
        )


    # ========================================================
    # GROUP DETECTED TEXT INTO LINES
    # ========================================================

    regions.sort(
        key=lambda item:
            (
                item["y"],
                item["x"],
            )
    )


    heights = sorted(
        item["height"]
        for item in regions
    )

    median_height = (
        heights[
            len(heights) // 2
        ]
    )


    row_tolerance = max(
        10.0,
        median_height * 0.60,
    )


    lines = []

    for region in regions:

        assigned = False

        for line in lines:

            mean_y = (
                sum(
                    item["y"]
                    for item in line
                )
                / len(line)
            )

            if (
                abs(
                    region["y"]
                    - mean_y
                )
                <= row_tolerance
            ):

                line.append(
                    region
                )

                assigned = True

                break


        if not assigned:

            lines.append(
                [region]
            )


    lines.sort(
        key=lambda line:
            (
                sum(
                    item["y"]
                    for item in line
                )
                / len(line)
            )
    )


    combined_lines = []

    weighted_confidence = 0.0

    character_count = 0


    for line in lines:

        line.sort(
            key=lambda item:
                item["x"]
        )

        line_text = "".join(
            item["text"]
            for item in line
        )

        combined_lines.append(
            line_text
        )


        for item in line:

            length = len(
                item["text"]
            )

            weighted_confidence += (
                item[
                    "confidence"
                ]
                * length
            )

            character_count += (
                length
            )


    prediction = clean_text(
        "".join(
            combined_lines
        )
    )


    confidence = (
        weighted_confidence
        / character_count
        if character_count
        else 0.0
    )


    return (
        prediction,
        confidence,
        len(regions),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)

    print(
        "M21C — PADDLEOCR RAW DEV BENCHMARK"
    )

    print("=" * 72)


    # ========================================================
    # LOAD VERIFIED DEV GT ONLY
    # ========================================================

    with open(
        GT_CSV,
        newline="",
        encoding="utf-8",
    ) as file:

        all_gt = list(
            csv.DictReader(file)
        )


    evaluation = [
        row
        for row in all_gt

        if (
            row[
                "benchmark_split"
            ]
            == "dev"

            and row[
                "status"
            ]
            == "VERIFIED"
        )
    ]


    print(
        "\nVerified DEV samples:",
        len(evaluation),
    )


    if len(evaluation) != 26:

        raise RuntimeError(
            f"Expected 26 verified DEV samples, "
            f"found {len(evaluation)}."
        )


    # ========================================================
    # LOAD PADDLEOCR ONCE
    # ========================================================

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

        device="gpu:0",
    )


    # ========================================================
    # WARMUP
    # ========================================================

    warmup_path = (
        IMAGE_DIR
        / (
            evaluation[0][
                "sample_id"
            ]
            + ".jpg"
        )
    )


    warmup_image = cv2.imread(
        str(
            warmup_path
        )
    )


    print(
        "Running GPU warmup..."
    )


    for _ in range(3):

        ocr.predict(
            warmup_image
        )


    synchronize_gpu()


    # ========================================================
    # EVALUATION
    # ========================================================

    result_rows = []

    exact_matches = 0

    total_edit_distance = 0

    total_gt_characters = 0

    blank_predictions = 0

    latencies = []


    for index, row in enumerate(
        evaluation,
        start=1,
    ):

        sample_id = row[
            "sample_id"
        ]


        ground_truth = clean_text(
            row[
                "ground_truth"
            ]
        )


        image_path = (
            IMAGE_DIR
            / f"{sample_id}.jpg"
        )


        image = cv2.imread(
            str(
                image_path
            )
        )


        if image is None:

            raise RuntimeError(
                f"Could not load: "
                f"{image_path}"
            )


        # ====================================================
        # OCR LATENCY
        #
        # Model loading and warmup excluded.
        # ====================================================

        synchronize_gpu()

        start = time.perf_counter()


        prediction, confidence, regions = (
            extract_prediction(
                ocr,
                image,
            )
        )


        synchronize_gpu()

        latency_ms = (
            time.perf_counter()
            - start
        ) * 1000.0


        edit_distance = (
            levenshtein(
                ground_truth,
                prediction,
            )
        )


        exact = (
            prediction
            == ground_truth
        )


        if exact:
            exact_matches += 1


        if not prediction:
            blank_predictions += 1


        total_edit_distance += (
            edit_distance
        )


        total_gt_characters += (
            len(
                ground_truth
            )
        )


        latencies.append(
            latency_ms
        )


        result_rows.append({
            "sample_id":
                sample_id,

            "ground_truth":
                ground_truth,

            "prediction":
                prediction,

            "exact_match":
                exact,

            "edit_distance":
                edit_distance,

            "gt_length":
                len(
                    ground_truth
                ),

            "ocr_confidence":
                round(
                    confidence,
                    6,
                ),

            "text_regions":
                regions,

            "latency_ms":
                round(
                    latency_ms,
                    3,
                ),
        })


        symbol = (
            "MATCH"
            if exact
            else "MISS"
        )


        print(
            f"[{index:02d}/"
            f"{len(evaluation)}] "
            f"{sample_id} | "
            f"GT={ground_truth} | "
            f"PRED="
            f"{prediction or 'NONE'} | "
            f"{symbol} | "
            f"{latency_ms:.1f} ms"
        )


    # ========================================================
    # SAVE PER-SAMPLE RESULTS
    # ========================================================

    with open(
        RESULT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=(
                result_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            result_rows
        )


    # ========================================================
    # METRICS
    # ========================================================

    exact_accuracy = (
        exact_matches
        / len(
            evaluation
        )
    )


    cer = (
        total_edit_distance
        / total_gt_characters
    )


    mean_latency = (
        sum(
            latencies
        )
        / len(
            latencies
        )
    )


    sorted_latency = sorted(
        latencies
    )


    median_latency = (
        sorted_latency[
            len(sorted_latency)
            // 2
        ]
    )


    blank_rate = (
        blank_predictions
        / len(
            evaluation
        )
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = f"""M21C — PADDLEOCR RAW DEV BENCHMARK
============================================================

Evaluation split
----------------
OCR DEV only

Verified readable samples:
{len(evaluation)}

OCR engine
----------
PaddleOCR 3.7.0

Detector:
PP-OCRv5_server_det

Recognizer:
PP-OCRv5_server_rec

Device:
NVIDIA RTX 4050 Laptop GPU

Preprocessing
-------------
NONE

Grammar correction
------------------
NONE

Orientation correction
----------------------
NONE

Prediction normalization
------------------------
Uppercase
Remove spaces/punctuation/non-alphanumeric characters

Results
-------
Exact matches:
{exact_matches}/{len(evaluation)}

Exact-match accuracy:
{exact_accuracy:.4f}
({exact_accuracy * 100:.2f}%)

Total GT characters:
{total_gt_characters}

Total Levenshtein edit distance:
{total_edit_distance}

Character Error Rate (CER):
{cer:.4f}
({cer * 100:.2f}%)

Blank predictions:
{blank_predictions}

Blank prediction rate:
{blank_rate:.4f}
({blank_rate * 100:.2f}%)

Runtime
-------
Model initialization excluded.
Three warmup inferences excluded.

Mean latency per crop:
{mean_latency:.2f} ms

Median latency per crop:
{median_latency:.2f} ms

Methodology
-----------
Only manually VERIFIED DEV samples were scored.

UNREADABLE samples were excluded.

OCR TEST was NOT evaluated.

No preprocessing, orientation correction, grammar repair,
or OCR configuration was selected using TEST data.
"""


    SUMMARY_TXT.write_text(
        summary,
        encoding="utf-8",
    )


    # ========================================================
    # PRINT RESULT
    # ========================================================

    print("\n" + "=" * 72)

    print(
        "M21C COMPLETE"
    )

    print("=" * 72)


    print(
        f"\nExact matches : "
        f"{exact_matches}/"
        f"{len(evaluation)}"
    )


    print(
        f"Exact accuracy: "
        f"{exact_accuracy:.4f} "
        f"({exact_accuracy * 100:.2f}%)"
    )


    print(
        f"CER           : "
        f"{cer:.4f} "
        f"({cer * 100:.2f}%)"
    )


    print(
        f"Blank rate    : "
        f"{blank_rate:.4f}"
    )


    print(
        f"Mean latency  : "
        f"{mean_latency:.2f} ms"
    )


    print(
        f"Median latency: "
        f"{median_latency:.2f} ms"
    )


    print(
        "\nResults:"
    )

    print(
        RESULT_CSV
    )


    print(
        "\nSummary:"
    )

    print(
        SUMMARY_TXT
    )


if __name__ == "__main__":
    main()