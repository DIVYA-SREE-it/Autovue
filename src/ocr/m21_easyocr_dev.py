from pathlib import Path
import csv
import re
import time

import cv2
import easyocr
import torch


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
    / "M21B_easyocr_dev_results.csv"
)

SUMMARY_TXT = (
    OUTPUT_DIR
    / "M21B_easyocr_dev_summary.txt"
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


def center(box):

    xs = [
        point[0]
        for point in box
    ]

    ys = [
        point[1]
        for point in box
    ]

    return (
        sum(xs) / len(xs),
        sum(ys) / len(ys),
    )


def combine_easyocr_results(results):

    """
    Combine EasyOCR text regions in approximate
    top-to-bottom, then left-to-right reading order.
    """

    regions = []

    for item in results:

        if len(item) < 3:
            continue

        box = item[0]

        text = clean_text(
            item[1]
        )

        confidence = float(
            item[2]
        )

        if not text:
            continue

        x, y = center(
            box
        )

        height = max(
            point[1]
            for point in box
        ) - min(
            point[1]
            for point in box
        )

        regions.append({
            "text":
                text,

            "confidence":
                confidence,

            "x":
                x,

            "y":
                y,

            "height":
                max(
                    height,
                    1.0,
                ),
        })


    if not regions:

        return "", 0.0, 0


    # --------------------------------------------------------
    # GROUP INTO TEXT LINES
    # --------------------------------------------------------

    regions.sort(
        key=lambda item:
            item["y"]
    )

    lines = []

    for region in regions:

        assigned = False

        for line in lines:

            line_y = sum(
                item["y"]
                for item in line
            ) / len(line)

            mean_height = sum(
                item["height"]
                for item in line
            ) / len(line)

            if (
                abs(
                    region["y"]
                    - line_y
                )
                <=
                0.60
                * max(
                    region[
                        "height"
                    ],
                    mean_height,
                )
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


    # Top → bottom.
    lines.sort(
        key=lambda line:
            sum(
                item["y"]
                for item in line
            ) / len(line)
    )


    texts = []

    weighted_confidence = 0.0

    character_count = 0


    for line in lines:

        # Left → right.
        line.sort(
            key=lambda item:
                item["x"]
        )

        line_text = "".join(
            item["text"]
            for item in line
        )

        texts.append(
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


    combined = clean_text(
        "".join(
            texts
        )
    )

    confidence = (
        weighted_confidence
        / character_count
        if character_count
        else 0.0
    )

    return (
        combined,
        confidence,
        len(regions),
    )


def synchronize_gpu():

    if torch.cuda.is_available():

        torch.cuda.synchronize()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("M21B — EASYOCR RAW DEV BENCHMARK")
    print("=" * 72)


    # --------------------------------------------------------
    # LOAD ONLY VERIFIED DEV GT
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # INITIALIZE ONCE
    # --------------------------------------------------------

    print(
        "\nLoading EasyOCR..."
    )

    reader = easyocr.Reader(
        ["en"],
        gpu=True,
    )


    # --------------------------------------------------------
    # GPU WARMUP
    # --------------------------------------------------------

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

        reader.readtext(
            warmup_image,
            detail=1,
            paragraph=False,
        )

    synchronize_gpu()


    # --------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------

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


        # ----------------------------------------------------
        # END-TO-END OCR CALL LATENCY
        #
        # Model loading is excluded.
        # ----------------------------------------------------

        synchronize_gpu()

        start = time.perf_counter()

        raw_result = (
            reader.readtext(
                image,
                detail=1,
                paragraph=False,
            )
        )

        synchronize_gpu()

        latency_ms = (
            time.perf_counter()
            - start
        ) * 1000.0


        prediction, confidence, regions = (
            combine_easyocr_results(
                raw_result
            )
        )


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


    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

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


    summary = f"""M21B — EASYOCR RAW DEV BENCHMARK
============================================================

Evaluation split
----------------
OCR DEV only

Verified readable samples:
{len(evaluation)}

OCR engine
----------
EasyOCR 1.7.2
Language: English
GPU: NVIDIA RTX 4050 Laptop GPU

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

UNREADABLE samples were excluded because reliable
character-level ground truth does not exist for them.

OCR TEST was NOT evaluated.

No OCR configuration or post-processing was selected
using TEST data.
"""


    SUMMARY_TXT.write_text(
        summary,
        encoding="utf-8",
    )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("M21B COMPLETE")
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

    print("\nResults:")
    print(RESULT_CSV)

    print("\nSummary:")
    print(SUMMARY_TXT)


if __name__ == "__main__":
    main()