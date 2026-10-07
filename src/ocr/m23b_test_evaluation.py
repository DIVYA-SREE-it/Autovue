from pathlib import Path
import csv
import re
import time

import cv2
import paddle
from paddleocr import PaddleOCR

from m21_paddleocr_dev import (
    extract_prediction,
)

from m22a_plate_aware_parser import (
    clean_text,
    extract_indian_plate,
    levenshtein,
)


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
    / "test"
    / "images"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "M23_ocr_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_CSV = (
    OUTPUT_DIR
    / "M23B_test_results.csv"
)

CANDIDATE_CSV = (
    OUTPUT_DIR
    / "M23B_orientation_candidates.csv"
)

SUMMARY_TXT = (
    OUTPUT_DIR
    / "M23B_test_summary.txt"
)


# ============================================================
# HELPERS
# ============================================================

def synchronize_gpu():

    if (
        paddle.device
        .is_compiled_with_cuda()
    ):

        try:

            paddle.device.synchronize()

        except Exception:

            pass


def rotate_image(
    image,
    angle,
):

    if angle == 90:

        return cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE,
        )

    if angle == 180:

        return cv2.rotate(
            image,
            cv2.ROTATE_180,
        )

    if angle == 270:

        return cv2.rotate(
            image,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        )

    return image


def removed_count(
    parser_info,
):

    match = re.search(
        r"removed=(\d+)",
        parser_info or "",
    )

    if not match:

        return 999

    return int(
        match.group(1)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "M23B — ONE-TIME HELD-OUT OCR TEST EVALUATION"
    )
    print("=" * 72)


    # ========================================================
    # LOAD VERIFIED TEST GT
    #
    # Ground truth is loaded ONLY for evaluation after
    # inference decisions.
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
            == "test"

            and row[
                "status"
            ]
            == "VERIFIED"
        )
    ]


    if len(evaluation) != 14:

        raise RuntimeError(
            f"Expected 14 VERIFIED TEST samples, "
            f"found {len(evaluation)}"
        )


    print(
        "\nVerified TEST samples:",
        len(evaluation),
    )


    # ========================================================
    # LOAD FROZEN OCR MODEL
    # ========================================================

    print(
        "\nLoading frozen PaddleOCR configuration..."
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

    warmup_id = (
        evaluation[0][
            "sample_id"
        ]
    )


    warmup_image = cv2.imread(
        str(
            IMAGE_DIR
            / f"{warmup_id}.jpg"
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
    # TEST
    # ========================================================

    result_rows = []

    candidate_rows = []

    exact_matches = 0

    total_edit_distance = 0

    total_gt_characters = 0

    blank_predictions = 0

    orientation_triggered = 0

    orientation_selected = 0

    latencies = []


    for index, row in enumerate(
        evaluation,
        start=1,
    ):

        sample_id = (
            row[
                "sample_id"
            ]
        )


        image = cv2.imread(
            str(
                IMAGE_DIR
                / f"{sample_id}.jpg"
            )
        )


        if image is None:

            raise RuntimeError(
                f"Could not read TEST image: "
                f"{sample_id}"
            )


        # ====================================================
        # STAGE 1
        # ORIGINAL 0-DEGREE OCR
        # ====================================================

        synchronize_gpu()

        start = (
            time.perf_counter()
        )


        (
            raw_0,
            confidence_0,
            regions_0,
        ) = extract_prediction(
            ocr,
            image,
        )


        synchronize_gpu()


        base_latency = (
            time.perf_counter()
            - start
        ) * 1000.0


        (
            parsed_0,
            correction_0,
            parser_info_0,
        ) = extract_indian_plate(
            raw_0
        )


        # Frozen M22A behavior:
        #
        # if parser succeeds -> parsed plate
        # otherwise retain raw OCR prediction.
        final_prediction = (
            parsed_0
            if parsed_0
            else clean_text(
                raw_0
            )
        )


        selected_rotation = 0

        final_confidence = (
            confidence_0
        )

        extra_latency = 0.0


        candidate_rows.append({
            "sample_id":
                sample_id,

            "rotation":
                0,

            "raw_prediction":
                clean_text(
                    raw_0
                ),

            "parsed_candidate":
                parsed_0,

            "correction_cost":
                correction_0,

            "removed_characters":
                removed_count(
                    parser_info_0
                ),

            "ocr_confidence":
                round(
                    confidence_0,
                    6,
                ),

            "text_regions":
                regions_0,

            "latency_ms":
                round(
                    base_latency,
                    3,
                ),
        })


        # ====================================================
        # STAGE 2
        # FROZEN ADAPTIVE ORIENTATION RESCUE
        #
        # Trigger ONLY if 0-degree parser did not produce
        # a plausible Indian registration.
        # ====================================================

        if not parsed_0:

            orientation_triggered += 1

            valid_candidates = []


            for angle in [
                90,
                180,
                270,
            ]:

                rotated = (
                    rotate_image(
                        image,
                        angle,
                    )
                )


                synchronize_gpu()

                rotation_start = (
                    time.perf_counter()
                )


                (
                    raw_prediction,
                    confidence,
                    regions,
                ) = extract_prediction(
                    ocr,
                    rotated,
                )


                synchronize_gpu()


                latency_ms = (
                    time.perf_counter()
                    - rotation_start
                ) * 1000.0


                extra_latency += (
                    latency_ms
                )


                (
                    parsed_candidate,
                    correction_cost,
                    parser_info,
                ) = extract_indian_plate(
                    raw_prediction
                )


                removed = (
                    removed_count(
                        parser_info
                    )
                )


                candidate_rows.append({
                    "sample_id":
                        sample_id,

                    "rotation":
                        angle,

                    "raw_prediction":
                        clean_text(
                            raw_prediction
                        ),

                    "parsed_candidate":
                        parsed_candidate,

                    "correction_cost":
                        correction_cost,

                    "removed_characters":
                        removed,

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


                if parsed_candidate:

                    valid_candidates.append({
                        "rotation":
                            angle,

                        "plate":
                            parsed_candidate,

                        "correction_cost":
                            correction_cost,

                        "removed":
                            removed,

                        "confidence":
                            confidence,
                    })


            # ================================================
            # FROZEN CANDIDATE SELECTION
            #
            # Ground truth is NOT involved.
            # ================================================

            if valid_candidates:

                valid_candidates.sort(
                    key=lambda candidate: (
                        candidate[
                            "correction_cost"
                        ],

                        candidate[
                            "removed"
                        ],

                        -candidate[
                            "confidence"
                        ],

                        candidate[
                            "rotation"
                        ],
                    )
                )


                best = (
                    valid_candidates[
                        0
                    ]
                )


                final_prediction = (
                    best[
                        "plate"
                    ]
                )


                selected_rotation = (
                    best[
                        "rotation"
                    ]
                )


                final_confidence = (
                    best[
                        "confidence"
                    ]
                )


                orientation_selected += 1


        # ====================================================
        # TOTAL INFERENCE LATENCY
        # ====================================================

        total_latency = (
            base_latency
            + extra_latency
        )


        latencies.append(
            total_latency
        )


        # ====================================================
        # ONLY NOW USE GT FOR SCORING
        # ====================================================

        ground_truth = clean_text(
            row[
                "ground_truth"
            ]
        )


        edit_distance = (
            levenshtein(
                ground_truth,
                final_prediction,
            )
        )


        exact = (
            final_prediction
            == ground_truth
        )


        if exact:

            exact_matches += 1


        if not final_prediction:

            blank_predictions += 1


        total_edit_distance += (
            edit_distance
        )


        total_gt_characters += (
            len(
                ground_truth
            )
        )


        result_rows.append({
            "sample_id":
                sample_id,

            "ground_truth":
                ground_truth,

            "raw_0deg":
                clean_text(
                    raw_0
                ),

            "parsed_0deg":
                parsed_0,

            "selected_rotation":
                selected_rotation,

            "final_prediction":
                final_prediction,

            "exact_match":
                exact,

            "edit_distance":
                edit_distance,

            "gt_length":
                len(
                    ground_truth
                ),

            "final_confidence":
                round(
                    final_confidence,
                    6,
                ),

            "base_latency_ms":
                round(
                    base_latency,
                    3,
                ),

            "extra_orientation_latency_ms":
                round(
                    extra_latency,
                    3,
                ),

            "total_latency_ms":
                round(
                    total_latency,
                    3,
                ),
        })


        result = (
            "MATCH"
            if exact
            else "MISS"
        )


        print(
            f"[{index:02d}/14] "
            f"{sample_id} | "
            f"GT={ground_truth} | "
            f"RAW="
            f"{clean_text(raw_0) or 'NONE'} | "
            f"ROT="
            f"{selected_rotation} | "
            f"FINAL="
            f"{final_prediction or 'NONE'} | "
            f"{result}"
        )


    # ========================================================
    # SAVE
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


    with open(
        CANDIDATE_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=(
                candidate_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            candidate_rows
        )


    # ========================================================
    # METRICS
    # ========================================================

    total = len(
        evaluation
    )


    exact_accuracy = (
        exact_matches
        / total
    )


    cer = (
        total_edit_distance
        / total_gt_characters
    )


    blank_rate = (
        blank_predictions
        / total
    )


    mean_latency = (
        sum(
            latencies
        )
        / total
    )


    sorted_latencies = sorted(
        latencies
    )


    if (
        len(
            sorted_latencies
        )
        % 2
        == 0
    ):

        middle = (
            len(
                sorted_latencies
            )
            // 2
        )

        median_latency = (
            sorted_latencies[
                middle - 1
            ]
            +
            sorted_latencies[
                middle
            ]
        ) / 2.0

    else:

        median_latency = (
            sorted_latencies[
                len(
                    sorted_latencies
                )
                // 2
            ]
        )


    summary = f"""M23B — HELD-OUT OCR TEST EVALUATION
============================================================

Evaluation status
-----------------
ONE-TIME TEST EVALUATION

Pipeline was frozen during M23A.

No TEST result was used for method selection.

Evaluation samples
------------------
Verified TEST plates:
{total}

Unverified/unreadable TEST samples:
excluded from character-level scoring

Frozen OCR pipeline
-------------------
PaddleOCR 3.7.0

PP-OCRv5_server_det
PP-OCRv5_server_rec

0-degree OCR
+
Indian registration parser

If unresolved:
adaptive 90 / 180 / 270 degree OCR

Candidate ranking:
1. lower correction cost
2. fewer removed characters
3. higher OCR confidence
4. lower rotation angle on exact tie

Results
-------
Exact matches:
{exact_matches}/{total}

Exact-match accuracy:
{exact_accuracy:.4f}
({exact_accuracy * 100:.2f}%)

Total GT characters:
{total_gt_characters}

Total edit distance:
{total_edit_distance}

Character Error Rate:
{cer:.4f}
({cer * 100:.2f}%)

Blank final predictions:
{blank_predictions}

Blank prediction rate:
{blank_rate:.4f}
({blank_rate * 100:.2f}%)

Orientation behavior
--------------------
Orientation search triggered:
{orientation_triggered}/{total}

Rotated candidate selected:
{orientation_selected}/{total}

Runtime
-------
Mean adaptive latency:
{mean_latency:.2f} ms

Median adaptive latency:
{median_latency:.2f} ms

Research rule
-------------
THIS TEST RESULT IS FINAL FOR THE FROZEN OCR METHOD.

The pipeline must not be modified based on these TEST
predictions and then re-reported on the same TEST set.

Any future method development requires a new development
set or a new untouched evaluation set.
"""


    SUMMARY_TXT.write_text(
        summary,
        encoding="utf-8",
    )


    # ========================================================
    # PRINT FINAL TEST RESULT
    # ========================================================

    print("\n" + "=" * 72)

    print(
        "M23B TEST EVALUATION COMPLETE"
    )

    print("=" * 72)


    print(
        f"\nExact matches : "
        f"{exact_matches}/{total}"
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
        f"\nOrientation search: "
        f"{orientation_triggered}/{total}"
    )


    print(
        f"Rotation selected : "
        f"{orientation_selected}/{total}"
    )


    print(
        f"\nMean latency   : "
        f"{mean_latency:.2f} ms"
    )


    print(
        f"Median latency : "
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