from pathlib import Path
import csv
import re
import time

import cv2
from paddleocr import PaddleOCR

from m21_paddleocr_dev import (
    extract_prediction,
    synchronize_gpu,
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

M21C_CSV = (
    ROOT
    / "outputs"
    / "M21_ocr_benchmark"
    / "M21C_paddleocr_dev_results.csv"
)

M22A_CSV = (
    ROOT
    / "outputs"
    / "M22_ocr_improvements"
    / "M22A_plate_parser_dev_results.csv"
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
    / "M22_ocr_improvements"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_CSV = (
    OUTPUT_DIR
    / "M22B_orientation_rescue_dev_results.csv"
)

CANDIDATE_CSV = (
    OUTPUT_DIR
    / "M22B_orientation_candidates.csv"
)

SUMMARY_TXT = (
    OUTPUT_DIR
    / "M22B_orientation_rescue_dev_summary.txt"
)


# ============================================================
# HELPERS
# ============================================================

def load_csv(path):

    with open(
        path,
        newline="",
        encoding="utf-8",
    ) as file:

        return {
            row["sample_id"]: row
            for row in csv.DictReader(file)
        }


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
        "M22B — ADAPTIVE ORIENTATION "
        "RESCUE"
    )
    print("=" * 72)


    raw_results = load_csv(
        M21C_CSV
    )

    parser_results = load_csv(
        M22A_CSV
    )


    sample_ids = sorted(
        parser_results.keys()
    )


    if len(sample_ids) != 26:

        raise RuntimeError(
            f"Expected 26 DEV samples, "
            f"found {len(sample_ids)}"
        )


    # ========================================================
    # LOAD OCR ONCE
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

    warmup_id = sample_ids[0]

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
    # EVALUATION
    # ========================================================

    result_rows = []

    candidate_rows = []

    baseline_exact = 0

    final_exact = 0

    baseline_edits = 0

    final_edits = 0

    gt_chars = 0

    orientation_rescues = 0

    orientation_attempted = 0

    extra_latency_total = 0.0

    adaptive_latency_total = 0.0


    for index, sample_id in enumerate(
        sample_ids,
        start=1,
    ):

        baseline = parser_results[
            sample_id
        ]

        raw_row = raw_results[
            sample_id
        ]


        gt = clean_text(
            baseline[
                "ground_truth"
            ]
        )


        baseline_prediction = (
            clean_text(
                baseline[
                    "final_prediction"
                ]
            )
        )


        existing_parsed = (
            clean_text(
                baseline[
                    "parsed_candidate"
                ]
            )
        )


        baseline_latency = float(
            raw_row[
                "latency_ms"
            ]
        )


        baseline_edit = (
            levenshtein(
                gt,
                baseline_prediction,
            )
        )


        baseline_match = (
            baseline_prediction
            == gt
        )


        if baseline_match:
            baseline_exact += 1


        baseline_edits += (
            baseline_edit
        )

        gt_chars += len(
            gt
        )


        # ----------------------------------------------------
        # DEFAULT:
        # keep M22A result.
        # ----------------------------------------------------

        final_prediction = (
            baseline_prediction
        )

        selected_rotation = 0

        rescue_candidate = ""

        extra_latency = 0.0

        valid_candidates = []


        # ----------------------------------------------------
        # ADAPTIVE RULE:
        #
        # If 0° already gave a plausible parsed plate,
        # trust it and DO NOT run extra orientations.
        #
        # Only unresolved samples get 90/180/270 search.
        # ----------------------------------------------------

        if not existing_parsed:

            orientation_attempted += 1


            image = cv2.imread(
                str(
                    IMAGE_DIR
                    / f"{sample_id}.jpg"
                )
            )


            if image is None:

                raise RuntimeError(
                    f"Could not load image "
                    f"{sample_id}"
                )


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

                start = (
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
                    - start
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
                        raw_prediction,

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


            # ------------------------------------------------
            # CHOOSE WITHOUT USING GROUND TRUTH
            #
            # Priority:
            # 1. fewer letter/digit corrections
            # 2. less discarded OCR noise
            # 3. higher OCR confidence
            # 4. smaller rotation on exact tie
            # ------------------------------------------------

            if valid_candidates:

                valid_candidates.sort(
                    key=lambda x: (
                        x[
                            "correction_cost"
                        ],

                        x[
                            "removed"
                        ],

                        -x[
                            "confidence"
                        ],

                        x[
                            "rotation"
                        ],
                    )
                )


                best = (
                    valid_candidates[0]
                )


                final_prediction = (
                    best[
                        "plate"
                    ]
                )

                rescue_candidate = (
                    best[
                        "plate"
                    ]
                )

                selected_rotation = (
                    best[
                        "rotation"
                    ]
                )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        final_edit = (
            levenshtein(
                gt,
                final_prediction,
            )
        )


        final_match = (
            final_prediction
            == gt
        )


        if final_match:
            final_exact += 1


        if (
            final_match
            and not baseline_match
        ):

            orientation_rescues += 1


        final_edits += (
            final_edit
        )


        extra_latency_total += (
            extra_latency
        )


        adaptive_latency = (
            baseline_latency
            + extra_latency
        )


        adaptive_latency_total += (
            adaptive_latency
        )


        if final_match:

            status = "EXACT"

        elif (
            final_edit
            < baseline_edit
        ):

            status = "IMPROVED"

        elif (
            final_edit
            > baseline_edit
        ):

            status = "WORSE"

        else:

            status = "UNCHANGED"


        result_rows.append({
            "sample_id":
                sample_id,

            "ground_truth":
                gt,

            "m22a_prediction":
                baseline_prediction,

            "orientation_candidate":
                rescue_candidate,

            "selected_rotation":
                selected_rotation,

            "final_prediction":
                final_prediction,

            "m22a_edit_distance":
                baseline_edit,

            "final_edit_distance":
                final_edit,

            "m22a_exact":
                baseline_match,

            "final_exact":
                final_match,

            "status":
                status,

            "baseline_latency_ms":
                round(
                    baseline_latency,
                    3,
                ),

            "extra_orientation_latency_ms":
                round(
                    extra_latency,
                    3,
                ),

            "adaptive_total_latency_ms":
                round(
                    adaptive_latency,
                    3,
                ),
        })


        print(
            f"[{index:02d}/26] "
            f"{sample_id} | "
            f"GT={gt} | "
            f"M22A="
            f"{baseline_prediction or 'NONE'} | "
            f"ROT="
            f"{selected_rotation} | "
            f"FINAL="
            f"{final_prediction or 'NONE'} | "
            f"{status}"
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


    if candidate_rows:

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
    # SUMMARY
    # ========================================================

    total = len(
        sample_ids
    )


    baseline_accuracy = (
        baseline_exact
        / total
    )


    final_accuracy = (
        final_exact
        / total
    )


    baseline_cer = (
        baseline_edits
        / gt_chars
    )


    final_cer = (
        final_edits
        / gt_chars
    )


    improved = sum(
        row["status"]
        == "IMPROVED"
        for row in result_rows
    )


    worse = sum(
        row["status"]
        == "WORSE"
        for row in result_rows
    )


    mean_extra_latency = (
        extra_latency_total
        / total
    )


    mean_adaptive_latency = (
        adaptive_latency_total
        / total
    )


    summary = f"""M22B — ADAPTIVE ORIENTATION RESCUE
============================================================

Evaluation
----------
OCR DEV only

Samples:
{total}

TEST used:
NO

Baseline
--------
M22A PaddleOCR + Indian plate parser

Exact matches:
{baseline_exact}/{total}

Exact accuracy:
{baseline_accuracy:.4f}
({baseline_accuracy * 100:.2f}%)

CER:
{baseline_cer:.4f}
({baseline_cer * 100:.2f}%)

Adaptive orientation strategy
-----------------------------
Run normal 0-degree OCR first.

If a plausible Indian registration is found:
STOP.

If no plausible registration is found:
try 90, 180 and 270 degree rotations.

Valid rotated candidates are selected using:
1. lower grammar correction cost
2. fewer discarded characters
3. higher OCR confidence
4. lower rotation angle on exact tie

Ground truth is NOT used for candidate selection.

Results
-------
Exact matches:
{final_exact}/{total}

Exact accuracy:
{final_accuracy:.4f}
({final_accuracy * 100:.2f}%)

CER:
{final_cer:.4f}
({final_cer * 100:.2f}%)

New exact matches recovered:
{orientation_rescues}

Improved but not exact:
{improved}

Worse:
{worse}

Orientation search triggered:
{orientation_attempted}/{total}

Runtime
-------
Mean additional orientation-search latency
averaged across ALL DEV samples:

{mean_extra_latency:.2f} ms

Mean adaptive total latency per crop:

{mean_adaptive_latency:.2f} ms

Important
---------
Existing valid 0-degree parsed plates are preserved.

Only unresolved samples receive additional rotation OCR.

OCR TEST remains untouched.
"""


    SUMMARY_TXT.write_text(
        summary,
        encoding="utf-8",
    )


    print("\n" + "=" * 72)
    print("M22B COMPLETE")
    print("=" * 72)


    print(
        f"\nM22A exact       : "
        f"{baseline_exact}/{total}"
    )

    print(
        f"Orientation exact: "
        f"{final_exact}/{total}"
    )


    print(
        f"\nM22A CER         : "
        f"{baseline_cer:.4f}"
    )

    print(
        f"Orientation CER  : "
        f"{final_cer:.4f}"
    )


    print(
        f"\nNew exact rescues: "
        f"{orientation_rescues}"
    )

    print(
        f"Search triggered : "
        f"{orientation_attempted}/{total}"
    )


    print(
        f"\nMean adaptive latency: "
        f"{mean_adaptive_latency:.2f} ms"
    )


    print(
        "\nSummary:"
    )

    print(
        SUMMARY_TXT
    )


if __name__ == "__main__":
    main()