from pathlib import Path
import csv
import re
import time

import cv2
import numpy as np

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
    / "M22C_preprocessing_rescue_dev_results.csv"
)

CANDIDATE_CSV = (
    OUTPUT_DIR
    / "M22C_preprocessing_candidates.csv"
)

SUMMARY_TXT = (
    OUTPUT_DIR
    / "M22C_preprocessing_rescue_dev_summary.txt"
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


def removed_count(parser_info):

    match = re.search(
        r"removed=(\d+)",
        parser_info or "",
    )

    if not match:
        return 999

    return int(
        match.group(1)
    )


def preprocessing_variants(image):

    variants = {}

    # --------------------------------------------------------
    # UPSCALE
    # --------------------------------------------------------

    upscaled = cv2.resize(
        image,
        None,
        fx=3.0,
        fy=3.0,
        interpolation=cv2.INTER_CUBIC,
    )

    variants[
        "upscaled"
    ] = upscaled


    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        upscaled,
        cv2.COLOR_BGR2GRAY,
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    clahe_gray = clahe.apply(
        gray
    )

    variants[
        "clahe"
    ] = cv2.cvtColor(
        clahe_gray,
        cv2.COLOR_GRAY2BGR,
    )


    # --------------------------------------------------------
    # SHARPEN
    # --------------------------------------------------------

    sharpen_kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ]
    )

    sharpened = cv2.filter2D(
        upscaled,
        -1,
        sharpen_kernel,
    )

    variants[
        "sharpened"
    ] = sharpened


    # --------------------------------------------------------
    # OTSU THRESHOLD
    # --------------------------------------------------------

    _, threshold = cv2.threshold(
        gray,
        0,
        255,
        (
            cv2.THRESH_BINARY
            + cv2.THRESH_OTSU
        ),
    )

    variants[
        "threshold"
    ] = cv2.cvtColor(
        threshold,
        cv2.COLOR_GRAY2BGR,
    )

    return variants


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "M22C — ADAPTIVE PREPROCESSING RESCUE"
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
    # LOAD OCR
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

    warmup_image = cv2.imread(
        str(
            IMAGE_DIR
            / f"{sample_ids[0]}.jpg"
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
    # METRICS
    # ========================================================

    result_rows = []

    candidate_rows = []

    baseline_exact = 0

    final_exact = 0

    baseline_edits = 0

    final_edits = 0

    gt_chars = 0

    preprocessing_rescues = 0

    preprocessing_attempted = 0

    extra_latency_total = 0.0

    adaptive_latency_total = 0.0


    # ========================================================
    # PROCESS DEV
    # ========================================================

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


        baseline_prediction = clean_text(
            baseline[
                "final_prediction"
            ]
        )


        existing_parsed = clean_text(
            baseline[
                "parsed_candidate"
            ]
        )


        baseline_latency = float(
            raw_row[
                "latency_ms"
            ]
        )


        baseline_edit = levenshtein(
            gt,
            baseline_prediction,
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
        # preserve M22A output.
        # ----------------------------------------------------

        final_prediction = (
            baseline_prediction
        )

        selected_variant = (
            "original"
        )

        rescue_candidate = ""

        extra_latency = 0.0

        valid_candidates = []


        # ----------------------------------------------------
        # ONLY unresolved M22A samples
        # ----------------------------------------------------

        if not existing_parsed:

            preprocessing_attempted += 1


            image = cv2.imread(
                str(
                    IMAGE_DIR
                    / f"{sample_id}.jpg"
                )
            )


            if image is None:

                raise RuntimeError(
                    f"Could not load "
                    f"{sample_id}"
                )


            variants = (
                preprocessing_variants(
                    image
                )
            )


            for (
                variant_name,
                variant_image
            ) in variants.items():

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
                    variant_image,
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


                removed = removed_count(
                    parser_info
                )


                candidate_rows.append({
                    "sample_id":
                        sample_id,

                    "variant":
                        variant_name,

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
                        "variant":
                            variant_name,

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
            # Select candidate WITHOUT GT.
            # ------------------------------------------------

            variant_priority = {
                "upscaled": 0,
                "clahe": 1,
                "sharpened": 2,
                "threshold": 3,
            }


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

                        variant_priority[
                            x[
                                "variant"
                            ]
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

                selected_variant = (
                    best[
                        "variant"
                    ]
                )


        # ====================================================
        # SCORE
        # ====================================================

        final_edit = levenshtein(
            gt,
            final_prediction,
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

            preprocessing_rescues += 1


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

        elif final_edit < baseline_edit:

            status = "IMPROVED"

        elif final_edit > baseline_edit:

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

            "preprocessing_candidate":
                rescue_candidate,

            "selected_variant":
                selected_variant,

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

            "extra_preprocessing_latency_ms":
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
            f"VAR="
            f"{selected_variant} | "
            f"FINAL="
            f"{final_prediction or 'NONE'} | "
            f"{status}"
        )


    # ========================================================
    # SAVE RESULTS
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


    summary = f"""M22C — ADAPTIVE PREPROCESSING RESCUE
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

Preprocessing strategy
----------------------
Normal raw OCR is preserved when it already yields a
plausible Indian registration.

Only unresolved samples are tested with:

1. 3x bicubic upscale
2. CLAHE
3. sharpening
4. Otsu thresholding

Each OCR output is passed through the same Indian
registration parser.

Candidate selection uses:

1. lower grammar correction cost
2. fewer discarded OCR characters
3. higher OCR confidence
4. fixed variant priority on exact ties

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
{preprocessing_rescues}

Improved but not exact:
{improved}

Worse:
{worse}

Preprocessing search triggered:
{preprocessing_attempted}/{total}

Runtime
-------
Mean additional preprocessing latency
averaged across ALL DEV samples:

{mean_extra_latency:.2f} ms

Mean adaptive total latency per crop:

{mean_adaptive_latency:.2f} ms

Important
---------
M22C is an independent ablation from M22B.

Rotations are NOT used in this experiment.

OCR TEST remains untouched.
"""


    SUMMARY_TXT.write_text(
        summary,
        encoding="utf-8",
    )


    print("\n" + "=" * 72)
    print("M22C COMPLETE")
    print("=" * 72)


    print(
        f"\nM22A exact       : "
        f"{baseline_exact}/{total}"
    )

    print(
        f"Preprocess exact : "
        f"{final_exact}/{total}"
    )


    print(
        f"\nM22A CER         : "
        f"{baseline_cer:.4f}"
    )

    print(
        f"Preprocess CER   : "
        f"{final_cer:.4f}"
    )


    print(
        f"\nNew exact rescues: "
        f"{preprocessing_rescues}"
    )

    print(
        f"Search triggered : "
        f"{preprocessing_attempted}/{total}"
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