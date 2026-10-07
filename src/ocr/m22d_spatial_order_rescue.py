from pathlib import Path
from statistics import median
import csv
import re

import cv2
from paddleocr import PaddleOCR

from m22a_plate_aware_parser import (
    clean_text,
    extract_indian_plate,
    levenshtein,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

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
    / "M22D_spatial_order_dev_results.csv"
)

CANDIDATE_CSV = (
    OUTPUT_DIR
    / "M22D_spatial_order_candidates.csv"
)

SUMMARY_TXT = (
    OUTPUT_DIR
    / "M22D_spatial_order_summary.txt"
)


# ============================================================
# HELPERS
# ============================================================

def load_csv(path):

    with open(
        path,
        newline="",
        encoding="utf-8",
    ) as f:

        return {
            row["sample_id"]: row
            for row in csv.DictReader(f)
        }


def get_result_data(result):

    data = result.json

    if callable(data):
        data = data()

    if "res" in data:
        data = data["res"]

    return data


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


def extract_text_lines(
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

                box = boxes[index]

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

                x1 = float(index)
                y1 = 0.0
                x2 = float(index + 1)
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
        return []


    regions.sort(
        key=lambda item: (
            item["y"],
            item["x"],
        )
    )


    typical_height = median(
        [
            item["height"]
            for item in regions
        ]
    )


    row_tolerance = max(
        10.0,
        typical_height * 0.60,
    )


    rows = []


    for region in regions:

        assigned = False


        for row in rows:

            mean_y = sum(
                item["y"]
                for item in row
            ) / len(row)


            if (
                abs(
                    region["y"]
                    - mean_y
                )
                <= row_tolerance
            ):

                row.append(
                    region
                )

                assigned = True

                break


        if not assigned:

            rows.append(
                [region]
            )


    # Top → bottom.
    rows.sort(
        key=lambda row:
            sum(
                item["y"]
                for item in row
            ) / len(row)
    )


    line_data = []


    for row in rows:

        # Left → right.
        row.sort(
            key=lambda item:
                item["x"]
        )


        text = "".join(
            item["text"]
            for item in row
        )


        total_chars = sum(
            len(
                item["text"]
            )
            for item in row
        )


        confidence = (
            sum(
                item["confidence"]
                * len(
                    item["text"]
                )
                for item in row
            )
            / max(
                total_chars,
                1,
            )
        )


        line_data.append({
            "text":
                text,

            "confidence":
                confidence,
        })


    return line_data


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "M22D — TWO-LINE SPATIAL-ORDER RESCUE"
    )
    print("=" * 72)


    baseline_rows = load_csv(
        M22A_CSV
    )

    sample_ids = sorted(
        baseline_rows.keys()
    )


    if len(sample_ids) != 26:

        raise RuntimeError(
            f"Expected 26 DEV samples, "
            f"found {len(sample_ids)}"
        )


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


    result_rows = []

    candidate_rows = []


    baseline_exact = 0

    final_exact = 0

    baseline_edits = 0

    final_edits = 0

    gt_chars = 0

    rescues = 0

    attempted = 0


    for index, sample_id in enumerate(
        sample_ids,
        start=1,
    ):

        baseline = baseline_rows[
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


        final_prediction = (
            baseline_prediction
        )

        selected_order = (
            "baseline"
        )


        # ----------------------------------------------------
        # Preserve any already-valid M22A plate.
        #
        # Only unresolved samples are inspected.
        # ----------------------------------------------------

        if not existing_parsed:

            attempted += 1


            image = cv2.imread(
                str(
                    IMAGE_DIR
                    / f"{sample_id}.jpg"
                )
            )


            if image is None:

                raise RuntimeError(
                    f"Could not read "
                    f"{sample_id}"
                )


            lines = extract_text_lines(
                ocr,
                image,
            )


            top_to_bottom = clean_text(
                "".join(
                    line[
                        "text"
                    ]
                    for line in lines
                )
            )


            bottom_to_top = clean_text(
                "".join(
                    line[
                        "text"
                    ]
                    for line in reversed(
                        lines
                    )
                )
            )


            (
                top_plate,
                top_cost,
                top_info,
            ) = extract_indian_plate(
                top_to_bottom
            )


            (
                reverse_plate,
                reverse_cost,
                reverse_info,
            ) = extract_indian_plate(
                bottom_to_top
            )


            candidate_rows.append({
                "sample_id":
                    sample_id,

                "line_count":
                    len(lines),

                "ocr_lines":
                    " | ".join(
                        line[
                            "text"
                        ]
                        for line in lines
                    ),

                "top_to_bottom":
                    top_to_bottom,

                "top_parsed":
                    top_plate,

                "top_correction_cost":
                    top_cost,

                "bottom_to_top":
                    bottom_to_top,

                "reverse_parsed":
                    reverse_plate,

                "reverse_correction_cost":
                    reverse_cost,
            })


            # ------------------------------------------------
            # IMPORTANT:
            #
            # Rescue only when normal top-to-bottom ordering
            # does NOT produce a valid registration but the
            # reversed line order DOES.
            #
            # This isolates the spatial-order effect.
            # ------------------------------------------------

            if (
                len(lines) >= 2
                and not top_plate
                and reverse_plate
            ):

                final_prediction = (
                    reverse_plate
                )

                selected_order = (
                    "bottom_to_top"
                )


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

            rescues += 1


        final_edits += (
            final_edit
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

            "selected_order":
                selected_order,

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
        })


        print(
            f"[{index:02d}/26] "
            f"{sample_id} | "
            f"GT={gt} | "
            f"M22A="
            f"{baseline_prediction or 'NONE'} | "
            f"ORDER={selected_order} | "
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
    ) as f:

        writer = csv.DictWriter(
            f,
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
    ) as f:

        writer = csv.DictWriter(
            f,
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


    summary = f"""M22D — TWO-LINE SPATIAL-ORDER RESCUE
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

Spatial-order method
--------------------
For unresolved samples, PaddleOCR text regions are grouped
into visual rows.

Two reading orders are examined:

1. top-to-bottom
2. bottom-to-top

A reversed order is accepted only when:

- normal order does NOT yield a plausible Indian plate
- reversed order DOES yield a plausible Indian plate

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
{rescues}

Improved but not exact:
{improved}

Worse:
{worse}

Spatial-order inspection triggered:
{attempted}/{total}

Deployment note
---------------
This DEV script reruns PaddleOCR because M21C did not
persist individual OCR region text.

In the final pipeline, both row orders can be generated
from the SAME initial OCR result, so no additional model
inference is required.

OCR TEST remains untouched.
"""


    SUMMARY_TXT.write_text(
        summary,
        encoding="utf-8",
    )


    print("\n" + "=" * 72)
    print("M22D COMPLETE")
    print("=" * 72)


    print(
        f"\nM22A exact : "
        f"{baseline_exact}/{total}"
    )


    print(
        f"M22D exact : "
        f"{final_exact}/{total}"
    )


    print(
        f"\nM22A CER   : "
        f"{baseline_cer:.4f}"
    )


    print(
        f"M22D CER   : "
        f"{final_cer:.4f}"
    )


    print(
        f"\nNew exact rescues: "
        f"{rescues}"
    )


    print(
        f"Worse samples    : "
        f"{worse}"
    )


    print(
        "\nSummary:"
    )

    print(
        SUMMARY_TXT
    )


if __name__ == "__main__":
    main()