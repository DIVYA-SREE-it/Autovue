from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "M21_ocr_benchmark"
)

EASY_CSV = (
    OUTPUT_DIR
    / "M21B_easyocr_dev_results.csv"
)

PADDLE_CSV = (
    OUTPUT_DIR
    / "M21C_paddleocr_dev_results.csv"
)

COMPARISON_CSV = (
    OUTPUT_DIR
    / "M21D_engine_comparison.csv"
)

SUMMARY_TXT = (
    OUTPUT_DIR
    / "M21D_engine_comparison_summary.txt"
)


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


def main():

    print("=" * 72)
    print("M21D — OCR ENGINE FAILURE ANALYSIS")
    print("=" * 72)

    easy = load_csv(
        EASY_CSV
    )

    paddle = load_csv(
        PADDLE_CSV
    )

    common_ids = sorted(
        set(easy)
        & set(paddle)
    )

    if len(common_ids) != 26:

        raise RuntimeError(
            f"Expected 26 common samples, "
            f"found {len(common_ids)}"
        )


    rows = []

    easy_wins = 0
    paddle_wins = 0
    ties = 0

    paddle_contains_gt = 0
    easy_contains_gt = 0

    paddle_blank = 0
    easy_blank = 0

    paddle_exact = 0
    easy_exact = 0


    for sample_id in common_ids:

        e = easy[
            sample_id
        ]

        p = paddle[
            sample_id
        ]

        gt = e[
            "ground_truth"
        ]

        easy_pred = e[
            "prediction"
        ]

        paddle_pred = p[
            "prediction"
        ]

        easy_edit = int(
            e[
                "edit_distance"
            ]
        )

        paddle_edit = int(
            p[
                "edit_distance"
            ]
        )


        if easy_edit < paddle_edit:

            winner = "EasyOCR"
            easy_wins += 1

        elif paddle_edit < easy_edit:

            winner = "PaddleOCR"
            paddle_wins += 1

        else:

            winner = "Tie"
            ties += 1


        e_exact = (
            easy_pred == gt
        )

        p_exact = (
            paddle_pred == gt
        )


        if e_exact:
            easy_exact += 1

        if p_exact:
            paddle_exact += 1


        e_contains = (
            bool(easy_pred)
            and gt in easy_pred
        )

        p_contains = (
            bool(paddle_pred)
            and gt in paddle_pred
        )


        if e_contains:
            easy_contains_gt += 1

        if p_contains:
            paddle_contains_gt += 1


        if not easy_pred:
            easy_blank += 1

        if not paddle_pred:
            paddle_blank += 1


        # --------------------------------------------
        # Simple diagnostic categories.
        # These DO NOT alter predictions.
        # --------------------------------------------

        paddle_issue = ""

        if not paddle_pred:

            paddle_issue = (
                "NO_TEXT"
            )

        elif p_exact:

            paddle_issue = (
                "EXACT"
            )

        elif gt in paddle_pred:

            paddle_issue = (
                "GT_WITH_EXTRA_TEXT"
            )

        elif (
            sorted(paddle_pred)
            == sorted(gt)
        ):

            paddle_issue = (
                "CHAR_ORDER_ERROR"
            )

        elif (  
            len(paddle_pred)
            == len(gt)
        ):

            paddle_issue = (
                "SAME_LENGTH_CHAR_ERRORS"
            )

        elif paddle_edit <= 2:

            paddle_issue = (
                "NEAR_MATCH"
            )

        else:

            paddle_issue = (
                "MAJOR_ERROR"
            )


        easy_issue = ""

        if not easy_pred:

            easy_issue = (
                "NO_TEXT"
            )

        elif e_exact:

            easy_issue = (
                "EXACT"
            )

        elif gt in easy_pred:

            easy_issue = (
                "GT_WITH_EXTRA_TEXT"
            )

        elif (
            len(easy_pred)
            == len(gt)
        ):

            easy_issue = (
                "SAME_LENGTH_CHAR_ERRORS"
            )

        elif (
            sorted(easy_pred)
            == sorted(gt)
        ):

            easy_issue = (
                "CHAR_ORDER_ERROR"
            )

        elif easy_edit <= 2:

            easy_issue = (
                "NEAR_MATCH"
            )

        else:

            easy_issue = (
                "MAJOR_ERROR"
            )


        rows.append({
            "sample_id":
                sample_id,

            "ground_truth":
                gt,

            "easy_prediction":
                easy_pred,

            "easy_edit_distance":
                easy_edit,

            "easy_issue":
                easy_issue,

            "paddle_prediction":
                paddle_pred,

            "paddle_edit_distance":
                paddle_edit,

            "paddle_issue":
                paddle_issue,

            "winner":
                winner,
        })


    with open(
        COMPARISON_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


    # --------------------------------------------
    # Count Paddle error types.
    # --------------------------------------------

    issue_counts = {}

    for row in rows:

        issue = row[
            "paddle_issue"
        ]

        issue_counts[
            issue
        ] = (
            issue_counts.get(
                issue,
                0,
            )
            + 1
        )


    summary = f"""M21D — OCR ENGINE COMPARISON
============================================================

Evaluation
----------
Split:
DEV only

Samples:
{len(common_ids)}

Ground truth:
Manual VERIFIED plates only

Raw baseline
------------
EasyOCR exact:
{easy_exact}/{len(common_ids)}

PaddleOCR exact:
{paddle_exact}/{len(common_ids)}

Per-sample edit-distance comparison
-----------------------------------
PaddleOCR better:
{paddle_wins}

EasyOCR better:
{easy_wins}

Tie:
{ties}

Prediction behavior
-------------------
EasyOCR blanks:
{easy_blank}

PaddleOCR blanks:
{paddle_blank}

EasyOCR predictions containing complete GT plus extra text:
{easy_contains_gt}

PaddleOCR predictions containing complete GT plus extra text:
{paddle_contains_gt}

PaddleOCR diagnostic categories
-------------------------------
EXACT:
{issue_counts.get("EXACT", 0)}

GT_WITH_EXTRA_TEXT:
{issue_counts.get("GT_WITH_EXTRA_TEXT", 0)}

NEAR_MATCH:
{issue_counts.get("NEAR_MATCH", 0)}

SAME_LENGTH_CHAR_ERRORS:
{issue_counts.get("SAME_LENGTH_CHAR_ERRORS", 0)}

CHAR_ORDER_ERROR:
{issue_counts.get("CHAR_ORDER_ERROR", 0)}

NO_TEXT:
{issue_counts.get("NO_TEXT", 0)}

MAJOR_ERROR:
{issue_counts.get("MAJOR_ERROR", 0)}

Interpretation
--------------
PaddleOCR is currently the stronger raw OCR engine on DEV
because it has lower total character error and produces
the only exact matches.

However, its high blank rate shows that raw OCR alone is
not sufficient.

Several PaddleOCR outputs preserve most or all plate
characters but contain extra regional/emblem text or
incorrect spatial ordering. These are candidates for
plate-aware post-processing and orientation handling.

No TEST samples were used.
"""


    SUMMARY_TXT.write_text(
        summary,
        encoding="utf-8",
    )


    print("\n" + "=" * 72)
    print("M21D COMPLETE")
    print("=" * 72)

    print(
        "\nPaddleOCR wins :",
        paddle_wins,
    )

    print(
        "EasyOCR wins   :",
        easy_wins,
    )

    print(
        "Ties           :",
        ties,
    )

    print(
        "\nPaddle exact   :",
        paddle_exact,
    )

    print(
        "Paddle blanks  :",
        paddle_blank,
    )

    print(
        "Paddle full-GT + extra text:",
        paddle_contains_gt,
    )

    print(
        "\nSummary:"
    )

    print(
        SUMMARY_TXT
    )


if __name__ == "__main__":
    main()