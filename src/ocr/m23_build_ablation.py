from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "M23_ocr_ablation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ABLATION_CSV = (
    OUTPUT_DIR
    / "M23A_ablation_table.csv"
)

SUMMARY_TXT = (
    OUTPUT_DIR
    / "M23A_ablation_summary.txt"
)

FROZEN_TXT = (
    OUTPUT_DIR
    / "M23A_frozen_pipeline.txt"
)


# ============================================================
# INPUT PATHS
# ============================================================

EASY_CSV = (
    ROOT
    / "outputs"
    / "M21_ocr_benchmark"
    / "M21B_easyocr_dev_results.csv"
)

PADDLE_CSV = (
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

M22B_CSV = (
    ROOT
    / "outputs"
    / "M22_ocr_improvements"
    / "M22B_orientation_rescue_dev_results.csv"
)

M22C_CSV = (
    ROOT
    / "outputs"
    / "M22_ocr_improvements"
    / "M22C_preprocessing_rescue_dev_results.csv"
)

M22D_CSV = (
    ROOT
    / "outputs"
    / "M22_ocr_improvements"
    / "M22D_spatial_order_dev_results.csv"
)


# ============================================================
# HELPERS
# ============================================================

def read_rows(path):

    with open(
        path,
        newline="",
        encoding="utf-8",
    ) as f:

        return list(
            csv.DictReader(f)
        )


def metrics_raw(rows):

    total = len(rows)

    exact = sum(
        str(row["exact_match"]).lower()
        == "true"
        for row in rows
    )

    total_edits = sum(
        int(row["edit_distance"])
        for row in rows
    )

    total_chars = sum(
        int(row["gt_length"])
        for row in rows
    )

    mean_latency = sum(
        float(row["latency_ms"])
        for row in rows
    ) / total

    return {
        "exact":
            exact,

        "total":
            total,

        "accuracy":
            exact / total,

        "cer":
            total_edits
            / total_chars,

        "latency":
            mean_latency,
    }


def metrics_improved(
    rows,
    final_field,
    edit_field,
    latency_field=None,
):

    total = len(rows)

    exact = sum(
        str(
            row[
                "final_exact"
            ]
        ).lower()
        == "true"
        for row in rows
    )

    edits = sum(
        int(
            row[
                edit_field
            ]
        )
        for row in rows
    )

    chars = sum(
        len(
            row[
                "ground_truth"
            ]
        )
        for row in rows
    )

    latency = None

    if latency_field:

        latency = (
            sum(
                float(
                    row[
                        latency_field
                    ]
                )
                for row in rows
            )
            / total
        )

    return {
        "exact":
            exact,

        "total":
            total,

        "accuracy":
            exact / total,

        "cer":
            edits / chars,

        "latency":
            latency,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("M23A — OCR DEV ABLATION TABLE")
    print("=" * 72)


    easy_rows = read_rows(
        EASY_CSV
    )

    paddle_rows = read_rows(
        PADDLE_CSV
    )

    m22a_rows = read_rows(
        M22A_CSV
    )

    m22b_rows = read_rows(
        M22B_CSV
    )

    m22c_rows = read_rows(
        M22C_CSV
    )

    m22d_rows = read_rows(
        M22D_CSV
    )


    easy = metrics_raw(
        easy_rows
    )

    paddle = metrics_raw(
        paddle_rows
    )


    # M22A final fields have different names.
    m22a_total = len(
        m22a_rows
    )

    m22a_exact = sum(
        str(
            row[
                "final_exact"
            ]
        ).lower()
        == "true"
        for row in m22a_rows
    )

    m22a_edits = sum(
        int(
            row[
                "final_edit_distance"
            ]
        )
        for row in m22a_rows
    )

    m22a_chars = sum(
        len(
            row[
                "ground_truth"
            ]
        )
        for row in m22a_rows
    )

    m22a = {
        "exact":
            m22a_exact,

        "total":
            m22a_total,

        "accuracy":
            m22a_exact
            / m22a_total,

        "cer":
            m22a_edits
            / m22a_chars,

        # Parser CPU overhead is tiny relative
        # to OCR and was not isolated.
        "latency":
            paddle[
                "latency"
            ],
    }


    m22b = metrics_improved(
        m22b_rows,
        "final_prediction",
        "final_edit_distance",
        "adaptive_total_latency_ms",
    )


    m22c = metrics_improved(
        m22c_rows,
        "final_prediction",
        "final_edit_distance",
        "adaptive_total_latency_ms",
    )


    m22d = metrics_improved(
        m22d_rows,
        "final_prediction",
        "final_edit_distance",
        None,
    )


    stages = [
        {
            "stage":
                "M21B",

            "method":
                "EasyOCR raw",

            "metrics":
                easy,

            "decision":
                "BASELINE",

            "reason":
                "Comparison OCR baseline",
        },

        {
            "stage":
                "M21C",

            "method":
                "PaddleOCR raw",

            "metrics":
                paddle,

            "decision":
                "KEEP_ENGINE",

            "reason":
                "Lower CER and only raw exact matches",
        },

        {
            "stage":
                "M22A",

            "method":
                "PaddleOCR + Indian plate parser",

            "metrics":
                m22a,

            "decision":
                "KEEP",

            "reason":
                "Recovered 4 additional exact plates",
        },

        {
            "stage":
                "M22B",

            "method":
                "Parser + adaptive orientation",

            "metrics":
                m22b,

            "decision":
                "KEEP_FINAL",

            "reason":
                "Best DEV exact accuracy and CER",
        },

        {
            "stage":
                "M22C",

            "method":
                "Parser + preprocessing rescue",

            "metrics":
                m22c,

            "decision":
                "REJECT",

            "reason":
                "No accuracy/CER gain; large latency cost",
        },

        {
            "stage":
                "M22D",

            "method":
                "Parser + spatial-order rescue",

            "metrics":
                m22d,

            "decision":
                "REJECT",

            "reason":
                "No DEV improvement",
        },
    ]


    output_rows = []


    for item in stages:

        metrics = item[
            "metrics"
        ]

        latency = (
            ""
            if metrics[
                "latency"
            ] is None
            else round(
                metrics[
                    "latency"
                ],
                2,
            )
        )

        output_rows.append({
            "stage":
                item[
                    "stage"
                ],

            "method":
                item[
                    "method"
                ],

            "exact_matches":
                metrics[
                    "exact"
                ],

            "total":
                metrics[
                    "total"
                ],

            "exact_accuracy_pct":
                round(
                    metrics[
                        "accuracy"
                    ]
                    * 100,
                    2,
                ),

            "cer_pct":
                round(
                    metrics[
                        "cer"
                    ]
                    * 100,
                    2,
                ),

            "mean_latency_ms":
                latency,

            "decision":
                item[
                    "decision"
                ],

            "reason":
                item[
                    "reason"
                ],
        })


    with open(
        ABLATION_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=(
                output_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )


    raw = paddle

    final = m22b


    accuracy_gain_pp = (
        final[
            "accuracy"
        ]
        -
        raw[
            "accuracy"
        ]
    ) * 100


    cer_reduction_pp = (
        raw[
            "cer"
        ]
        -
        final[
            "cer"
        ]
    ) * 100


    summary = f"""M23A — OCR DEV ABLATION SUMMARY
============================================================

Evaluation set
--------------
26 manually VERIFIED DEV plates.

OCR TEST remains untouched.

Selected OCR engine
-------------------
PaddleOCR 3.7.0

PP-OCRv5_server_det
PP-OCRv5_server_rec

DEV progression
---------------
Raw PaddleOCR:

Exact:
{raw["exact"]}/{raw["total"]}

Accuracy:
{raw["accuracy"] * 100:.2f}%

CER:
{raw["cer"] * 100:.2f}%

Mean latency:
{raw["latency"]:.2f} ms


Selected final DEV pipeline:

PaddleOCR
+ Indian plate-aware parser
+ adaptive orientation rescue

Exact:
{final["exact"]}/{final["total"]}

Accuracy:
{final["accuracy"] * 100:.2f}%

CER:
{final["cer"] * 100:.2f}%

Mean adaptive latency:
{final["latency"]:.2f} ms


Improvement over raw PaddleOCR
------------------------------
Exact-accuracy gain:

+{accuracy_gain_pp:.2f} percentage points

CER reduction:

-{cer_reduction_pp:.2f} percentage points


Accepted components
-------------------
KEEP:
Indian plate-aware parser

KEEP:
Adaptive orientation rescue


Rejected components
-------------------
REJECT:
Generic preprocessing rescue

Reason:
No exact-match or CER improvement and substantial
latency increase.

REJECT:
Spatial-order rescue

Reason:
No DEV improvement.


Research rule
-------------
All OCR method selection was performed on DEV.

The OCR TEST set has not been used for method selection.

The frozen method should now be evaluated on TEST once,
without further tuning from TEST results.
"""


    SUMMARY_TXT.write_text(
        summary,
        encoding="utf-8",
    )


    frozen = """M23A — FROZEN OCR PIPELINE
============================================================

The OCR pipeline is frozen BEFORE TEST evaluation.

FINAL METHOD
------------

1. Input:
   license-plate crop.

2. OCR:
   PaddleOCR 3.7.0
   PP-OCRv5_server_det
   PP-OCRv5_server_rec

3. First pass:
   OCR at original 0-degree orientation.

4. Normalize OCR text:
   uppercase;
   remove spaces, punctuation and non-alphanumeric text.

5. Indian registration parser:
   search for plausible registrations using:

   AA00A0000
   AA00AA0000

   Validate:
   - Indian state/UT prefix
   - two district digits
   - one/two-letter series
   - four registration digits

   Limited OCR letter/digit confusion repair is allowed
   only according to expected character type.

6. Early exit:
   if the 0-degree result produces a plausible plate,
   accept it.

7. Orientation rescue:
   if unresolved, run OCR at:

   90 degrees
   180 degrees
   270 degrees

8. Parse each rotated OCR result using the SAME parser.

9. Candidate selection:
   choose by:

   a. lower correction cost
   b. fewer removed/noise characters
   c. higher OCR confidence
   d. lower rotation angle on exact tie

10. No generic preprocessing rescue.

11. No spatial-order rescue.

12. Ground truth must never participate in inference or
    candidate selection.

13. TEST results must not be used to modify this pipeline.

STATUS
------
FROZEN FOR HELD-OUT OCR TEST EVALUATION.
"""


    FROZEN_TXT.write_text(
        frozen,
        encoding="utf-8",
    )


    print("\n" + "=" * 72)
    print("M23A COMPLETE")
    print("=" * 72)

    print("\nAblation table:")
    print(ABLATION_CSV)

    print("\nSummary:")
    print(SUMMARY_TXT)

    print("\nFrozen method:")
    print(FROZEN_TXT)


if __name__ == "__main__":
    main()