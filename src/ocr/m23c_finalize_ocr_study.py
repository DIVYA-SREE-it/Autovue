from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parents[2]

DEV_CSV = (
    ROOT
    / "outputs"
    / "M22_ocr_improvements"
    / "M22B_orientation_rescue_dev_results.csv"
)

TEST_CSV = (
    ROOT
    / "outputs"
    / "M23_ocr_test"
    / "M23B_test_results.csv"
)

ABLATION_CSV = (
    ROOT
    / "outputs"
    / "M23_ocr_ablation"
    / "M23A_ablation_table.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "M23_ocr_final"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_TXT = (
    OUTPUT_DIR
    / "M23C_final_ocr_study.txt"
)

COMPARISON_CSV = (
    OUTPUT_DIR
    / "M23C_dev_test_comparison.csv"
)


def read_csv(path):

    with open(
        path,
        newline="",
        encoding="utf-8",
    ) as file:

        return list(
            csv.DictReader(file)
        )


def dev_metrics(rows):

    total = len(rows)

    exact = sum(
        str(
            row["final_exact"]
        ).lower()
        == "true"
        for row in rows
    )

    total_edits = sum(
        int(
            row["final_edit_distance"]
        )
        for row in rows
    )

    total_chars = sum(
        len(
            row["ground_truth"]
        )
        for row in rows
    )

    mean_latency = (
        sum(
            float(
                row[
                    "adaptive_total_latency_ms"
                ]
            )
            for row in rows
        )
        / total
    )

    orientation_selected = sum(
        int(
            row[
                "selected_rotation"
            ]
        )
        != 0
        for row in rows
    )

    return {
        "total": total,
        "exact": exact,
        "accuracy": exact / total,
        "cer": total_edits / total_chars,
        "mean_latency": mean_latency,
        "orientation_selected":
            orientation_selected,
    }


def test_metrics(rows):

    total = len(rows)

    exact = sum(
        str(
            row["exact_match"]
        ).lower()
        == "true"
        for row in rows
    )

    total_edits = sum(
        int(
            row["edit_distance"]
        )
        for row in rows
    )

    total_chars = sum(
        int(
            row["gt_length"]
        )
        for row in rows
    )

    mean_latency = (
        sum(
            float(
                row[
                    "total_latency_ms"
                ]
            )
            for row in rows
        )
        / total
    )

    orientation_selected = sum(
        int(
            row[
                "selected_rotation"
            ]
        )
        != 0
        for row in rows
    )

    orientation_triggered = sum(
        not bool(
            row[
                "parsed_0deg"
            ].strip()
        )
        for row in rows
    )

    raw_exact = sum(
        row[
            "raw_0deg"
        ]
        == row[
            "ground_truth"
        ]
        for row in rows
    )

    return {
        "total":
            total,

        "exact":
            exact,

        "accuracy":
            exact / total,

        "cer":
            total_edits / total_chars,

        "mean_latency":
            mean_latency,

        "orientation_selected":
            orientation_selected,

        "orientation_triggered":
            orientation_triggered,

        "raw_exact":
            raw_exact,
    }


def main():

    print("=" * 72)
    print(
        "M23C — FINAL OCR STUDY SUMMARY"
    )
    print("=" * 72)


    dev_rows = read_csv(
        DEV_CSV
    )

    test_rows = read_csv(
        TEST_CSV
    )

    ablation_rows = read_csv(
        ABLATION_CSV
    )


    dev = dev_metrics(
        dev_rows
    )

    test = test_metrics(
        test_rows
    )


    comparison_rows = [
        {
            "split":
                "DEV",

            "verified_samples":
                dev["total"],

            "exact_matches":
                dev["exact"],

            "exact_accuracy_pct":
                round(
                    dev["accuracy"]
                    * 100,
                    2,
                ),

            "cer_pct":
                round(
                    dev["cer"]
                    * 100,
                    2,
                ),

            "mean_latency_ms":
                round(
                    dev["mean_latency"],
                    2,
                ),

            "rotated_candidate_selected":
                dev[
                    "orientation_selected"
                ],
        },

        {
            "split":
                "TEST",

            "verified_samples":
                test["total"],

            "exact_matches":
                test["exact"],

            "exact_accuracy_pct":
                round(
                    test["accuracy"]
                    * 100,
                    2,
                ),

            "cer_pct":
                round(
                    test["cer"]
                    * 100,
                    2,
                ),

            "mean_latency_ms":
                round(
                    test["mean_latency"],
                    2,
                ),

            "rotated_candidate_selected":
                test[
                    "orientation_selected"
                ],
        },
    ]


    with open(
        COMPARISON_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=(
                comparison_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            comparison_rows
        )


    # Locate useful ablation rows.
    ablation = {
        row["stage"]: row
        for row in ablation_rows
    }


    raw_easy = (
        ablation[
            "M21B"
        ]
    )

    raw_paddle = (
        ablation[
            "M21C"
        ]
    )

    parser = (
        ablation[
            "M22A"
        ]
    )

    orientation = (
        ablation[
            "M22B"
        ]
    )

    preprocessing = (
        ablation[
            "M22C"
        ]
    )

    spatial = (
        ablation[
            "M22D"
        ]
    )


    summary = f"""M23C — FINAL STATIC OCR STUDY
============================================================

Purpose
-------
Evaluate OCR recognition on manually transcribed Indian
number-plate crops.

This study evaluates OCR recognition only.

It does NOT represent complete ANPR performance because
the benchmark uses ground-truth plate crops rather than
detections produced by the full road-video pipeline.


DATASET
-------
OCR DEV:

Verified readable plates:
{dev["total"]}

OCR TEST:

Verified readable plates:
{test["total"]}

TEST was not used for OCR method selection.


OCR ENGINE COMPARISON — DEV
---------------------------
EasyOCR raw:

Exact accuracy:
{raw_easy["exact_accuracy_pct"]}%

CER:
{raw_easy["cer_pct"]}%


PaddleOCR raw:

Exact accuracy:
{raw_paddle["exact_accuracy_pct"]}%

CER:
{raw_paddle["cer_pct"]}%


PaddleOCR was selected as the primary OCR engine.


DEV ABLATION
------------
Raw PaddleOCR:

Exact:
{raw_paddle["exact_matches"]}/{raw_paddle["total"]}

Accuracy:
{raw_paddle["exact_accuracy_pct"]}%

CER:
{raw_paddle["cer_pct"]}%


+ Indian registration parser:

Exact:
{parser["exact_matches"]}/{parser["total"]}

Accuracy:
{parser["exact_accuracy_pct"]}%

CER:
{parser["cer_pct"]}%


+ adaptive orientation rescue:

Exact:
{orientation["exact_matches"]}/{orientation["total"]}

Accuracy:
{orientation["exact_accuracy_pct"]}%

CER:
{orientation["cer_pct"]}%

Mean latency:
{orientation["mean_latency_ms"]} ms


REJECTED DEV EXPERIMENTS
------------------------
Generic preprocessing:

Accuracy:
{preprocessing["exact_accuracy_pct"]}%

CER:
{preprocessing["cer_pct"]}%

Mean latency:
{preprocessing["mean_latency_ms"]} ms

Decision:
REJECTED because it produced no recognition improvement
while greatly increasing runtime.


Spatial-order rescue:

Accuracy:
{spatial["exact_accuracy_pct"]}%

CER:
{spatial["cer_pct"]}%

Decision:
REJECTED because it produced no DEV improvement.


FROZEN PIPELINE
---------------
PaddleOCR
+
Indian registration parser
+
adaptive orientation rescue

The method was frozen before TEST evaluation.


HELD-OUT TEST RESULT
--------------------
Exact matches:

{test["exact"]}/{test["total"]}

Exact accuracy:

{test["accuracy"] * 100:.2f}%

Character Error Rate:

{test["cer"] * 100:.2f}%

Mean adaptive latency:

{test["mean_latency"]:.2f} ms

Orientation search triggered:

{test["orientation_triggered"]}/{test["total"]}

Rotated candidate selected:

{test["orientation_selected"]}/{test["total"]}


TEST ORIENTATION CONTRIBUTION
-----------------------------
Correct directly at raw 0-degree OCR:

{test["raw_exact"]}/{test["total"]}

Correct after complete frozen pipeline:

{test["exact"]}/{test["total"]}

Additional exact TEST plates obtained by the frozen
pipeline relative to raw 0-degree OCR:

{test["exact"] - test["raw_exact"]}


DEV VS TEST
-----------
DEV exact accuracy:

{dev["accuracy"] * 100:.2f}%

TEST exact accuracy:

{test["accuracy"] * 100:.2f}%


DEV CER:

{dev["cer"] * 100:.2f}%

TEST CER:

{test["cer"] * 100:.2f}%


Interpretation:

TEST performance is higher than DEV performance.

This should not be interpreted as evidence that the model
improved after freezing.

The most likely explanation is different sample
difficulty together with the small benchmark size.

Both DEV and TEST results should therefore be reported.


RESEARCH LIMITATIONS
--------------------
1. OCR TEST contains only 14 verified readable crops.

2. Static OCR uses ground-truth plate crops.

3. These results do not include plate-detector recall,
   vehicle tracking errors or road-video degradation.

4. TEST results must not be used to retune this frozen
   method and then be re-reported on the same TEST set.

5. Full ANPR performance requires separate road-video
   end-to-end evaluation.


FINAL STATIC-OCR CONCLUSION
---------------------------
PaddleOCR performed better than EasyOCR on the DEV
benchmark.

Indian-registration-aware parsing substantially improved
exact plate recognition.

Adaptive orientation handling provided further gains.

Generic preprocessing and the tested spatial-order rescue
were not beneficial.

The frozen OCR pipeline achieved:

{test["exact"]}/{test["total"]}
({test["accuracy"] * 100:.2f}%)

exact plate recognition on the held-out static OCR TEST
benchmark, with:

{test["cer"] * 100:.2f}%

character error rate.
"""


    SUMMARY_TXT.write_text(
        summary,
        encoding="utf-8",
    )


    print("\n" + "=" * 72)
    print("M23C COMPLETE")
    print("=" * 72)

    print(
        f"\nDEV : "
        f"{dev['exact']}/{dev['total']} "
        f"exact | "
        f"{dev['accuracy'] * 100:.2f}% | "
        f"CER {dev['cer'] * 100:.2f}%"
    )

    print(
        f"TEST: "
        f"{test['exact']}/{test['total']} "
        f"exact | "
        f"{test['accuracy'] * 100:.2f}% | "
        f"CER {test['cer'] * 100:.2f}%"
    )

    print(
        "\nSummary:"
    )

    print(
        SUMMARY_TXT
    )

    print(
        "\nComparison:"
    )

    print(
        COMPARISON_CSV
    )


if __name__ == "__main__":
    main()