from pathlib import Path
import csv
import re


ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = (
    ROOT
    / "outputs"
    / "M21_ocr_benchmark"
    / "M21C_paddleocr_dev_results.csv"
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
    / "M22A_plate_parser_dev_results.csv"
)

SUMMARY_TXT = (
    OUTPUT_DIR
    / "M22A_plate_parser_dev_summary.txt"
)


# ============================================================
# INDIAN STATE / UT REGISTRATION PREFIXES
# ============================================================

STATE_CODES = {
    "AN",
    "AP",
    "AR",
    "AS",
    "BR",
    "CG",
    "CH",
    "DD",
    "DL",
    "DN",
    "GA",
    "GJ",
    "HP",
    "HR",
    "JH",
    "JK",
    "KA",
    "KL",
    "LA",
    "LD",
    "MH",
    "ML",
    "MN",
    "MP",
    "MZ",
    "NL",
    "OD",
    "PB",
    "PY",
    "RJ",
    "SK",
    "TN",
    "TR",
    "TS",
    "UK",
    "UP",
    "WB",
}


# OCR-confusion mappings.
DIGIT_MAP = {
    "O": "0",
    "Q": "0",
    "D": "0",

    "I": "1",
    "L": "1",

    "Z": "2",

    "S": "5",

    "G": "6",

    "B": "8",
}


LETTER_MAP = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
}


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

    for i, ca in enumerate(
        a,
        start=1,
    ):

        current = [i]

        for j, cb in enumerate(
            b,
            start=1,
        ):

            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1]
                    + (
                        ca != cb
                    ),
                )
            )

        previous = current

    return previous[-1]


def convert_char(
    character,
    expected_type,
):

    """
    expected_type:
    L = letter
    D = digit
    """

    character = (
        character.upper()
    )

    if expected_type == "L":

        if character.isalpha():
            return (
                character,
                0,
            )

        if character in LETTER_MAP:

            return (
                LETTER_MAP[
                    character
                ],
                1,
            )

        return (
            None,
            99,
        )


    if expected_type == "D":

        if character.isdigit():

            return (
                character,
                0,
            )

        if character in DIGIT_MAP:

            return (
                DIGIT_MAP[
                    character
                ],
                1,
            )

        return (
            None,
            99,
        )


def apply_pattern(
    text,
    pattern,
):

    if len(text) != len(pattern):

        return None


    output = []

    correction_cost = 0


    for char, expected in zip(
        text,
        pattern,
    ):

        converted, cost = (
            convert_char(
                char,
                expected,
            )
        )

        if converted is None:

            return None

        output.append(
            converted
        )

        correction_cost += cost


    return (
        "".join(
            output
        ),
        correction_cost,
    )


def plausible_candidate(
    plate,
):

    if len(plate) < 9:
        return False

    if plate[:2] not in STATE_CODES:
        return False

    # State letters + two district digits.
    if not plate[2:4].isdigit():
        return False

    # Last four characters should be numeric.
    if not plate[-4:].isdigit():
        return False

    # Middle series should be one or two letters.
    series = plate[
        4:-4
    ]

    if len(series) not in {
        1,
        2,
    }:
        return False

    if not series.isalpha():
        return False

    return True


def extract_indian_plate(
    raw_text,
):

    text = clean_text(
        raw_text
    )

    if not text:

        return (
            "",
            -1,
            "",
        )


    candidates = []


    # ========================================================
    # COMMON STANDARD FORMATS
    #
    # 9 chars:
    # AA00A0000
    #
    # 10 chars:
    # AA00AA0000
    # ========================================================

    patterns = {
        9:
            "LLDD L DDDD"
            .replace(
                " ",
                "",
            ),

        10:
            "LLDD LL DDDD"
            .replace(
                " ",
                "",
            ),
    }


    for target_len, pattern in (
        patterns.items()
    ):

        if len(text) < target_len:
            continue


        # Search every contiguous substring.
        for start in range(
            0,
            len(text)
            - target_len
            + 1,
        ):

            piece = text[
                start:
                start + target_len
            ]


            converted = (
                apply_pattern(
                    piece,
                    pattern,
                )
            )


            if converted is None:
                continue


            plate, correction_cost = (
                converted
            )


            if not plausible_candidate(
                plate
            ):
                continue


            # Prefer:
            # fewer OCR corrections,
            # less discarded surrounding text,
            # longer 10-character plate.
            removed = (
                len(text)
                - target_len
            )


            score = (
                correction_cost
                +
                removed * 0.25
                -
                target_len * 0.02
            )


            candidates.append({
                "plate":
                    plate,

                "correction_cost":
                    correction_cost,

                "removed_characters":
                    removed,

                "start":
                    start,

                "score":
                    score,
            })


    if not candidates:

        return (
            "",
            -1,
            "",
        )


    candidates.sort(
        key=lambda x: (
            x[
                "score"
            ],
            x[
                "correction_cost"
            ],
            x[
                "removed_characters"
            ],
        )
    )


    best = candidates[
        0
    ]


    return (
        best[
            "plate"
        ],

        best[
            "correction_cost"
        ],

        (
            f"substring_start="
            f"{best['start']};"
            f"removed="
            f"{best['removed_characters']}"
        ),
    )


def main():

    print("=" * 72)
    print(
        "M22A — INDIAN PLATE-AWARE "
        "POST-PROCESSING"
    )
    print("=" * 72)


    with open(
        INPUT_CSV,
        newline="",
        encoding="utf-8",
    ) as file:

        input_rows = list(
            csv.DictReader(file)
        )


    result_rows = []

    raw_exact = 0

    parsed_exact = 0

    raw_edit_total = 0

    parsed_edit_total = 0

    gt_character_total = 0

    recovered_exact = 0


    for row in input_rows:

        sample_id = row[
            "sample_id"
        ]

        gt = clean_text(
            row[
                "ground_truth"
            ]
        )

        raw = clean_text(
            row[
                "prediction"
            ]
        )


        (
            parsed,
            correction_cost,
            parser_info,
        ) = extract_indian_plate(
            raw
        )


        # If parser finds no plausible candidate,
        # keep the raw OCR output for CER comparison.
        final_prediction = (
            parsed
            if parsed
            else raw
        )


        raw_edit = (
            levenshtein(
                gt,
                raw,
            )
        )

        parsed_edit = (
            levenshtein(
                gt,
                final_prediction,
            )
        )


        raw_match = (
            raw == gt
        )

        parsed_match = (
            final_prediction
            == gt
        )


        if raw_match:
            raw_exact += 1

        if parsed_match:
            parsed_exact += 1


        if (
            parsed_match
            and not raw_match
        ):
            recovered_exact += 1


        raw_edit_total += (
            raw_edit
        )

        parsed_edit_total += (
            parsed_edit
        )

        gt_character_total += (
            len(gt)
        )


        if parsed_match:
            status = (
                "EXACT"
            )

        elif parsed_edit < raw_edit:

            status = (
                "IMPROVED"
            )

        elif parsed_edit > raw_edit:

            status = (
                "WORSE"
            )

        else:

            status = (
                "UNCHANGED"
            )


        result_rows.append({
            "sample_id":
                sample_id,

            "ground_truth":
                gt,

            "raw_prediction":
                raw,

            "parsed_candidate":
                parsed,

            "final_prediction":
                final_prediction,

            "correction_cost":
                correction_cost,

            "raw_edit_distance":
                raw_edit,

            "final_edit_distance":
                parsed_edit,

            "raw_exact":
                raw_match,

            "final_exact":
                parsed_match,

            "status":
                status,

            "parser_info":
                parser_info,
        })


        print(
            f"{sample_id} | "
            f"GT={gt} | "
            f"RAW={raw or 'NONE'} | "
            f"PARSED="
            f"{parsed or 'NONE'} | "
            f"{status}"
        )


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


    raw_accuracy = (
        raw_exact
        / len(
            input_rows
        )
    )

    parsed_accuracy = (
        parsed_exact
        / len(
            input_rows
        )
    )


    raw_cer = (
        raw_edit_total
        / gt_character_total
    )

    parsed_cer = (
        parsed_edit_total
        / gt_character_total
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

    unchanged = sum(
        row["status"]
        == "UNCHANGED"
        for row in result_rows
    )


    summary = f"""M22A — INDIAN PLATE-AWARE PARSER
============================================================

Evaluation
----------
OCR DEV only

Samples:
{len(input_rows)}

OCR engine:
PaddleOCR raw predictions from M21C

No OCR was rerun.

No image preprocessing was added.

No TEST data was used.

Baseline
--------
Exact matches:
{raw_exact}/{len(input_rows)}

Exact accuracy:
{raw_accuracy:.4f}
({raw_accuracy * 100:.2f}%)

CER:
{raw_cer:.4f}
({raw_cer * 100:.2f}%)

After plate-aware parsing
-------------------------
Exact matches:
{parsed_exact}/{len(input_rows)}

Exact accuracy:
{parsed_accuracy:.4f}
({parsed_accuracy * 100:.2f}%)

CER:
{parsed_cer:.4f}
({parsed_cer * 100:.2f}%)

Exact matches recovered:
{recovered_exact}

Per-sample effect
-----------------
Improved but not exact:
{improved}

Worse:
{worse}

Unchanged:
{unchanged}

Method
------
The parser searches the OCR output for plausible
standard Indian registration substrings.

Supported standard structures:

AA00A0000
AA00AA0000

Candidate validation includes:
- recognized Indian state/UT prefix
- two district digits
- one/two-letter series
- four-digit registration number

Limited OCR-confusion substitutions are permitted
according to the expected letter/digit position.

Important
---------
Rules do not contain any DEV ground-truth plate strings.

OCR TEST remains untouched.
"""


    SUMMARY_TXT.write_text(
        summary,
        encoding="utf-8",
    )


    print("\n" + "=" * 72)
    print("M22A COMPLETE")
    print("=" * 72)

    print(
        f"\nRaw exact    : "
        f"{raw_exact}/"
        f"{len(input_rows)}"
    )

    print(
        f"Parsed exact : "
        f"{parsed_exact}/"
        f"{len(input_rows)}"
    )

    print(
        f"\nRaw CER      : "
        f"{raw_cer:.4f}"
    )

    print(
        f"Parsed CER   : "
        f"{parsed_cer:.4f}"
    )

    print(
        f"\nRecovered exact matches: "
        f"{recovered_exact}"
    )

    print(
        "\nSummary:"
    )

    print(
        SUMMARY_TXT
    )


if __name__ == "__main__":
    main()