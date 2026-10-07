from pathlib import Path
from collections import defaultdict
import csv
import math
import re


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

M24A_FRAME_CSV = (
    ROOT
    / "outputs"
    / "M24_temporal_ocr"
    / "M24A_frame_predictions.csv"
)

M24B_FRAME_CSV = (
    ROOT
    / "outputs"
    / "M24_temporal_ocr"
    / "M24B_frame_predictions.csv"
)

M24B_PREPROC_CSV = (
    ROOT
    / "outputs"
    / "M24_temporal_ocr"
    / "M24B_preprocessing_candidates.csv"
)

M24B_CONSENSUS_CSV = (
    ROOT
    / "outputs"
    / "M24_temporal_ocr"
    / "M24B_track_consensus.csv"
)

GT_CSV = (
    ROOT
    / "outputs"
    / "M20_ocr_ground_truth"
    / "track_ground_truth.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "M24_temporal_ocr"
)

EVIDENCE_CSV = (
    OUTPUT_DIR
    / "M24C_fragment_evidence.csv"
)

RESULT_CSV = (
    OUTPUT_DIR
    / "M24C_track_corroboration.csv"
)

SUMMARY_TXT = (
    OUTPUT_DIR
    / "M24C_fragment_corroboration_summary.txt"
)


# ============================================================
# HELPERS
# ============================================================

def load_rows(path):

    with open(
        path,
        newline="",
        encoding="utf-8",
    ) as file:

        return list(
            csv.DictReader(file)
        )


def clean_text(text):

    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(text).upper(),
    )


def longest_common_substring(
    a,
    b,
):

    """
    Return the longest EXACT contiguous substring
    shared by strings a and b.

    This is intentionally stricter than:
    - fuzzy edit distance
    - subsequence matching
    - grammar reconstruction

    We want strong supporting evidence only.
    """

    if not a or not b:

        return ""


    previous = [
        0
    ] * (
        len(b)
        + 1
    )

    best_length = 0

    best_end = 0


    for i in range(
        1,
        len(a) + 1,
    ):

        current = [
            0
        ] * (
            len(b)
            + 1
        )


        for j in range(
            1,
            len(b) + 1,
        ):

            if (
                a[i - 1]
                == b[j - 1]
            ):

                current[j] = (
                    previous[
                        j - 1
                    ]
                    + 1
                )


                if (
                    current[j]
                    > best_length
                ):

                    best_length = (
                        current[j]
                    )

                    best_end = i


        previous = current


    return a[
        best_end - best_length:
        best_end
    ]


def mixed_alphanumeric(text):

    """
    A supporting fragment should contain
    both a letter and a digit.

    This reduces accidental matches to words
    or purely numeric noise.
    """

    has_letter = any(
        character.isalpha()
        for character in text
    )

    has_digit = any(
        character.isdigit()
        for character in text
    )

    return (
        has_letter
        and has_digit
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)

    print(
        "M24C — PARTIAL TEMPORAL "
        "CORROBORATION"
    )

    print("=" * 72)


    # ========================================================
    # LOAD DATA
    # ========================================================

    m24a_frames = load_rows(
        M24A_FRAME_CSV
    )

    m24b_frames = load_rows(
        M24B_FRAME_CSV
    )

    preprocessing_rows = (
        load_rows(
            M24B_PREPROC_CSV
        )
    )

    consensus_rows = load_rows(
        M24B_CONSENSUS_CSV
    )

    gt_rows = load_rows(
        GT_CSV
    )


    verified_gt = {
        row["track_id"]:
            clean_text(
                row[
                    "ground_truth"
                ]
            )

        for row in gt_rows

        if row[
            "status"
        ] == "VERIFIED"
    }


    consensus = {
        row["track_id"]:
            row

        for row in consensus_rows
    }


    # ========================================================
    # COLLECT RAW OCR OBSERVATIONS
    #
    # These observations will NEVER generate a new plate.
    # They can only corroborate an already existing full
    # parser-valid candidate.
    # ========================================================

    observations = defaultdict(
        list
    )


    # Original raw OCR.
    for row in m24a_frames:

        raw = clean_text(
            row[
                "raw_0"
            ]
        )

        if not raw:
            continue


        observations[
            row[
                "track_id"
            ]
        ].append({
            "frame":
                int(
                    row[
                        "frame"
                    ]
                ),

            "source":
                "raw_0deg",

            "text":
                raw,
        })


    # Preprocessing variants.
    for row in preprocessing_rows:

        raw = clean_text(
            row[
                "raw_prediction"
            ]
        )

        if not raw:
            continue


        observations[
            row[
                "track_id"
            ]
        ].append({
            "frame":
                int(
                    row[
                        "frame"
                    ]
                ),

            "source":
                (
                    "preprocess_"
                    + row[
                        "variant"
                    ]
                ),

            "text":
                raw,
        })


    # ========================================================
    # FIND FRAMES THAT PRODUCED THE COMPLETE VALID CANDIDATE
    # ========================================================

    full_candidate_frames = defaultdict(
        lambda: defaultdict(
            set
        )
    )


    for row in m24b_frames:

        plate = clean_text(
            row[
                "final_valid_plate"
            ]
        )

        if not plate:
            continue


        full_candidate_frames[
            row[
                "track_id"
            ]
        ][
            plate
        ].add(
            int(
                row[
                    "frame"
                ]
            )
        )


    # ========================================================
    # TEMPORAL CORROBORATION
    # ========================================================

    evidence_rows = []

    result_rows = []


    exact_candidates = 0

    corroborated_candidates = 0


    for track_id in sorted(
        verified_gt,
        key=int,
    ):

        candidate = clean_text(
            consensus.get(
                track_id,
                {}
            ).get(
                "final_plate",
                "",
            )
        )


        # ----------------------------------------------------
        # No full candidate = fragments cannot construct one.
        # ----------------------------------------------------

        if not candidate:

            result_rows.append({
                "track_id":
                    track_id,

                "final_candidate":
                    "",

                "full_candidate_frames":
                    0,

                "fragment_support_frames":
                    0,

                "best_fragment":
                    "",

                "best_fragment_length":
                    0,

                "status":
                    "REJECTED",

                "ground_truth":
                    verified_gt[
                        track_id
                    ],

                "candidate_exact":
                    False,
            })

            continue


        candidate_frames = set(
            full_candidate_frames[
                track_id
            ][
                candidate
            ]
        )


        # ----------------------------------------------------
        # General threshold:
        #
        # at least 60% of candidate length,
        # and never fewer than 6 characters.
        #
        # For a 10-char plate:
        # threshold = 6.
        #
        # This is a development heuristic and must be
        # validated later on unseen road data.
        # ----------------------------------------------------

        minimum_fragment = max(
            6,
            math.ceil(
                0.60
                * len(
                    candidate
                )
            ),
        )


        best_per_frame = {}


        for observation in (
            observations[
                track_id
            ]
        ):

            frame = observation[
                "frame"
            ]


            # Must be an independent frame.
            if frame in candidate_frames:

                continue


            fragment = (
                longest_common_substring(
                    candidate,
                    observation[
                        "text"
                    ],
                )
            )


            if (
                len(fragment)
                < minimum_fragment
            ):

                continue


            if not mixed_alphanumeric(
                fragment
            ):

                continue


            evidence = {
                "track_id":
                    track_id,

                "candidate":
                    candidate,

                "candidate_frame_count":
                    len(
                        candidate_frames
                    ),

                "support_frame":
                    frame,

                "source":
                    observation[
                        "source"
                    ],

                "raw_text":
                    observation[
                        "text"
                    ],

                "matching_fragment":
                    fragment,

                "fragment_length":
                    len(
                        fragment
                    ),

                "required_length":
                    minimum_fragment,
            }


            # Keep strongest evidence from each frame.
            old = best_per_frame.get(
                frame
            )


            if (
                old is None
                or len(fragment)
                > old[
                    "fragment_length"
                ]
            ):

                best_per_frame[
                    frame
                ] = evidence


        supporting_frames = set(
            best_per_frame.keys()
        )


        for evidence in (
            best_per_frame.values()
        ):

            evidence_rows.append(
                evidence
            )


        if best_per_frame:

            strongest = max(
                best_per_frame.values(),
                key=lambda item:
                    item[
                        "fragment_length"
                    ],
            )

            best_fragment = (
                strongest[
                    "matching_fragment"
                ]
            )

            best_length = (
                strongest[
                    "fragment_length"
                ]
            )

        else:

            best_fragment = ""

            best_length = 0


        # ----------------------------------------------------
        # CONFIDENCE STATUS
        #
        # >=2 independent FULL frames:
        # VERIFIED_FULL
        #
        # one full candidate + fragment on another frame:
        # CORROBORATED_FRAGMENT
        #
        # one full candidate only:
        # NEEDS_REVIEW
        #
        # Fragment evidence NEVER constructs the candidate.
        # ----------------------------------------------------

        if (
            len(
                candidate_frames
            )
            >= 2
        ):

            status = (
                "VERIFIED_FULL"
            )

        elif (
            len(
                supporting_frames
            )
            >= 1
        ):

            status = (
                "CORROBORATED_FRAGMENT"
            )

        else:

            status = (
                "NEEDS_REVIEW"
            )


        gt = verified_gt[
            track_id
        ]


        exact = (
            candidate
            == gt
        )


        if exact:

            exact_candidates += 1


        if (
            exact
            and status
            == "CORROBORATED_FRAGMENT"
        ):

            corroborated_candidates += 1


        result_rows.append({
            "track_id":
                track_id,

            "final_candidate":
                candidate,

            "full_candidate_frames":
                len(
                    candidate_frames
                ),

            "fragment_support_frames":
                len(
                    supporting_frames
                ),

            "best_fragment":
                best_fragment,

            "best_fragment_length":
                best_length,

            "status":
                status,

            "ground_truth":
                gt,

            "candidate_exact":
                exact,
        })


    # ========================================================
    # SAVE
    # ========================================================

    with open(
        EVIDENCE_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fieldnames = [
            "track_id",
            "candidate",
            "candidate_frame_count",
            "support_frame",
            "source",
            "raw_text",
            "matching_fragment",
            "fragment_length",
            "required_length",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            evidence_rows
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


    # ========================================================
    # SUMMARY
    # ========================================================

    corroborated = sum(
        row[
            "status"
        ]
        == "CORROBORATED_FRAGMENT"

        for row in result_rows
    )


    full_verified = sum(
        row[
            "status"
        ]
        == "VERIFIED_FULL"

        for row in result_rows
    )


    needs_review = sum(
        row[
            "status"
        ]
        == "NEEDS_REVIEW"

        for row in result_rows
    )


    rejected = sum(
        row[
            "status"
        ]
        == "REJECTED"

        for row in result_rows
    )


    summary = f"""M24C — PARTIAL TEMPORAL CORROBORATION
============================================================

Purpose
-------
Test whether partial OCR evidence from independent video
frames can corroborate an already complete parser-valid
registration.

IMPORTANT
---------
Fragments NEVER create or reconstruct a registration.

A full parser-valid plate must already exist before
fragment corroboration is considered.

This prevents missing characters from being guessed.

Road pilot
----------
Verified analysis tracks:
{len(verified_gt)}

Unique registration strings:
{len(set(verified_gt.values()))}

Complete candidate tracks:
{sum(bool(row["final_candidate"]) for row in result_rows)}

Exact complete candidates:
{exact_candidates}

Corroboration rule
------------------
For each existing complete plate candidate:

1. Ignore the frame(s) that produced the full plate.

2. Inspect OCR strings from other frames of the SAME
   vehicle track.

3. Calculate the longest exact contiguous substring
   shared with the complete candidate.

4. Require the fragment to contain both letters and
   digits.

5. Require fragment length >= 60% of the full plate,
   with an absolute minimum of 6 characters.

Status
------
VERIFIED_FULL:
At least two independent frames produced the complete
valid plate.

CORROBORATED_FRAGMENT:
A complete plate exists on one frame and a sufficiently
long exact fragment supports it on another frame.

NEEDS_REVIEW:
A complete valid plate exists but has no independent
temporal corroboration.

REJECTED:
No complete parser-valid plate exists.

Results
-------
VERIFIED_FULL:
{full_verified}

CORROBORATED_FRAGMENT:
{corroborated}

NEEDS_REVIEW:
{needs_review}

REJECTED:
{rejected}

Exact candidates receiving fragment corroboration:
{corroborated_candidates}

Research interpretation
-----------------------
This experiment evaluates temporal confidence, not
general OCR accuracy.

Partial fragments are used only as supporting evidence.

They are never used to hallucinate or fill missing
registration characters.

The 60% / minimum-six-character rule is a development
heuristic and must be validated on a NEW unseen road
video before deployment use.
"""


    SUMMARY_TXT.write_text(
        summary,
        encoding="utf-8",
    )


    # ========================================================
    # PRINT
    # ========================================================

    print("\n" + "=" * 72)

    print(
        "M24C COMPLETE"
    )

    print("=" * 72)


    for row in result_rows:

        print(
            f"Track "
            f"{row['track_id']} | "
            f"PLATE="
            f"{row['final_candidate'] or 'NONE'} | "
            f"full_frames="
            f"{row['full_candidate_frames']} | "
            f"fragment_frames="
            f"{row['fragment_support_frames']} | "
            f"best="
            f"{row['best_fragment'] or '-'} | "
            f"{row['status']} | "
            f"{'MATCH' if row['candidate_exact'] else 'MISS'}"
        )


    print(
        "\nEvidence:"
    )

    print(
        EVIDENCE_CSV
    )


    print(
        "\nTrack results:"
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