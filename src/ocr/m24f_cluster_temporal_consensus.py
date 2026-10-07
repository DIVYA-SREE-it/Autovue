from pathlib import Path
from collections import defaultdict
import csv
import math
import re
import statistics


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

REASSOCIATED_FRAME_CSV = (
    ROOT
    / "outputs"
    / "M24_temporal_ocr"
    / "M24E_reassociated_frames.csv"
)

M24A_FRAME_CSV = (
    ROOT
    / "outputs"
    / "M24_temporal_ocr"
    / "M24A_frame_predictions.csv"
)

M24B_PREPROC_CSV = (
    ROOT
    / "outputs"
    / "M24_temporal_ocr"
    / "M24B_preprocessing_candidates.csv"
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

CONSENSUS_CSV = (
    OUTPUT_DIR
    / "M24F_cluster_consensus.csv"
)

EVIDENCE_CSV = (
    OUTPUT_DIR
    / "M24F_cluster_fragment_evidence.csv"
)

EVAL_CSV = (
    OUTPUT_DIR
    / "M24F_cluster_evaluation.csv"
)

SUMMARY_TXT = (
    OUTPUT_DIR
    / "M24F_cluster_temporal_summary.txt"
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

        return list(
            csv.DictReader(file)
        )


def clean_text(text):

    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(text).upper(),
    )


def longest_common_substring(a, b):

    if not a or not b:
        return ""

    previous = [
        0
    ] * (
        len(b) + 1
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
            len(b) + 1
        )

        for j in range(
            1,
            len(b) + 1,
        ):

            if a[i - 1] == b[j - 1]:

                current[j] = (
                    previous[j - 1]
                    + 1
                )

                if current[j] > best_length:

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

    has_letter = any(
        c.isalpha()
        for c in text
    )

    has_digit = any(
        c.isdigit()
        for c in text
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
        "M24F — RE-ASSOCIATED CLUSTER TEMPORAL CONSENSUS"
    )
    print("=" * 72)


    # ========================================================
    # LOAD RE-ASSOCIATED FRAME DATA
    # ========================================================

    reassociated = load_csv(
        REASSOCIATED_FRAME_CSV
    )

    raw_frames = load_csv(
        M24A_FRAME_CSV
    )

    preprocessing = load_csv(
        M24B_PREPROC_CSV
    )

    gt_rows = load_csv(
        GT_CSV
    )


    # ========================================================
    # TRACK -> CLUSTER MAPPING
    # ========================================================

    track_to_cluster = {}

    cluster_members = defaultdict(
        set
    )


    for row in reassociated:

        track_id = row[
            "original_track_id"
        ]

        cluster_id = row[
            "cluster_id"
        ]

        track_to_cluster[
            track_id
        ] = cluster_id

        cluster_members[
            cluster_id
        ].add(
            track_id
        )


    # ========================================================
    # VERIFIED GT
    #
    # Used only AFTER re-association for evaluation.
    # ========================================================

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


    # ========================================================
    # FULL VALID PLATE VOTES AT CLUSTER LEVEL
    # ========================================================

    votes = defaultdict(
        lambda: defaultdict(
            lambda: {
                "frames":
                    set(),

                "tracks":
                    set(),

                "confidences":
                    [],

                "correction_costs":
                    [],

                "quality_scores":
                    [],
            }
        )
    )


    candidate_frames = defaultdict(
        lambda: defaultdict(
            set
        )
    )


    for row in reassociated:

        cluster_id = row[
            "cluster_id"
        ]

        track_id = row[
            "original_track_id"
        ]

        plate = clean_text(
            row[
                "final_valid_plate"
            ]
        )

        if not plate:
            continue


        frame = int(
            row[
                "frame"
            ]
        )


        vote = votes[
            cluster_id
        ][
            plate
        ]


        vote[
            "frames"
        ].add(
            frame
        )

        vote[
            "tracks"
        ].add(
            track_id
        )

        vote[
            "confidences"
        ].append(
            float(
                row[
                    "ocr_confidence"
                ]
            )
        )

        vote[
            "correction_costs"
        ].append(
            float(
                row[
                    "correction_cost"
                ]
            )
        )

        vote[
            "quality_scores"
        ].append(
            float(
                row[
                    "quality_score"
                ]
            )
        )


        candidate_frames[
            cluster_id
        ][
            plate
        ].add(
            frame
        )


    # ========================================================
    # CHOOSE FULL CANDIDATE WITHOUT GT
    # ========================================================

    cluster_candidates = {}


    for cluster_id in sorted(
        cluster_members,
        key=int,
    ):

        options = []


        for plate, data in (
            votes[
                cluster_id
            ].items()
        ):

            options.append({
                "plate":
                    plate,

                "frame_support":
                    len(
                        data[
                            "frames"
                        ]
                    ),

                "track_support":
                    len(
                        data[
                            "tracks"
                        ]
                    ),

                "mean_confidence":
                    statistics.mean(
                        data[
                            "confidences"
                        ]
                    ),

                "mean_correction":
                    statistics.mean(
                        data[
                            "correction_costs"
                        ]
                    ),

                "mean_quality":
                    statistics.mean(
                        data[
                            "quality_scores"
                        ]
                    ),
            })


        if options:

            options.sort(
                key=lambda item: (
                    -item[
                        "frame_support"
                    ],

                    -item[
                        "mean_confidence"
                    ],

                    item[
                        "mean_correction"
                    ],

                    -item[
                        "mean_quality"
                    ],
                )
            )


            cluster_candidates[
                cluster_id
            ] = options[0]


    # ========================================================
    # COLLECT RAW OCR OBSERVATIONS AT CLUSTER LEVEL
    # ========================================================

    observations = defaultdict(
        list
    )


    for row in raw_frames:

        track_id = row[
            "track_id"
        ]

        cluster_id = (
            track_to_cluster.get(
                track_id,
                track_id,
            )
        )

        raw = clean_text(
            row[
                "raw_0"
            ]
        )

        if not raw:
            continue


        observations[
            cluster_id
        ].append({
            "original_track_id":
                track_id,

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


    for row in preprocessing:

        track_id = row[
            "track_id"
        ]

        cluster_id = (
            track_to_cluster.get(
                track_id,
                track_id,
            )
        )

        raw = clean_text(
            row[
                "raw_prediction"
            ]
        )

        if not raw:
            continue


        observations[
            cluster_id
        ].append({
            "original_track_id":
                track_id,

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
    # FRAGMENT CORROBORATION
    #
    # EXACTLY SAME RULE AS M24C:
    #
    # >= 60% of candidate length
    # absolute minimum 6
    # mixed letters + digits
    # independent frame
    # ========================================================

    evidence_rows = []

    consensus_rows = []


    for cluster_id in sorted(
        cluster_members,
        key=int,
    ):

        members = sorted(
            cluster_members[
                cluster_id
            ],
            key=int,
        )


        candidate_info = (
            cluster_candidates.get(
                cluster_id
            )
        )


        if candidate_info is None:

            consensus_rows.append({
                "cluster_id":
                    cluster_id,

                "member_tracks":
                    "|".join(
                        members
                    ),

                "final_candidate":
                    "",

                "full_frame_support":
                    0,

                "full_track_support":
                    0,

                "fragment_support_frames":
                    0,

                "fragment_support_tracks":
                    0,

                "best_fragment":
                    "",

                "best_fragment_length":
                    0,

                "status":
                    "REJECTED",
            })

            continue


        candidate = (
            candidate_info[
                "plate"
            ]
        )


        full_frames = set(
            candidate_frames[
                cluster_id
            ][
                candidate
            ]
        )


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
                cluster_id
            ]
        ):

            frame = observation[
                "frame"
            ]


            # Must come from a frame that did NOT
            # produce the complete candidate.
            if frame in full_frames:

                continue


            fragment = (
                longest_common_substring(
                    candidate,
                    observation[
                        "text"
                    ],
                )
            )


            if len(
                fragment
            ) < minimum_fragment:

                continue


            if not mixed_alphanumeric(
                fragment
            ):

                continue


            evidence = {
                "cluster_id":
                    cluster_id,

                "member_tracks":
                    "|".join(
                        members
                    ),

                "candidate":
                    candidate,

                "support_track":
                    observation[
                        "original_track_id"
                    ],

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


        for evidence in (
            best_per_frame.values()
        ):

            evidence_rows.append(
                evidence
            )


        fragment_frames = set(
            best_per_frame.keys()
        )


        fragment_tracks = {
            evidence[
                "support_track"
            ]
            for evidence
            in best_per_frame.values()
        }


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

            best_fragment_length = (
                strongest[
                    "fragment_length"
                ]
            )

        else:

            best_fragment = ""

            best_fragment_length = 0


        # ====================================================
        # SAME CONFIDENCE LOGIC AS M24C
        # ====================================================

        if (
            candidate_info[
                "frame_support"
            ]
            >= 2
        ):

            status = (
                "VERIFIED_FULL"
            )

        elif len(
            fragment_frames
        ) >= 1:

            status = (
                "CORROBORATED_FRAGMENT"
            )

        else:

            status = (
                "NEEDS_REVIEW"
            )


        consensus_rows.append({
            "cluster_id":
                cluster_id,

            "member_tracks":
                "|".join(
                    members
                ),

            "final_candidate":
                candidate,

            "full_frame_support":
                candidate_info[
                    "frame_support"
                ],

            "full_track_support":
                candidate_info[
                    "track_support"
                ],

            "fragment_support_frames":
                len(
                    fragment_frames
                ),

            "fragment_support_tracks":
                len(
                    fragment_tracks
                ),

            "best_fragment":
                best_fragment,

            "best_fragment_length":
                best_fragment_length,

            "status":
                status,
        })


    # ========================================================
    # SAVE CONSENSUS + EVIDENCE
    # ========================================================

    with open(
        CONSENSUS_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=(
                consensus_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            consensus_rows
        )


    evidence_fields = [
        "cluster_id",
        "member_tracks",
        "candidate",
        "support_track",
        "support_frame",
        "source",
        "raw_text",
        "matching_fragment",
        "fragment_length",
        "required_length",
    ]


    with open(
        EVIDENCE_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=evidence_fields,
        )

        writer.writeheader()

        writer.writerows(
            evidence_rows
        )


    # ========================================================
    # EVALUATION AFTER CLUSTERING
    #
    # GT DOES NOT PARTICIPATE IN CLUSTERING OR PREDICTION.
    # ========================================================

    evaluation_rows = []

    exact_clusters = 0

    scoreable_clusters = 0


    for row in consensus_rows:

        cluster_id = row[
            "cluster_id"
        ]

        members = row[
            "member_tracks"
        ].split("|")


        member_gt = sorted(
            {
                verified_gt[
                    member
                ]

                for member in members

                if member
                in verified_gt
            }
        )


        if not member_gt:

            cluster_gt = ""

            gt_status = (
                "NO_VERIFIED_GT"
            )

            exact = ""

        elif len(
            member_gt
        ) == 1:

            cluster_gt = (
                member_gt[0]
            )

            gt_status = (
                "CONSISTENT"
            )

            exact = (
                clean_text(
                    row[
                        "final_candidate"
                    ]
                )
                == cluster_gt
            )

            scoreable_clusters += 1

            if exact:
                exact_clusters += 1

        else:

            cluster_gt = (
                "|".join(
                    member_gt
                )
            )

            gt_status = (
                "GT_CONFLICT"
            )

            exact = ""


        evaluation_rows.append({
            "cluster_id":
                cluster_id,

            "member_tracks":
                row[
                    "member_tracks"
                ],

            "cluster_ground_truth":
                cluster_gt,

            "gt_status":
                gt_status,

            "prediction":
                row[
                    "final_candidate"
                ],

            "temporal_status":
                row[
                    "status"
                ],

            "full_frame_support":
                row[
                    "full_frame_support"
                ],

            "fragment_support_frames":
                row[
                    "fragment_support_frames"
                ],

            "exact":
                exact,
        })


    with open(
        EVAL_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=(
                evaluation_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            evaluation_rows
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    merged_clusters = sum(
        "|" in row[
            "member_tracks"
        ]
        for row in consensus_rows
    )


    verified_full = sum(
        row[
            "status"
        ]
        == "VERIFIED_FULL"
        for row in consensus_rows
    )


    corroborated = sum(
        row[
            "status"
        ]
        == "CORROBORATED_FRAGMENT"
        for row in consensus_rows
    )


    needs_review = sum(
        row[
            "status"
        ]
        == "NEEDS_REVIEW"
        for row in consensus_rows
    )


    rejected = sum(
        row[
            "status"
        ]
        == "REJECTED"
        for row in consensus_rows
    )


    summary = f"""M24F — RE-ASSOCIATED CLUSTER TEMPORAL CONSENSUS
============================================================

Purpose
-------
Recompute temporal OCR evidence after conservative
tracker-ID re-association.

Re-association
--------------
Track IDs were merged only when M24E found a
pixel-identical crop in the same video frame.

OCR predictions and ground truth were NOT used to create
clusters.

Merged clusters in this pilot:
{merged_clusters}

Temporal candidate selection
----------------------------
Complete parser-valid candidates are selected using:

1. more distinct-frame support
2. higher OCR confidence
3. lower grammar correction cost
4. higher crop quality

Fragment corroboration
----------------------
The exact SAME rule from M24C is retained:

- fragment cannot create a plate
- it can only support an existing full candidate
- independent frame required
- longest exact contiguous substring
- letters and digits required
- >= 60% of plate length
- minimum 6 characters

Results
-------
VERIFIED_FULL:
{verified_full}

CORROBORATED_FRAGMENT:
{corroborated}

NEEDS_REVIEW:
{needs_review}

REJECTED:
{rejected}

Evaluation
----------
Scoreable re-associated clusters:
{scoreable_clusters}

Clusters with exact complete candidate:
{exact_clusters}

Do NOT convert this count into general road accuracy.

This remains a small development pilot.

Research interpretation
-----------------------
M24E/M24F test whether identity fragmentation prevents
OCR evidence from being accumulated at the physical
vehicle level.

Any improvement in temporal evidence after re-association
comes from combining observations across tracker IDs,
not from changing the OCR model or using ground truth.

A future unseen road-video evaluation is still required.
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
        "M24F COMPLETE"
    )

    print("=" * 72)


    for row in consensus_rows:

        print(
            f"Cluster "
            f"{row['cluster_id']} "
            f"[{row['member_tracks']}] | "
            f"PLATE="
            f"{row['final_candidate'] or 'NONE'} | "
            f"full="
            f"{row['full_frame_support']} | "
            f"fragments="
            f"{row['fragment_support_frames']} | "
            f"best="
            f"{row['best_fragment'] or '-'} | "
            f"{row['status']}"
        )


    print(
        "\nConsensus:"
    )

    print(
        CONSENSUS_CSV
    )


    print(
        "\nFragment evidence:"
    )

    print(
        EVIDENCE_CSV
    )


    print(
        "\nEvaluation:"
    )

    print(
        EVAL_CSV
    )


    print(
        "\nSummary:"
    )

    print(
        SUMMARY_TXT
    )


if __name__ == "__main__":
    main()