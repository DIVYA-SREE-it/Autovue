from pathlib import Path
from collections import defaultdict, Counter
import csv
import math
import statistics

from m22a_plate_aware_parser import (
    clean_text,
)

from m24c_fragment_corroboration import (
    longest_common_substring,
    mixed_alphanumeric,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FRAME_CSV = (
    ROOT
    / "outputs"
    / "M25_final_road_eval"
    / "M25D_frame_predictions.csv"
)

PREPROCESS_CSV = (
    ROOT
    / "outputs"
    / "M25_final_road_eval"
    / "M25D_preprocessing_candidates.csv"
)

TRACK_MAP_CSV = (
    ROOT
    / "outputs"
    / "M25_final_road_eval"
    / "M25C_track_to_cluster.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "M25_final_road_eval"
)

OPTIONS_CSV = (
    OUTPUT_DIR
    / "M25E_candidate_options.csv"
)

EVIDENCE_CSV = (
    OUTPUT_DIR
    / "M25E_fragment_evidence.csv"
)

CONSENSUS_CSV = (
    OUTPUT_DIR
    / "M25E_cluster_consensus.csv"
)

SUMMARY_TXT = (
    OUTPUT_DIR
    / "M25E_temporal_consensus_summary.txt"
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


def safe_float(
    value,
    default=0.0,
):

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "M25E — FROZEN CLUSTER TEMPORAL CONSENSUS"
    )
    print("=" * 72)


    frame_rows = load_csv(
        FRAME_CSV
    )

    preprocessing_rows = load_csv(
        PREPROCESS_CSV
    )

    mapping_rows = load_csv(
        TRACK_MAP_CSV
    )


    # ========================================================
    # CLUSTER MEMBERSHIP
    # ========================================================

    track_to_cluster = {}

    cluster_members = defaultdict(
        set
    )


    for row in mapping_rows:

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
    # FULL VALID PLATE VOTES
    # ========================================================

    votes = defaultdict(
        lambda: defaultdict(
            lambda: {
                "frames": set(),
                "tracks": set(),
                "confidences": [],
                "correction_costs": [],
                "quality_scores": [],
            }
        )
    )


    candidate_frames = defaultdict(
        lambda: defaultdict(
            set
        )
    )


    for row in frame_rows:

        cluster_id = row[
            "cluster_id"
        ]

        track_id = row[
            "track_id"
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
            safe_float(
                row[
                    "final_confidence"
                ]
            )
        )

        vote[
            "correction_costs"
        ].append(
            safe_float(
                row[
                    "correction_cost"
                ]
            )
        )

        vote[
            "quality_scores"
        ].append(
            safe_float(
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
    # SELECT COMPLETE CANDIDATE
    #
    # SAME M24F ORDER:
    #
    # 1. more full-support frames
    # 2. higher OCR confidence
    # 3. lower correction cost
    # 4. higher crop quality
    #
    # NO GROUND TRUTH
    # ========================================================

    cluster_candidates = {}

    option_rows = []


    for cluster_id in sorted(
        cluster_members,
        key=int,
    ):

        options = []


        for (
            plate,
            data
        ) in votes[
            cluster_id
        ].items():

            option = {
                "cluster_id":
                    cluster_id,

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
            }

            options.append(
                option
            )


        if not options:
            continue


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


        for index, option in enumerate(
            options
        ):

            option_rows.append({
                **option,

                "selected":
                    index == 0,
            })


    # ========================================================
    # COLLECT RAW OCR OBSERVATIONS
    #
    # SAME M24F SOURCES:
    #
    # - raw 0-degree OCR
    # - preprocessing OCR variants
    #
    # Orientation alternatives are NOT added here because
    # M24F did not use them for fragment corroboration.
    # ========================================================

    observations = defaultdict(
        list
    )


    for row in frame_rows:

        track_id = row[
            "track_id"
        ]

        cluster_id = row[
            "cluster_id"
        ]

        raw = clean_text(
            row[
                "raw_0deg"
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


    for row in preprocessing_rows:

        track_id = row[
            "track_id"
        ]

        cluster_id = (
            track_to_cluster.get(
                track_id,
                row.get(
                    "cluster_id",
                    track_id,
                ),
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
    # FROZEN FRAGMENT CORROBORATION
    #
    # SAME RULE AS M24C / M24F:
    #
    # - complete candidate must already exist
    # - independent frame
    # - longest exact contiguous substring
    # - mixed letters + digits
    # - >= 60% candidate length
    # - absolute minimum length 6
    #
    # A FRAGMENT NEVER CREATES A PLATE.
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


        # ----------------------------------------------------
        # NO COMPLETE CANDIDATE
        # ----------------------------------------------------

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

                "mean_confidence":
                    "",

                "mean_correction":
                    "",

                "mean_quality":
                    "",

                "status":
                    "REJECTED",
            })

            continue


        candidate = candidate_info[
            "plate"
        ]


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


        # ----------------------------------------------------
        # At most one fragment-support observation per frame.
        # ----------------------------------------------------

        best_per_frame = {}


        for observation in (
            observations[
                cluster_id
            ]
        ):

            frame = observation[
                "frame"
            ]


            # Independent evidence only.
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


            if (
                len(
                    fragment
                )
                <
                minimum_fragment
            ):
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

                "candidate_frame_count":
                    len(
                        full_frames
                    ),

                "support_frame":
                    frame,

                "original_track_id":
                    observation[
                        "original_track_id"
                    ],

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


            existing = (
                best_per_frame.get(
                    frame
                )
            )


            # Same M24 behavior:
            # keep the strongest fragment from each frame.
            if (
                existing is None
                or
                len(
                    fragment
                )
                >
                existing[
                    "fragment_length"
                ]
            ):

                best_per_frame[
                    frame
                ] = evidence


        selected_evidence = sorted(
            best_per_frame.values(),
            key=lambda item: (
                item[
                    "support_frame"
                ],
                -item[
                    "fragment_length"
                ],
            ),
        )


        evidence_rows.extend(
            selected_evidence
        )


        fragment_frames = set(
            best_per_frame.keys()
        )


        fragment_tracks = {
            item[
                "original_track_id"
            ]

            for item in (
                best_per_frame.values()
            )
        }


        if best_per_frame:

            best_evidence = max(
                best_per_frame.values(),
                key=lambda item:
                    item[
                        "fragment_length"
                    ],
            )

            best_fragment = (
                best_evidence[
                    "matching_fragment"
                ]
            )

            best_fragment_length = (
                best_evidence[
                    "fragment_length"
                ]
            )

        else:

            best_fragment = ""

            best_fragment_length = 0


        # ====================================================
        # FROZEN STATUS LOGIC
        # ====================================================

        if len(
            full_frames
        ) >= 2:

            status = (
                "VERIFIED_FULL"
            )

        elif fragment_frames:

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
                len(
                    full_frames
                ),

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

            "mean_confidence":
                round(
                    candidate_info[
                        "mean_confidence"
                    ],
                    6,
                ),

            "mean_correction":
                round(
                    candidate_info[
                        "mean_correction"
                    ],
                    4,
                ),

            "mean_quality":
                round(
                    candidate_info[
                        "mean_quality"
                    ],
                    4,
                ),

            "status":
                status,
        })


    # ========================================================
    # WRITE OUTPUTS
    # ========================================================

    option_fields = [
        "cluster_id",
        "plate",
        "frame_support",
        "track_support",
        "mean_confidence",
        "mean_correction",
        "mean_quality",
        "selected",
    ]


    with open(
        OPTIONS_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=option_fields,
        )

        writer.writeheader()

        writer.writerows(
            option_rows
        )


    evidence_fields = [
        "cluster_id",
        "member_tracks",
        "candidate",
        "candidate_frame_count",
        "support_frame",
        "original_track_id",
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


    consensus_fields = [
        "cluster_id",
        "member_tracks",
        "final_candidate",
        "full_frame_support",
        "full_track_support",
        "fragment_support_frames",
        "fragment_support_tracks",
        "best_fragment",
        "best_fragment_length",
        "mean_confidence",
        "mean_correction",
        "mean_quality",
        "status",
    ]


    with open(
        CONSENSUS_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=consensus_fields,
        )

        writer.writeheader()

        writer.writerows(
            consensus_rows
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    status_counts = Counter(
        row[
            "status"
        ]
        for row in consensus_rows
    )


    complete_candidates = sum(
        bool(
            row[
                "final_candidate"
            ]
        )
        for row in consensus_rows
    )


    unique_candidates = len({
        row[
            "final_candidate"
        ]

        for row in consensus_rows

        if row[
            "final_candidate"
        ]
    })


    summary = f"""M25E — FROZEN CLUSTER TEMPORAL CONSENSUS
============================================================

Purpose
-------
Apply the already-developed M24F temporal confidence rules
to the frozen unseen-road M25 predictions.

Ground truth used:
NO

Input
-----
OCR crops:
{len(frame_rows)}

Raw tracker IDs:
{len(set(row["track_id"] for row in frame_rows))}

Conservative M25C clusters:
{len(cluster_members)}

Parser-valid crop predictions:
{sum(bool(clean_text(row["final_valid_plate"])) for row in frame_rows)}

Candidate selection
-------------------
For multiple complete candidates in one cluster:

1. More distinct complete-support frames
2. Higher mean OCR confidence
3. Lower mean grammar correction cost
4. Higher mean crop quality

Ground truth is never used.

Fragment corroboration
----------------------
A fragment can support an existing complete candidate only.

Fragments NEVER create or reconstruct a registration.

Requirements:

1. Different frame from complete candidate
2. Longest exact contiguous substring
3. Contains letters and digits
4. Length >= 60% of candidate
5. Absolute minimum length = 6

Final status
------------
VERIFIED_FULL:
At least two independent frames produce the complete
selected registration.

CORROBORATED_FRAGMENT:
One complete candidate exists and a qualifying fragment
from another frame supports it.

NEEDS_REVIEW:
A complete candidate exists but has no independent
temporal corroboration.

REJECTED:
No complete parser-valid candidate exists.

Results
-------
Clusters with complete candidate:
{complete_candidates}

Unique selected candidate strings:
{unique_candidates}

VERIFIED_FULL:
{status_counts["VERIFIED_FULL"]}

CORROBORATED_FRAGMENT:
{status_counts["CORROBORATED_FRAGMENT"]}

NEEDS_REVIEW:
{status_counts["NEEDS_REVIEW"]}

REJECTED:
{status_counts["REJECTED"]}

Fragment evidence frames:
{len(evidence_rows)}

Important
---------
These statuses describe temporal evidence strength.

They are NOT OCR accuracy labels.

The M25 video has not yet been manually transcribed for
final ground-truth accuracy evaluation.

No cluster is merged based on matching OCR text.
"""


    SUMMARY_TXT.write_text(
        summary,
        encoding="utf-8",
    )


    # ========================================================
    # CONSOLE
    # ========================================================

    print()

    for row in consensus_rows:

        print(
            f"cluster={row['cluster_id']:>3} "
            f"tracks={row['member_tracks']:<10} "
            f"plate={row['final_candidate'] or 'NONE':<12} "
            f"full={row['full_frame_support']} "
            f"fragment={row['fragment_support_frames']} "
            f"best={row['best_fragment'] or '-':<10} "
            f"{row['status']}"
        )


    print()
    print("=" * 72)
    print("M25E COMPLETE")
    print("=" * 72)

    print(
        "Complete-candidate clusters:",
        complete_candidates,
    )

    print(
        "VERIFIED_FULL:",
        status_counts[
            "VERIFIED_FULL"
        ],
    )

    print(
        "CORROBORATED_FRAGMENT:",
        status_counts[
            "CORROBORATED_FRAGMENT"
        ],
    )

    print(
        "NEEDS_REVIEW:",
        status_counts[
            "NEEDS_REVIEW"
        ],
    )

    print(
        "REJECTED:",
        status_counts[
            "REJECTED"
        ],
    )

    print()
    print(
        "Candidate options:",
        OPTIONS_CSV
    )

    print(
        "Fragment evidence:",
        EVIDENCE_CSV
    )

    print(
        "Consensus:",
        CONSENSUS_CSV
    )

    print(
        "Summary:",
        SUMMARY_TXT
    )


if __name__ == "__main__":
    main()
