from pathlib import Path
from collections import defaultdict
import csv
import hashlib


ROOT = Path(__file__).resolve().parents[2]

TRACK_CSV = (
    ROOT
    / "outputs"
    / "M09_ocr_video_tracking"
    / "track_summary.csv"
)

CROP_DIR = (
    ROOT
    / "outputs"
    / "M09_ocr_video_tracking"
    / "best_plate_crops"
)

M24B_FRAME_CSV = (
    ROOT
    / "outputs"
    / "M24_temporal_ocr"
    / "M24B_frame_predictions.csv"
)

M24C_RESULT_CSV = (
    ROOT
    / "outputs"
    / "M24_temporal_ocr"
    / "M24C_track_corroboration.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "M24_temporal_ocr"
)

DUPLICATE_CSV = (
    OUTPUT_DIR
    / "M24E_duplicate_track_pairs.csv"
)

CLUSTER_CSV = (
    OUTPUT_DIR
    / "M24E_track_clusters.csv"
)

FRAME_CSV = (
    OUTPUT_DIR
    / "M24E_reassociated_frames.csv"
)

SUMMARY_TXT = (
    OUTPUT_DIR
    / "M24E_track_reassociation_summary.txt"
)


def load_csv(path):

    with open(
        path,
        newline="",
        encoding="utf-8",
    ) as f:

        return list(
            csv.DictReader(f)
        )


def md5_file(path):

    digest = hashlib.md5()

    with open(
        path,
        "rb",
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


class UnionFind:

    def __init__(self):

        self.parent = {}


    def add(self, item):

        if item not in self.parent:

            self.parent[item] = item


    def find(self, item):

        if self.parent[item] != item:

            self.parent[item] = self.find(
                self.parent[item]
            )

        return self.parent[item]


    def union(self, a, b):

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        # deterministic cluster ID:
        # keep numerically smaller track root.
        if int(root_a) <= int(root_b):

            self.parent[root_b] = root_a

        else:

            self.parent[root_a] = root_b


def main():

    print("=" * 72)
    print(
        "M24E — CONSERVATIVE TRACK RE-ASSOCIATION"
    )
    print("=" * 72)


    track_rows = load_csv(
        TRACK_CSV
    )


    # ========================================================
    # HASH EVERY SAVED CROP
    # ========================================================

    hash_groups = defaultdict(
        list
    )


    uf = UnionFind()


    for row in track_rows:

        track_id = row[
            "track_id"
        ]

        uf.add(
            track_id
        )


        path = (
            CROP_DIR
            / row[
                "filename"
            ]
        )


        if not path.exists():

            continue


        digest = md5_file(
            path
        )


        hash_groups[
            digest
        ].append({
            "track_id":
                track_id,

            "frame":
                int(
                    row[
                        "frame"
                    ]
                ),

            "vehicle":
                row[
                    "vehicle"
                ],

            "filename":
                row[
                    "filename"
                ],

            "md5":
                digest,
        })


    # ========================================================
    # DUPLICATE TRACK PAIRS
    #
    # Conservative rule:
    #
    # exact file-byte hash match
    # +
    # same frame
    # +
    # different track ID
    #
    # No OCR or GT is used.
    # ========================================================

    duplicate_rows = []


    for digest, entries in (
        hash_groups.items()
    ):

        if len(entries) < 2:
            continue


        for i in range(
            len(entries)
        ):

            for j in range(
                i + 1,
                len(entries),
            ):

                a = entries[i]
                b = entries[j]


                if (
                    a[
                        "track_id"
                    ]
                    ==
                    b[
                        "track_id"
                    ]
                ):

                    continue


                if (
                    a[
                        "frame"
                    ]
                    !=
                    b[
                        "frame"
                    ]
                ):

                    continue


                duplicate_rows.append({
                    "track_a":
                        a[
                            "track_id"
                        ],

                    "track_b":
                        b[
                            "track_id"
                        ],

                    "frame":
                        a[
                            "frame"
                        ],

                    "vehicle_a":
                        a[
                            "vehicle"
                        ],

                    "vehicle_b":
                        b[
                            "vehicle"
                        ],

                    "filename_a":
                        a[
                            "filename"
                        ],

                    "filename_b":
                        b[
                            "filename"
                        ],

                    "md5":
                        digest,

                    "evidence":
                        "EXACT_DUPLICATE_SAME_FRAME",
                })


                uf.union(
                    a[
                        "track_id"
                    ],
                    b[
                        "track_id"
                    ],
                )


    # ========================================================
    # BUILD CONNECTED COMPONENTS
    # ========================================================

    clusters = defaultdict(
        list
    )


    for track_id in uf.parent:

        root = uf.find(
            track_id
        )

        clusters[
            root
        ].append(
            track_id
        )


    cluster_rows = []


    for root, members in sorted(
        clusters.items(),
        key=lambda item:
            int(
                item[0]
            ),
    ):

        members = sorted(
            members,
            key=int,
        )


        # Only output actual merges.
        if len(members) < 2:
            continue


        cluster_rows.append({
            "cluster_id":
                root,

            "member_tracks":
                "|".join(
                    members
                ),

            "member_count":
                len(
                    members
                ),

            "merge_basis":
                (
                    "pixel-identical crop "
                    "in same video frame"
                ),
        })


    # ========================================================
    # SAVE DUPLICATE PAIRS
    # ========================================================

    with open(
        DUPLICATE_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fields = [
            "track_a",
            "track_b",
            "frame",
            "vehicle_a",
            "vehicle_b",
            "filename_a",
            "filename_b",
            "md5",
            "evidence",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            duplicate_rows
        )


    with open(
        CLUSTER_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fields = [
            "cluster_id",
            "member_tracks",
            "member_count",
            "merge_basis",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            cluster_rows
        )


    # ========================================================
    # APPLY CLUSTER ID TO M24B FRAME EVIDENCE
    # ========================================================

    frame_rows = load_csv(
        M24B_FRAME_CSV
    )


    reassociated_rows = []


    for row in frame_rows:

        track_id = row[
            "track_id"
        ]


        if track_id in uf.parent:

            cluster_id = uf.find(
                track_id
            )

        else:

            cluster_id = track_id


        reassociated_rows.append({
            "cluster_id":
                cluster_id,

            "original_track_id":
                track_id,

            "rank":
                row[
                    "rank"
                ],

            "frame":
                row[
                    "frame"
                ],

            "filename":
                row[
                    "filename"
                ],

            "quality_score":
                row[
                    "quality_score"
                ],

            "m24a_plate":
                row[
                    "m24a_plate"
                ],

            "selected_variant":
                row[
                    "selected_variant"
                ],

            "final_valid_plate":
                row[
                    "final_valid_plate"
                ],

            "ocr_confidence":
                row[
                    "ocr_confidence"
                ],

            "correction_cost":
                row[
                    "correction_cost"
                ],
        })


    with open(
        FRAME_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=(
                reassociated_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            reassociated_rows
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    merged_tracks = sum(
        int(
            row[
                "member_count"
            ]
        )
        for row in cluster_rows
    )


    summary = f"""M24E — CONSERVATIVE TRACK RE-ASSOCIATION
============================================================

Purpose
-------
Detect tracker identity fragmentation using duplicate
saved observations.

Merge rule
----------
Two track IDs are linked only when they contain:

1. a crop from the SAME video frame
2. byte-identical crop content
3. different track IDs

No OCR prediction is used.

No registration ground truth is used.

No known plate identity is used.

This makes the re-association rule independent of the
OCR evaluation labels.

Results
-------
Duplicate cross-track observations found:
{len(duplicate_rows)}

Merged track clusters:
{len(cluster_rows)}

Tracks participating in merged clusters:
{merged_tracks}

Interpretation
--------------
A pixel-identical crop appearing under two different
track IDs in the same frame is direct evidence that the
tracker assigned multiple identities to the same saved
observation.

This can split OCR evidence across IDs and weaken
temporal aggregation.

Important
---------
This is an intentionally conservative re-association
rule.

It detects obvious fragmentation only.

It does NOT attempt general vehicle re-identification.

A broader appearance/trajectory-based re-ID system would
require separate validation.
"""


    SUMMARY_TXT.write_text(
        summary,
        encoding="utf-8",
    )


    print("\n" + "=" * 72)
    print("M24E COMPLETE")
    print("=" * 72)


    print(
        "\nDuplicate cross-track observations:",
        len(
            duplicate_rows
        ),
    )


    print(
        "Merged clusters:",
        len(
            cluster_rows
        ),
    )


    print()


    for row in cluster_rows:

        print(
            f"cluster "
            f"{row['cluster_id']} "
            f"<- tracks "
            f"{row['member_tracks']}"
        )


    print(
        "\nDuplicate pairs:"
    )

    print(
        DUPLICATE_CSV
    )


    print(
        "\nClusters:"
    )

    print(
        CLUSTER_CSV
    )


    print(
        "\nRe-associated frame evidence:"
    )

    print(
        FRAME_CSV
    )


    print(
        "\nSummary:"
    )

    print(
        SUMMARY_TXT
    )


if __name__ == "__main__":
    main()