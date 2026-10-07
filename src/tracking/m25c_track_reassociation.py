from pathlib import Path
from collections import defaultdict
import csv
import hashlib


ROOT = Path(__file__).resolve().parents[2]

TRACK_CSV = (
    ROOT
    / "outputs"
    / "M25_final_road_eval"
    / "M25B_tracking"
    / "track_summary.csv"
)

CROP_DIR = (
    ROOT
    / "outputs"
    / "M25_final_road_eval"
    / "M25B_tracking"
    / "best_plate_crops"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "M25_final_road_eval"
)

DUPLICATE_CSV = (
    OUTPUT_DIR
    / "M25C_duplicate_track_pairs.csv"
)

MERGED_CLUSTER_CSV = (
    OUTPUT_DIR
    / "M25C_merged_clusters.csv"
)

TRACK_MAP_CSV = (
    OUTPUT_DIR
    / "M25C_track_to_cluster.csv"
)

SUMMARY_TXT = (
    OUTPUT_DIR
    / "M25C_track_reassociation_summary.txt"
)


def load_csv(path):

    with open(
        path,
        newline="",
        encoding="utf-8",
    ) as file:

        return list(
            csv.DictReader(file)
        )


def sha256_file(path):

    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as file:

        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):

            digest.update(chunk)

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

        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        # deterministic:
        # numerically smaller track becomes cluster ID
        if int(ra) <= int(rb):
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def main():

    print("=" * 72)
    print(
        "M25C — FROZEN CONSERVATIVE TRACK RE-ASSOCIATION"
    )
    print("=" * 72)


    rows = load_csv(
        TRACK_CSV
    )

    uf = UnionFind()

    hash_groups = defaultdict(
        list
    )


    # ========================================================
    # HASH SAVED CROP FILES
    # ========================================================

    for row in rows:

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

            raise FileNotFoundError(
                path
            )

        digest = sha256_file(
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

            "sha256":
                digest,
        })


    # ========================================================
    # SAME FROZEN RULE AS M24E
    #
    # different tracker IDs
    # +
    # exact same saved crop bytes
    # +
    # same video frame
    #
    # OCR and ground truth are forbidden here.
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
                    a["track_id"]
                    ==
                    b["track_id"]
                ):
                    continue


                if (
                    a["frame"]
                    !=
                    b["frame"]
                ):
                    continue


                duplicate_rows.append({
                    "track_a":
                        a["track_id"],

                    "track_b":
                        b["track_id"],

                    "frame":
                        a["frame"],

                    "vehicle_a":
                        a["vehicle"],

                    "vehicle_b":
                        b["vehicle"],

                    "filename_a":
                        a["filename"],

                    "filename_b":
                        b["filename"],

                    "sha256":
                        digest,

                    "evidence":
                        "EXACT_DUPLICATE_SAME_FRAME",
                })


                uf.union(
                    a["track_id"],
                    b["track_id"],
                )


    # ========================================================
    # BUILD CLUSTERS
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


    merged_cluster_rows = []

    track_map_rows = []


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


        for member in members:

            track_map_rows.append({
                "original_track_id":
                    member,

                "cluster_id":
                    root,

                "cluster_members":
                    "|".join(
                        members
                    ),

                "merged":
                    len(
                        members
                    ) > 1,
            })


        if len(
            members
        ) > 1:

            merged_cluster_rows.append({
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
                        "exact same crop bytes "
                        "in same video frame"
                    ),
            })


    # ========================================================
    # SAVE
    # ========================================================

    duplicate_fields = [
        "track_a",
        "track_b",
        "frame",
        "vehicle_a",
        "vehicle_b",
        "filename_a",
        "filename_b",
        "sha256",
        "evidence",
    ]


    with open(
        DUPLICATE_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=duplicate_fields,
        )

        writer.writeheader()

        writer.writerows(
            duplicate_rows
        )


    merged_fields = [
        "cluster_id",
        "member_tracks",
        "member_count",
        "merge_basis",
    ]


    with open(
        MERGED_CLUSTER_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=merged_fields,
        )

        writer.writeheader()

        writer.writerows(
            merged_cluster_rows
        )


    map_fields = [
        "original_track_id",
        "cluster_id",
        "cluster_members",
        "merged",
    ]


    with open(
        TRACK_MAP_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=map_fields,
        )

        writer.writeheader()

        writer.writerows(
            track_map_rows
        )


    number_clusters = len(
        clusters
    )

    merged_tracks = sum(
        len(
            members
        )
        for members in clusters.values()
        if len(
            members
        ) > 1
    )


    summary = f"""M25C — FROZEN CONSERVATIVE TRACK RE-ASSOCIATION
============================================================

Input tracker IDs:
{len(uf.parent)}

Conservative track clusters:
{number_clusters}

Exact duplicate cross-track observations:
{len(duplicate_rows)}

Merged clusters:
{len(merged_cluster_rows)}

Tracker IDs participating in merges:
{merged_tracks}

Frozen rule
-----------
Two tracker IDs are connected only when:

1. they contain a saved crop from the SAME video frame
2. the saved crop files are byte-identical
3. the tracker IDs are different

No OCR prediction is used.

No plate ground truth is used.

No registration identity is used.

Purpose
-------
This applies the same conservative identity-fragmentation
logic developed in M24E to the unseen M25 road video.

It can recover only obvious duplicate tracker identities.

It is NOT a general vehicle re-identification algorithm.
"""


    SUMMARY_TXT.write_text(
        summary,
        encoding="utf-8",
    )


    print()
    print(
        "Tracker IDs:",
        len(
            uf.parent
        ),
    )

    print(
        "Conservative track clusters:",
        number_clusters,
    )

    print(
        "Duplicate observations:",
        len(
            duplicate_rows
        ),
    )

    print(
        "Merged clusters:",
        len(
            merged_cluster_rows
        ),
    )


    print()

    for row in merged_cluster_rows:

        print(
            f"cluster "
            f"{row['cluster_id']} "
            f"<- "
            f"{row['member_tracks']}"
        )


    print()
    print(
        "Duplicate pairs:",
        DUPLICATE_CSV
    )

    print(
        "Merged clusters:",
        MERGED_CLUSTER_CSV
    )

    print(
        "Track mapping:",
        TRACK_MAP_CSV
    )

    print(
        "Summary:",
        SUMMARY_TXT
    )


if __name__ == "__main__":
    main()