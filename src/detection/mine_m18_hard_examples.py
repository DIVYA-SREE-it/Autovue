from pathlib import Path
import csv
import shutil


ROOT = Path(__file__).resolve().parents[2]

M18_DIR = (
    ROOT
    / "outputs"
    / "M18_real_road_eval"
)

REVIEW_CSV = (
    M18_DIR
    / "manual_review.csv"
)

RAW_DIR = (
    M18_DIR
    / "raw_frames"
)

PANEL_DIR = (
    M18_DIR
    / "review_panels"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "M19_hard_example_mining"
)

PURE_NEG_DIR = (
    OUTPUT_DIR
    / "pure_hard_negatives"
)

MIXED_FP_DIR = (
    OUTPUT_DIR
    / "mixed_false_positive_frames"
)

FN_DIR = (
    OUTPUT_DIR
    / "false_negative_candidates"
)

for directory in [
    PURE_NEG_DIR,
    MIXED_FP_DIR,
    FN_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def to_int(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


def find_raw_frame(
    sample_id,
    video_name,
):

    video_folder = (
        RAW_DIR
        / Path(video_name).stem
    )

    matches = list(
        video_folder.glob(
            f"{sample_id}_*.jpg"
        )
    )

    if not matches:
        return None

    return matches[0]


def copy_candidate(
    source,
    destination,
):

    if source is None:
        return

    shutil.copy2(
        source,
        destination / source.name,
    )


def main():

    print("=" * 72)
    print("M19A — REAL-ROAD HARD-EXAMPLE MINING")
    print("=" * 72)

    with open(
        REVIEW_CSV,
        newline="",
        encoding="utf-8",
    ) as file:

        rows = list(
            csv.DictReader(file)
        )

    manifest = []

    pure_negative_count = 0
    mixed_fp_count = 0
    fn_count = 0

    for row in rows:

        if row.get("reviewed") != "1":
            continue

        sample_id = row["sample_id"]
        video = row["video"]

        visible = to_int(
            row.get(
                "visible_plate_count"
            )
        )

        fp = to_int(
            row.get(
                "false_positives"
            )
        )

        fn = to_int(
            row.get(
                "false_negatives"
            )
        )

        raw_path = find_raw_frame(
            sample_id,
            video,
        )

        panel_path = (
            PANEL_DIR
            / f"{sample_id}.jpg"
        )

        categories = []

        # --------------------------------------------
        # SAFE HARD NEGATIVE
        # --------------------------------------------

        if (
            fp > 0
            and visible == 0
        ):

            categories.append(
                "pure_hard_negative"
            )

            pure_negative_count += 1

            copy_candidate(
                raw_path,
                PURE_NEG_DIR,
            )

            if panel_path.exists():
                shutil.copy2(
                    panel_path,
                    PURE_NEG_DIR
                    / f"{sample_id}_panel.jpg",
                )

        # --------------------------------------------
        # MIXED FALSE-POSITIVE FRAME
        # --------------------------------------------

        if (
            fp > 0
            and visible > 0
        ):

            categories.append(
                "mixed_false_positive"
            )

            mixed_fp_count += 1

            copy_candidate(
                raw_path,
                MIXED_FP_DIR,
            )

            if panel_path.exists():
                shutil.copy2(
                    panel_path,
                    MIXED_FP_DIR
                    / f"{sample_id}_panel.jpg",
                )

        # --------------------------------------------
        # FALSE-NEGATIVE / DOMAIN POSITIVE
        # --------------------------------------------

        if fn > 0:

            categories.append(
                "false_negative_positive_candidate"
            )

            fn_count += 1

            copy_candidate(
                raw_path,
                FN_DIR,
            )

            if panel_path.exists():
                shutil.copy2(
                    panel_path,
                    FN_DIR
                    / f"{sample_id}_panel.jpg",
                )

        if categories:

            manifest.append({
                "sample_id":
                    sample_id,

                "video":
                    video,

                "timestamp_sec":
                    row[
                        "timestamp_sec"
                    ],

                "visible_plates":
                    visible,

                "detections":
                    to_int(
                        row.get(
                            "detections"
                        )
                    ),

                "true_positives":
                    to_int(
                        row.get(
                            "true_positives"
                        )
                    ),

                "false_positives":
                    fp,

                "false_negatives":
                    fn,

                "categories":
                    ";".join(
                        categories
                    ),
            })

    manifest_path = (
        OUTPUT_DIR
        / "M19_candidate_manifest.csv"
    )

    fields = [
        "sample_id",
        "video",
        "timestamp_sec",
        "visible_plates",
        "detections",
        "true_positives",
        "false_positives",
        "false_negatives",
        "categories",
    ]

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            manifest
        )

    summary = f"""M19A — REAL-ROAD HARD-EXAMPLE MINING
============================================================

Source:
M18 human-reviewed real-road evaluation

Pure hard-negative frames:
{pure_negative_count}

Definition:
False positive present AND zero visible real plates.
These frames can safely become YOLO negative examples
with empty annotation files.

Mixed false-positive frames:
{mixed_fp_count}

Definition:
False positive present AND at least one real plate.
These frames must NOT be treated as empty negatives.
Real plate boxes must be annotated before training.

False-negative / domain-positive candidates:
{fn_count}

Definition:
At least one clearly visible real plate was missed.
These are candidates for real-road positive annotation
and domain-adaptation training.

Important:
A single frame may belong to both mixed-FP and FN groups.

Current M18 baseline:
TP = 19
FP = 8
FN = 38
Precision = 0.7037
Recall = 0.3333
F1 = 0.4524

Primary M19 objective:
Improve real-road recall while reducing text/signboard
false positives without harming NPDS validation performance.
"""

    (
        OUTPUT_DIR
        / "M19A_summary.txt"
    ).write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print(
        "Pure hard negatives :",
        pure_negative_count,
    )

    print(
        "Mixed FP frames     :",
        mixed_fp_count,
    )

    print(
        "FN candidates       :",
        fn_count,
    )

    print(
        "\nManifest:"
    )

    print(
        manifest_path
    )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()