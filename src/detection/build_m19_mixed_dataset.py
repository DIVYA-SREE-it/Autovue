from pathlib import Path
import csv
import os
import random
import shutil
import yaml


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SEED = 42

NPDS_SAMPLE_COUNT = 2500

ROAD_REPEAT_COUNT = 20


# ============================================================
# SOURCE PATHS
# ============================================================

NPDS_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "npds_leakage_safe"
)

NPDS_TRAIN_IMAGES = (
    NPDS_ROOT
    / "images"
    / "train"
)

NPDS_TRAIN_LABELS = (
    NPDS_ROOT
    / "labels"
    / "train"
)

NPDS_VAL_IMAGES = (
    NPDS_ROOT
    / "images"
    / "val"
)

ROAD_IMAGES = (
    ROOT
    / "data"
    / "processed"
    / "m19_adaptation"
    / "images"
)

ROAD_LABELS = (
    ROOT
    / "data"
    / "processed"
    / "m19_adaptation"
    / "labels"
)


# ============================================================
# DESTINATION PATHS
# ============================================================

MIXED_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "m19_mixed_train"
)

MIXED_IMAGES = (
    MIXED_ROOT
    / "images"
    / "train"
)

MIXED_LABELS = (
    MIXED_ROOT
    / "labels"
    / "train"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "M19_domain_adaptation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HELPERS
# ============================================================

def link_or_copy(
    source,
    destination,
):

    """
    Use a hard link where possible so we do not
    unnecessarily duplicate image data.
    Fall back to copying if linking fails.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        destination.unlink()

    try:

        os.link(
            source,
            destination,
        )

        return "hardlink"

    except OSError:

        shutil.copy2(
            source,
            destination,
        )

        return "copy"


def count_boxes(
    label_path,
):

    if not label_path.exists():
        return 0

    count = 0

    for line in label_path.read_text(
        encoding="utf-8"
    ).splitlines():

        if len(
            line.split()
        ) == 5:

            count += 1

    return count


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("M19E — BUILD MIXED DOMAIN-ADAPTATION DATASET")
    print("=" * 72)

    random.seed(
        SEED
    )

    # --------------------------------------------------------
    # CLEAN PREVIOUS GENERATED MIXED DATASET
    # --------------------------------------------------------

    if MIXED_ROOT.exists():

        shutil.rmtree(
            MIXED_ROOT
        )

    MIXED_IMAGES.mkdir(
        parents=True,
        exist_ok=True,
    )

    MIXED_LABELS.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOAD NPDS TRAIN IMAGES
    # --------------------------------------------------------

    npds_images = sorted([
        path
        for path in NPDS_TRAIN_IMAGES.iterdir()
        if path.suffix.lower()
        in {
            ".jpg",
            ".jpeg",
            ".png",
        }
    ])

    if len(npds_images) < NPDS_SAMPLE_COUNT:

        raise RuntimeError(
            f"Requested {NPDS_SAMPLE_COUNT} NPDS images, "
            f"but only {len(npds_images)} exist."
        )

    selected_npds = random.sample(
        npds_images,
        NPDS_SAMPLE_COUNT,
    )

    # --------------------------------------------------------
    # LOAD ROAD IMAGES
    # --------------------------------------------------------

    road_images = sorted(
        ROAD_IMAGES.glob(
            "*.jpg"
        )
    )

    if len(road_images) != 12:

        raise RuntimeError(
            f"Expected 12 road images, "
            f"found {len(road_images)}."
        )

    # --------------------------------------------------------
    # MANIFEST
    # --------------------------------------------------------

    manifest_rows = []

    npds_box_count = 0
    road_box_count_original = 0
    road_box_count_repeated = 0

    link_count = 0
    copy_count = 0

    # ========================================================
    # ADD NPDS SAMPLES
    # ========================================================

    print(
        f"\nAdding "
        f"{len(selected_npds)} "
        f"NPDS training images..."
    )

    for index, image_path in enumerate(
        selected_npds,
        start=1,
    ):

        source_label = (
            NPDS_TRAIN_LABELS
            / f"{image_path.stem}.txt"
        )

        if not source_label.exists():

            raise RuntimeError(
                f"Missing NPDS label: "
                f"{source_label}"
            )

        destination_stem = (
            f"npds_{index:05d}_"
            f"{image_path.stem}"
        )

        destination_image = (
            MIXED_IMAGES
            / (
                destination_stem
                + image_path.suffix.lower()
            )
        )

        destination_label = (
            MIXED_LABELS
            / f"{destination_stem}.txt"
        )

        mode = link_or_copy(
            image_path,
            destination_image,
        )

        link_or_copy(
            source_label,
            destination_label,
        )

        if mode == "hardlink":
            link_count += 1
        else:
            copy_count += 1

        boxes = count_boxes(
            source_label
        )

        npds_box_count += (
            boxes
        )

        manifest_rows.append({
            "destination_stem":
                destination_stem,

            "domain":
                "npds",

            "source_image":
                str(image_path),

            "source_label":
                str(source_label),

            "repeat_index":
                0,

            "plate_boxes":
                boxes,
        })


    # ========================================================
    # ADD ROAD ADAPTATION SAMPLES
    # ========================================================

    print(
        f"\nAdding "
        f"{len(road_images)} "
        f"road images × "
        f"{ROAD_REPEAT_COUNT} repeats..."
    )

    for image_path in road_images:

        source_label = (
            ROAD_LABELS
            / f"{image_path.stem}.txt"
        )

        if not source_label.exists():

            raise RuntimeError(
                f"Missing road label: "
                f"{source_label}"
            )

        boxes = count_boxes(
            source_label
        )

        road_box_count_original += (
            boxes
        )

        for repeat_index in range(
            1,
            ROAD_REPEAT_COUNT + 1,
        ):

            destination_stem = (
                f"road_r{repeat_index:02d}_"
                f"{image_path.stem}"
            )

            destination_image = (
                MIXED_IMAGES
                / (
                    destination_stem
                    + image_path.suffix.lower()
                )
            )

            destination_label = (
                MIXED_LABELS
                / f"{destination_stem}.txt"
            )

            mode = link_or_copy(
                image_path,
                destination_image,
            )

            link_or_copy(
                source_label,
                destination_label,
            )

            if mode == "hardlink":
                link_count += 1
            else:
                copy_count += 1

            road_box_count_repeated += (
                boxes
            )

            manifest_rows.append({
                "destination_stem":
                    destination_stem,

                "domain":
                    "road",

                "source_image":
                    str(image_path),

                "source_label":
                    str(source_label),

                "repeat_index":
                    repeat_index,

                "plate_boxes":
                    boxes,
            })


    # ========================================================
    # CREATE DATASET YAML
    # ========================================================

    dataset_yaml = {
        "path":
            str(
                MIXED_ROOT.resolve()
            ),

        "train":
            "images/train",

        # Keep validation completely original.
        "val":
            str(
                NPDS_VAL_IMAGES.resolve()
            ),

        "names": {
            0:
                "Number-Plate",
        },
    }

    yaml_path = (
        MIXED_ROOT
        / "dataset.yaml"
    )

    with open(
        yaml_path,
        "w",
        encoding="utf-8",
    ) as file:

        yaml.safe_dump(
            dataset_yaml,
            file,
            sort_keys=False,
        )


    # ========================================================
    # SAVE MANIFEST
    # ========================================================

    manifest_path = (
        OUTPUT_DIR
        / "M19E_training_manifest.csv"
    )

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "destination_stem",
                "domain",
                "source_image",
                "source_label",
                "repeat_index",
                "plate_boxes",
            ],
        )

        writer.writeheader()
        writer.writerows(
            manifest_rows
        )


    # ========================================================
    # FINAL COUNTS
    # ========================================================

    mixed_images = list(
        MIXED_IMAGES.glob("*")
    )

    mixed_labels = list(
        MIXED_LABELS.glob("*.txt")
    )

    road_instances = (
        len(road_images)
        * ROAD_REPEAT_COUNT
    )

    total_instances = (
        NPDS_SAMPLE_COUNT
        + road_instances
    )

    road_fraction = (
        road_instances
        / total_instances
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = f"""M19E — MIXED DOMAIN-ADAPTATION DATASET
============================================================

Purpose
-------
Create a controlled fine-tuning dataset combining
leakage-safe NPDS training examples with manually annotated
real-road adaptation examples.

Random seed:
{SEED}

NPDS source
-----------
Original leakage-safe training images:
{len(npds_images)}

Selected NPDS training images:
{NPDS_SAMPLE_COUNT}

Selected NPDS plate boxes:
{npds_box_count}

Real-road adaptation source
---------------------------
Unique road images:
{len(road_images)}

Unique road plate boxes:
{road_box_count_original}

Road repeat factor:
{ROAD_REPEAT_COUNT}

Effective road training instances:
{road_instances}

Effective repeated road plate boxes:
{road_box_count_repeated}

Mixed training dataset
----------------------
Total training instances:
{total_instances}

Road-instance fraction:
{road_fraction:.4f}
({road_fraction * 100:.2f}%)

Generated image files:
{len(mixed_images)}

Generated label files:
{len(mixed_labels)}

Hard-linked images:
{link_count}

Copied images:
{copy_count}

Validation
----------
Validation images remain the original leakage-safe NPDS
validation split:

{NPDS_VAL_IMAGES}

Evaluation separation
---------------------
traffic_baseline.mp4:
USED for domain adaptation.

traffic_ocr.mp4:
NOT used for training.
Reserved for held-out real-road comparison.

Important methodology
---------------------
The NPDS test split remains untouched.

M19 fine-tuning/model selection will use NPDS validation
plus the separately held-out traffic_ocr road evaluation.
"""

    summary_path = (
        OUTPUT_DIR
        / "M19E_summary.txt"
    )

    summary_path.write_text(
        summary,
        encoding="utf-8",
    )


    # ========================================================
    # PRINT
    # ========================================================

    print("\n" + "=" * 72)
    print("M19E DATASET CREATED")
    print("=" * 72)

    print(
        f"\nNPDS instances : "
        f"{NPDS_SAMPLE_COUNT}"
    )

    print(
        f"Road instances : "
        f"{road_instances}"
    )

    print(
        f"Road fraction  : "
        f"{road_fraction * 100:.2f}%"
    )

    print(
        f"\nTotal images   : "
        f"{len(mixed_images)}"
    )

    print(
        f"Total labels   : "
        f"{len(mixed_labels)}"
    )

    print(
        f"\nRoad boxes "
        f"(unique)      : "
        f"{road_box_count_original}"
    )

    print(
        "\nDataset YAML:"
    )

    print(
        yaml_path
    )

    print(
        "\nManifest:"
    )

    print(
        manifest_path
    )

    print(
        "\nSummary:"
    )

    print(
        summary_path
    )


if __name__ == "__main__":
    main()