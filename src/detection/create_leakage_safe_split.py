from pathlib import Path
from collections import defaultdict
import csv
import os
import random
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]

SOURCE_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "npds_yolo"
)

OUTPUT_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "npds_leakage_safe"
)

SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

ORIGINAL_SPLITS = [
    "train",
    "val",
    "test",
]


def source_id(filename: str) -> str:
    """
    Example:

    V_P_I_1889_jpg.rf.39199d....jpg
        ->
    V_P_I_1889
    """

    match = re.match(
        r"(.+?)_(?:jpg|jpeg|png)\.rf\.[^.]+\.(?:jpg|jpeg|png)$",
        filename,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1)

    return Path(filename).stem


def image_files(folder: Path):

    files = []

    for extension in (
        "*.jpg",
        "*.jpeg",
        "*.png",
    ):
        files.extend(folder.glob(extension))

    return sorted(files)


def link_or_copy(source: Path, destination: Path):

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if destination.exists():
        destination.unlink()

    try:

        os.link(
            source,
            destination
        )

    except OSError:

        shutil.copy2(
            source,
            destination
        )


def main():

    print("=" * 72)
    print("M14B - CREATE LEAKAGE-SAFE NPDS SPLIT")
    print("=" * 72)

    groups = defaultdict(list)

    missing_labels = []

    total_images = 0

    for old_split in ORIGINAL_SPLITS:

        image_dir = (
            SOURCE_ROOT
            / "images"
            / old_split
        )

        label_dir = (
            SOURCE_ROOT
            / "labels"
            / old_split
        )

        for image_path in image_files(
            image_dir
        ):

            total_images += 1

            label_path = (
                label_dir
                / f"{image_path.stem}.txt"
            )

            if not label_path.exists():

                missing_labels.append(
                    str(image_path)
                )

                continue

            sid = source_id(
                image_path.name
            )

            groups[sid].append({
                "source_id": sid,
                "old_split": old_split,
                "image": image_path,
                "label": label_path,
            })

    if missing_labels:

        print(
            "\nERROR: Missing labels:",
            len(missing_labels)
        )

        for item in missing_labels[:20]:
            print(item)

        raise RuntimeError(
            "Dataset contains images without labels."
        )

    source_ids = sorted(
        groups.keys()
    )

    print(
        "\nTotal images:",
        total_images
    )

    print(
        "Unique source IDs:",
        len(source_ids)
    )

    print(
        "Average images/source:",
        round(
            total_images
            / len(source_ids),
            2
        )
    )

    rng = random.Random(SEED)

    rng.shuffle(
        source_ids
    )

    source_count = len(
        source_ids
    )

    train_count = int(
        source_count
        * TRAIN_RATIO
    )

    val_count = int(
        source_count
        * VAL_RATIO
    )

    train_ids = set(
        source_ids[
            :train_count
        ]
    )

    val_ids = set(
        source_ids[
            train_count:
            train_count
            + val_count
        ]
    )

    test_ids = set(
        source_ids[
            train_count
            + val_count:
        ]
    )

    new_split_for_source = {}

    for sid in train_ids:
        new_split_for_source[sid] = "train"

    for sid in val_ids:
        new_split_for_source[sid] = "val"

    for sid in test_ids:
        new_split_for_source[sid] = "test"

    if OUTPUT_ROOT.exists():

        shutil.rmtree(
            OUTPUT_ROOT
        )

    for split in (
        "train",
        "val",
        "test",
    ):

        (
            OUTPUT_ROOT
            / "images"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        (
            OUTPUT_ROOT
            / "labels"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    manifest_rows = []

    image_counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    source_counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    for sid, records in groups.items():

        new_split = (
            new_split_for_source[
                sid
            ]
        )

        source_counts[
            new_split
        ] += 1

        for record in records:

            image_path = record[
                "image"
            ]

            label_path = record[
                "label"
            ]

            destination_image = (
                OUTPUT_ROOT
                / "images"
                / new_split
                / image_path.name
            )

            destination_label = (
                OUTPUT_ROOT
                / "labels"
                / new_split
                / label_path.name
            )

            link_or_copy(
                image_path,
                destination_image
            )

            link_or_copy(
                label_path,
                destination_label
            )

            image_counts[
                new_split
            ] += 1

            manifest_rows.append({
                "source_id": sid,
                "original_split": record[
                    "old_split"
                ],
                "new_split": new_split,
                "image_filename": (
                    image_path.name
                ),
                "label_filename": (
                    label_path.name
                ),
            })

    manifest_path = (
        OUTPUT_ROOT
        / "split_manifest.csv"
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
                "source_id",
                "original_split",
                "new_split",
                "image_filename",
                "label_filename",
            ],
        )

        writer.writeheader()

        writer.writerows(
            manifest_rows
        )

    yaml_path = (
        OUTPUT_ROOT
        / "dataset.yaml"
    )

    yaml_text = f"""path: {OUTPUT_ROOT}
train: images/train
val: images/val
test: images/test

names:
  0: Number-Plate
"""

    yaml_path.write_text(
        yaml_text,
        encoding="utf-8",
    )


    assert not (
        train_ids
        & val_ids
    )

    assert not (
        train_ids
        & test_ids
    )

    assert not (
        val_ids
        & test_ids
    )

    print("\n" + "=" * 72)
    print("NEW LEAKAGE-SAFE SPLIT")
    print("=" * 72)

    for split in (
        "train",
        "val",
        "test",
    ):

        print(
            f"{split.upper():5s} "
            f"source IDs = "
            f"{source_counts[split]:4d} | "
            f"images = "
            f"{image_counts[split]:5d}"
        )

    print(
        "\nTrain ∩ Val :",
        len(
            train_ids
            & val_ids
        )
    )

    print(
        "Train ∩ Test:",
        len(
            train_ids
            & test_ids
        )
    )

    print(
        "Val ∩ Test  :",
        len(
            val_ids
            & test_ids
        )
    )

    print(
        "\nSeed:",
        SEED
    )

    print(
        "Dataset YAML:",
        yaml_path
    )

    print(
        "Manifest:",
        manifest_path
    )

    print(
        "\nM14B COMPLETE"
    )


if __name__ == "__main__":
    main()