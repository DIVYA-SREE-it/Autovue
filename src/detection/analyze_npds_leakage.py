from pathlib import Path
from collections import defaultdict
import re


ROOT = Path(__file__).resolve().parents[2]

DATASET = (
    ROOT
    / "data"
    / "processed"
    / "npds_yolo"
    / "images"
)

SPLITS = ["train", "val", "test"]


def source_id(path: Path) -> str:
    """
    Convert Roboflow-style filenames such as:

    V_P_I_1889_jpg.rf.39199d00....jpg

    into:

    V_P_I_1889
    """

    name = path.name

    match = re.match(
        r"(.+?)_(?:jpg|jpeg|png)\.rf\.[^.]+\.(?:jpg|jpeg|png)$",
        name,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1)

    return path.stem


def collect_split(split):
    folder = DATASET / split

    images = []

    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(folder.glob(pattern))

    groups = defaultdict(list)

    for image in images:
        groups[source_id(image)].append(image.name)

    return images, groups


def main():

    print("=" * 72)
    print("M14A - NPDS SOURCE-LEVEL LEAKAGE ANALYSIS")
    print("=" * 72)

    split_images = {}
    split_groups = {}

    for split in SPLITS:

        images, groups = collect_split(split)

        split_images[split] = images
        split_groups[split] = groups

        augmented_groups = sum(
            1
            for files in groups.values()
            if len(files) > 1
        )

        print(f"\n{split.upper()}")
        print(f"  Images            : {len(images)}")
        print(f"  Unique source IDs : {len(groups)}")
        print(f"  Multi-image groups: {augmented_groups}")

    train_ids = set(split_groups["train"])
    val_ids = set(split_groups["val"])
    test_ids = set(split_groups["test"])

    train_val = train_ids & val_ids
    train_test = train_ids & test_ids
    val_test = val_ids & test_ids

    all_three = (
        train_ids
        & val_ids
        & test_ids
    )

    leaked_sources = (
        train_val
        | train_test
        | val_test
    )

    print("\n" + "=" * 72)
    print("CROSS-SPLIT SOURCE OVERLAP")
    print("=" * 72)

    print(
        "Train ∩ Val  :",
        len(train_val)
    )

    print(
        "Train ∩ Test :",
        len(train_test)
    )

    print(
        "Val ∩ Test   :",
        len(val_test)
    )

    print(
        "All 3 splits :",
        len(all_three)
    )

    print(
        "Total source IDs appearing "
        "in more than one split:",
        len(leaked_sources)
    )

    if leaked_sources:

        print("\nExample overlapping source IDs:")

        for sid in sorted(
            leaked_sources
        )[:20]:

            locations = []

            for split in SPLITS:

                if sid in split_groups[split]:

                    locations.append(
                        f"{split}:"
                        f"{len(split_groups[split][sid])}"
                    )

            print(
                f"  {sid:<25} "
                + " | ".join(locations)
            )

        print(
            "\nRESULT: POTENTIAL SOURCE-LEVEL "
            "DATA LEAKAGE DETECTED"
        )

    else:

        print(
            "\nRESULT: No source-level overlap "
            "detected between splits."
        )


if __name__ == "__main__":
    main()