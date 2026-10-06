from pathlib import Path
import json
import os
import random

import cv2
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "iurs_npds_extracted"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "npds_yolo"

PREVIEW_DIR = PROJECT_ROOT / "outputs" / "M03_plate_dataset"


SPLITS = {
    "train": "train",
    "valid": "val",
    "test": "test",
}


def link_file(source: Path, destination: Path):
    """
    Create a hard link instead of copying the image.
    This avoids using another ~1 GB of disk space.
    """

    if destination.exists():
        return

    try:
        os.link(source, destination)
    except OSError:
        # Fallback to symbolic link
        destination.symlink_to(source.resolve())


def convert_split(source_split, target_split):

    source_dir = RAW_ROOT / source_split
    annotation_file = source_dir / "_annotations.coco.json"

    image_output = OUTPUT_ROOT / "images" / target_split
    label_output = OUTPUT_ROOT / "labels" / target_split

    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    with open(annotation_file, "r", encoding="utf-8") as file:
        coco = json.load(file)

    images = {
        image["id"]: image
        for image in coco["images"]
    }

    # Find all category IDs named Number-Plate.
    # Roboflow exported duplicate category definitions,
    # so both IDs are intentionally collapsed into YOLO class 0.
    plate_category_ids = {
        category["id"]
        for category in coco["categories"]
        if category["name"].lower().replace("_", "-") == "number-plate"
    }

    print(
        f"\n{source_split.upper()} plate category IDs:",
        plate_category_ids
    )

    annotations_by_image = {}

    for annotation in coco["annotations"]:

        if annotation["category_id"] not in plate_category_ids:
            continue

        image_id = annotation["image_id"]

        annotations_by_image.setdefault(
            image_id,
            []
        ).append(annotation)

    converted_boxes = 0
    invalid_boxes = 0

    for image_id, image_info in images.items():

        filename = image_info["file_name"]

        width = image_info["width"]
        height = image_info["height"]

        source_image = source_dir / filename
        destination_image = image_output / filename

        if not source_image.exists():
            print(
                f"WARNING: Missing image: {source_image}"
            )
            continue

        link_file(
            source_image,
            destination_image
        )

        label_path = (
            label_output
            / f"{Path(filename).stem}.txt"
        )

        yolo_lines = []

        annotations = annotations_by_image.get(
            image_id,
            []
        )

        for annotation in annotations:

            x, y, box_width, box_height = annotation["bbox"]

            if box_width <= 0 or box_height <= 0:
                invalid_boxes += 1
                continue

            # COCO:
            # x = left
            # y = top
            # width
            # height

            # Clip box to image dimensions
            x = max(0, x)
            y = max(0, y)

            box_width = min(
                box_width,
                width - x
            )

            box_height = min(
                box_height,
                height - y
            )

            if box_width <= 0 or box_height <= 0:
                invalid_boxes += 1
                continue

            center_x = (
                x + box_width / 2
            ) / width

            center_y = (
                y + box_height / 2
            ) / height

            normalized_width = (
                box_width / width
            )

            normalized_height = (
                box_height / height
            )

            line = (
                f"0 "
                f"{center_x:.6f} "
                f"{center_y:.6f} "
                f"{normalized_width:.6f} "
                f"{normalized_height:.6f}"
            )

            yolo_lines.append(line)

            converted_boxes += 1

        label_path.write_text(
            "\n".join(yolo_lines),
            encoding="utf-8"
        )

    print(
        f"{source_split}: "
        f"{len(images)} images, "
        f"{converted_boxes} boxes, "
        f"{invalid_boxes} invalid boxes"
    )


def create_yaml():

    yaml_data = {
        "path": str(OUTPUT_ROOT),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "number_plate"
        }
    }

    yaml_path = OUTPUT_ROOT / "dataset.yaml"

    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as file:

        yaml.safe_dump(
            yaml_data,
            file,
            sort_keys=False
        )

    print(
        "\nDataset YAML:",
        yaml_path
    )


def create_preview():

    PREVIEW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    image_dir = OUTPUT_ROOT / "images" / "train"
    label_dir = OUTPUT_ROOT / "labels" / "train"

    image_files = list(
        image_dir.glob("*.jpg")
    )

    samples = random.sample(
        image_files,
        min(12, len(image_files))
    )

    previews = []

    for image_path in samples:

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        height, width = image.shape[:2]

        label_path = (
            label_dir
            / f"{image_path.stem}.txt"
        )

        if label_path.exists():

            for line in label_path.read_text().splitlines():

                values = line.split()

                if len(values) != 5:
                    continue

                _, cx, cy, bw, bh = map(
                    float,
                    values
                )

                x1 = int(
                    (cx - bw / 2) * width
                )

                y1 = int(
                    (cy - bh / 2) * height
                )

                x2 = int(
                    (cx + bw / 2) * width
                )

                y2 = int(
                    (cy + bh / 2) * height
                )

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )

                cv2.putText(
                    image,
                    "number_plate",
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        image = cv2.resize(
            image,
            (320, 320)
        )

        previews.append(image)

    if not previews:
        print("Could not generate preview.")
        return

    rows = []

    for i in range(0, len(previews), 4):

        row = previews[i:i + 4]

        while len(row) < 4:
            row.append(
                row[-1].copy()
            )

        rows.append(
            cv2.hconcat(row)
        )

    contact_sheet = cv2.vconcat(
        rows
    )

    preview_path = (
        PREVIEW_DIR
        / "npds_annotation_preview.jpg"
    )

    cv2.imwrite(
        str(preview_path),
        contact_sheet
    )

    print(
        "Preview:",
        preview_path
    )


def main():

    print("=" * 60)
    print("NPDS COCO -> YOLO PREPARATION")
    print("=" * 60)

    for source_split, target_split in SPLITS.items():

        convert_split(
            source_split,
            target_split
        )

    create_yaml()
    create_preview()

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()