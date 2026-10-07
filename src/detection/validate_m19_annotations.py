from pathlib import Path
import csv
import math

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

M19_DIR = (
    ROOT
    / "outputs"
    / "M19_hard_example_mining"
)

MANIFEST = (
    M19_DIR
    / "M19B_adaptation_subset.csv"
)

IMAGE_DIR = (
    ROOT
    / "data"
    / "processed"
    / "m19_adaptation"
    / "images"
)

LABEL_DIR = (
    ROOT
    / "data"
    / "processed"
    / "m19_adaptation"
    / "labels"
)

PREVIEW_DIR = (
    M19_DIR
    / "annotation_previews"
)

PREVIEW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_yolo_labels(path):

    if not path.exists():
        return []

    text = path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return []

    boxes = []

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):

        parts = line.split()

        if len(parts) != 5:

            raise ValueError(
                f"{path.name}: "
                f"invalid line {line_number}"
            )

        class_id = int(
            parts[0]
        )

        xc, yc, w, h = map(
            float,
            parts[1:],
        )

        boxes.append({
            "class_id": class_id,
            "xc": xc,
            "yc": yc,
            "w": w,
            "h": h,
        })

    return boxes


def main():

    print("=" * 72)
    print("M19D — ANNOTATION VALIDATION")
    print("=" * 72)

    with open(
        MANIFEST,
        newline="",
        encoding="utf-8",
    ) as file:

        rows = list(
            csv.DictReader(file)
        )

    report = []
    preview_images = []

    total_expected = 0
    total_boxes = 0
    errors = []

    for row in rows:

        sample_id = row[
            "sample_id"
        ]

        expected = int(
            row[
                "visible_plates"
            ]
        )

        image_path = (
            IMAGE_DIR
            / f"{sample_id}.jpg"
        )

        label_path = (
            LABEL_DIR
            / f"{sample_id}.txt"
        )

        if not image_path.exists():

            errors.append(
                f"{sample_id}: image missing"
            )

            continue

        if not label_path.exists():

            errors.append(
                f"{sample_id}: label missing"
            )

            continue

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            errors.append(
                f"{sample_id}: "
                f"image unreadable"
            )

            continue

        height, width = (
            image.shape[:2]
        )

        try:

            boxes = load_yolo_labels(
                label_path
            )

        except Exception as exc:

            errors.append(
                str(exc)
            )

            continue

        total_expected += (
            expected
        )

        total_boxes += len(
            boxes
        )

        valid = True
        issues = []

        if len(boxes) != expected:

            valid = False

            issues.append(
                f"expected {expected}, "
                f"found {len(boxes)}"
            )

        preview = image.copy()

        for box_number, box in enumerate(
            boxes,
            start=1,
        ):

            if box["class_id"] != 0:

                valid = False

                issues.append(
                    f"box {box_number}: "
                    f"class != 0"
                )

            for key in [
                "xc",
                "yc",
                "w",
                "h",
            ]:

                if not (
                    0.0
                    <= box[key]
                    <= 1.0
                ):

                    valid = False

                    issues.append(
                        f"box {box_number}: "
                        f"{key} outside 0..1"
                    )

            if (
                box["w"] <= 0
                or box["h"] <= 0
            ):

                valid = False

                issues.append(
                    f"box {box_number}: "
                    f"non-positive size"
                )

            x1 = int(
                (
                    box["xc"]
                    - box["w"] / 2
                )
                * width
            )

            y1 = int(
                (
                    box["yc"]
                    - box["h"] / 2
                )
                * height
            )

            x2 = int(
                (
                    box["xc"]
                    + box["w"] / 2
                )
                * width
            )

            y2 = int(
                (
                    box["yc"]
                    + box["h"] / 2
                )
                * height
            )

            x1 = max(
                0,
                min(
                    x1,
                    width - 1,
                ),
            )

            y1 = max(
                0,
                min(
                    y1,
                    height - 1,
                ),
            )

            x2 = max(
                0,
                min(
                    x2,
                    width - 1,
                ),
            )

            y2 = max(
                0,
                min(
                    y2,
                    height - 1,
                ),
            )

            cv2.rectangle(
                preview,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                4,
            )

            cv2.putText(
                preview,
                f"Plate {box_number}",
                (
                    x1,
                    max(
                        30,
                        y1 - 8,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            preview,
            (
                f"{sample_id} | "
                f"expected={expected} | "
                f"boxes={len(boxes)}"
            ),
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

        preview_path = (
            PREVIEW_DIR
            / f"{sample_id}.jpg"
        )

        cv2.imwrite(
            str(preview_path),
            preview,
        )

        # Resize for contact sheet.
        target_width = 640

        scale = (
            target_width
            / preview.shape[1]
        )

        target_height = int(
            preview.shape[0]
            * scale
        )

        thumb = cv2.resize(
            preview,
            (
                target_width,
                target_height,
            ),
        )

        preview_images.append(
            thumb
        )

        report.append({
            "sample_id":
                sample_id,

            "expected_boxes":
                expected,

            "actual_boxes":
                len(boxes),

            "valid":
                valid,

            "issues":
                "; ".join(
                    issues
                ),
        })


    # ========================================================
    # REPORT CSV
    # ========================================================

    report_path = (
        M19_DIR
        / "M19D_annotation_validation.csv"
    )

    with open(
        report_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_id",
                "expected_boxes",
                "actual_boxes",
                "valid",
                "issues",
            ],
        )

        writer.writeheader()
        writer.writerows(
            report
        )


    # ========================================================
    # CONTACT SHEET
    # ========================================================

    if preview_images:

        columns = 3

        rows_count = math.ceil(
            len(preview_images)
            / columns
        )

        max_height = max(
            img.shape[0]
            for img in preview_images
        )

        max_width = max(
            img.shape[1]
            for img in preview_images
        )

        contact_sheet = np.zeros(
            (
                rows_count
                * max_height,

                columns
                * max_width,

                3,
            ),
            dtype=np.uint8,
        )

        for index, img in enumerate(
            preview_images
        ):

            row_index = (
                index
                // columns
            )

            column_index = (
                index
                % columns
            )

            y = (
                row_index
                * max_height
            )

            x = (
                column_index
                * max_width
            )

            contact_sheet[
                y:y + img.shape[0],
                x:x + img.shape[1],
            ] = img

        contact_path = (
            M19_DIR
            / "M19D_annotation_contact_sheet.jpg"
        )

        cv2.imwrite(
            str(contact_path),
            contact_sheet,
        )

    else:

        contact_path = None


    # ========================================================
    # SUMMARY
    # ========================================================

    invalid_rows = [
        row
        for row in report
        if not row["valid"]
    ]

    summary = f"""M19D — ANNOTATION VALIDATION
============================================================

Adaptation images:
{len(rows)}

Expected plate boxes:
{total_expected}

Actual plate boxes:
{total_boxes}

Structurally valid images:
{len(report) - len(invalid_rows)}

Invalid images:
{len(invalid_rows)}

Missing/unreadable errors:
{len(errors)}

Overall structural validation:
{"PASS" if not invalid_rows and not errors else "FAIL"}

Important:
Structural validation confirms YOLO label syntax,
box count and coordinate range.

The contact sheet must still be visually inspected
to confirm each box tightly covers a real plate.
"""

    summary_path = (
        M19_DIR
        / "M19D_summary.txt"
    )

    summary_path.write_text(
        summary,
        encoding="utf-8",
    )


    # ========================================================
    # PRINT
    # ========================================================

    print()
    print(
        f"Images checked       : "
        f"{len(rows)}"
    )

    print(
        f"Expected boxes       : "
        f"{total_expected}"
    )

    print(
        f"Actual boxes         : "
        f"{total_boxes}"
    )

    print(
        f"Invalid annotations  : "
        f"{len(invalid_rows)}"
    )

    print(
        f"Other errors         : "
        f"{len(errors)}"
    )

    print()

    if (
        not invalid_rows
        and not errors
    ):

        print(
            "STRUCTURAL VALIDATION: PASS"
        )

    else:

        print(
            "STRUCTURAL VALIDATION: FAIL"
        )

        for error in errors:

            print(
                " -",
                error,
            )

        for row in invalid_rows:

            print(
                " -",
                row[
                    "sample_id"
                ],
                row[
                    "issues"
                ],
            )

    print(
        "\nSummary:"
    )

    print(
        summary_path
    )

    print(
        "\nContact sheet:"
    )

    print(
        contact_path
    )


if __name__ == "__main__":
    main()