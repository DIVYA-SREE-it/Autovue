from pathlib import Path
import csv
import math
import re
import shutil

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

NPDS_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "npds_leakage_safe"
)

DATASET_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "m20_ocr_benchmark"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "M20_ocr_dataset"
)

DEV_COUNT = 30
TEST_COUNT = 15

PAD_RATIO = 0.08


# ============================================================
# HELPERS
# ============================================================

def source_id_from_name(filename):

    stem = Path(filename).stem

    match = re.search(
        r"(V_P_I_\d+)",
        stem,
    )

    if match:
        return match.group(1)

    # Fallback:
    # remove Roboflow hash suffix if present.
    return stem.split(
        "_jpg.rf."
    )[0]


def load_boxes(label_path):

    boxes = []

    if not label_path.exists():
        return boxes

    for line in label_path.read_text(
        encoding="utf-8"
    ).splitlines():

        parts = line.split()

        if len(parts) != 5:
            continue

        class_id = int(
            parts[0]
        )

        if class_id != 0:
            continue

        xc, yc, w, h = map(
            float,
            parts[1:],
        )

        boxes.append(
            (
                xc,
                yc,
                w,
                h,
            )
        )

    return boxes


def yolo_to_pixels(
    box,
    width,
    height,
):

    xc, yc, w, h = box

    x1 = (
        xc - w / 2
    ) * width

    y1 = (
        yc - h / 2
    ) * height

    x2 = (
        xc + w / 2
    ) * width

    y2 = (
        yc + h / 2
    ) * height

    return (
        x1,
        y1,
        x2,
        y2,
    )


def crop_with_padding(
    image,
    box,
):

    height, width = (
        image.shape[:2]
    )

    x1, y1, x2, y2 = (
        yolo_to_pixels(
            box,
            width,
            height,
        )
    )

    box_width = (
        x2 - x1
    )

    box_height = (
        y2 - y1
    )

    pad_x = (
        box_width
        * PAD_RATIO
    )

    pad_y = (
        box_height
        * PAD_RATIO
    )

    x1 = int(
        max(
            0,
            x1 - pad_x,
        )
    )

    y1 = int(
        max(
            0,
            y1 - pad_y,
        )
    )

    x2 = int(
        min(
            width,
            x2 + pad_x,
        )
    )

    y2 = int(
        min(
            height,
            y2 + pad_y,
        )
    )

    crop = image[
        y1:y2,
        x1:x2,
    ]

    return (
        crop,
        x1,
        y1,
        x2,
        y2,
    )


def quality_metrics(crop):

    if crop.size == 0:
        return None

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY,
    )

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    contrast = float(
        gray.std()
    )

    height, width = (
        gray.shape[:2]
    )

    # Large plate characters matter strongly for OCR.
    size_score = (
        0.60
        * min(
            1.0,
            width / 180.0,
        )
        +
        0.40
        * min(
            1.0,
            height / 60.0,
        )
    )

    sharp_score = min(
        1.0,
        sharpness / 800.0,
    )

    contrast_score = min(
        1.0,
        contrast / 60.0,
    )

    quality_score = (
        0.50 * size_score
        +
        0.35 * sharp_score
        +
        0.15 * contrast_score
    )

    return {
        "width":
            width,

        "height":
            height,

        "sharpness":
            sharpness,

        "contrast":
            contrast,

        "quality_score":
            quality_score,
    }


def collect_candidates(split):

    image_dir = (
        NPDS_ROOT
        / "images"
        / split
    )

    label_dir = (
        NPDS_ROOT
        / "labels"
        / split
    )

    candidates = []

    image_paths = sorted(
        list(
            image_dir.glob(
                "*.jpg"
            )
        )
        +
        list(
            image_dir.glob(
                "*.jpeg"
            )
        )
        +
        list(
            image_dir.glob(
                "*.png"
            )
        )
    )

    for image_path in image_paths:

        label_path = (
            label_dir
            / f"{image_path.stem}.txt"
        )

        boxes = load_boxes(
            label_path
        )

        if not boxes:
            continue

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        # If more than one plate exists,
        # choose the largest GT plate.
        height, width = (
            image.shape[:2]
        )

        boxes = sorted(
            boxes,
            key=lambda box:
                (
                    box[2]
                    * width
                    * box[3]
                    * height
                ),
            reverse=True,
        )

        crop, x1, y1, x2, y2 = (
            crop_with_padding(
                image,
                boxes[0],
            )
        )

        metrics = quality_metrics(
            crop
        )

        if metrics is None:
            continue

        # Remove extremely tiny regions.
        if (
            metrics["width"] < 60
            or metrics["height"] < 18
        ):
            continue

        source_id = (
            source_id_from_name(
                image_path.name
            )
        )

        candidates.append({
            "split":
                split,

            "source_id":
                source_id,

            "image_path":
                image_path,

            "label_path":
                label_path,

            "crop":
                crop,

            "crop_width":
                metrics[
                    "width"
                ],

            "crop_height":
                metrics[
                    "height"
                ],

            "sharpness":
                metrics[
                    "sharpness"
                ],

            "contrast":
                metrics[
                    "contrast"
                ],

            "quality_score":
                metrics[
                    "quality_score"
                ],
        })

    return candidates


def unique_source_best(
    candidates,
):

    best = {}

    for candidate in candidates:

        source_id = candidate[
            "source_id"
        ]

        previous = best.get(
            source_id
        )

        if (
            previous is None
            or candidate[
                "quality_score"
            ]
            >
            previous[
                "quality_score"
            ]
        ):

            best[
                source_id
            ] = candidate

    return sorted(
        best.values(),
        key=lambda item:
            item[
                "quality_score"
            ],
        reverse=True,
    )


def save_selected(
    selected,
    benchmark_split,
):

    destination = (
        DATASET_ROOT
        / benchmark_split
        / "images"
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for index, candidate in enumerate(
        selected,
        start=1,
    ):

        sample_id = (
            f"M20D_"
            f"{benchmark_split.upper()}_"
            f"{index:03d}"
        )

        filename = (
            f"{sample_id}.jpg"
        )

        output_path = (
            destination
            / filename
        )

        cv2.imwrite(
            str(output_path),
            candidate[
                "crop"
            ],
        )

        rows.append({
            "sample_id":
                sample_id,

            "benchmark_split":
                benchmark_split,

            "npds_split":
                candidate[
                    "split"
                ],

            "source_id":
                candidate[
                    "source_id"
                ],

            "crop_file":
                filename,

            "crop_width":
                candidate[
                    "crop_width"
                ],

            "crop_height":
                candidate[
                    "crop_height"
                ],

            "sharpness":
                round(
                    candidate[
                        "sharpness"
                    ],
                    3,
                ),

            "contrast":
                round(
                    candidate[
                        "contrast"
                    ],
                    3,
                ),

            "quality_score":
                round(
                    candidate[
                        "quality_score"
                    ],
                    6,
                ),

            "source_image":
                str(
                    candidate[
                        "image_path"
                    ]
                ),
        })

    return rows


def build_contact_sheet(
    rows,
    benchmark_split,
):

    image_dir = (
        DATASET_ROOT
        / benchmark_split
        / "images"
    )

    thumbs = []

    for row in rows:

        image = cv2.imread(
            str(
                image_dir
                / row[
                    "crop_file"
                ]
            )
        )

        if image is None:
            continue

        # Enlarge OCR crops for visual review.
        target_width = 320

        scale = max(
            1.0,
            target_width
            / image.shape[1],
        )

        resized = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

        canvas = np.zeros(
            (
                resized.shape[0]
                + 40,

                max(
                    resized.shape[1],
                    320,
                ),

                3,
            ),
            dtype=np.uint8,
        )

        canvas[
            40:
            40 + resized.shape[0],

            0:
            resized.shape[1],
        ] = resized

        cv2.putText(
            canvas,
            row["sample_id"],
            (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        thumbs.append(
            canvas
        )

    if not thumbs:
        return None

    columns = 3

    cell_width = max(
        thumb.shape[1]
        for thumb in thumbs
    )

    cell_height = max(
        thumb.shape[0]
        for thumb in thumbs
    )

    row_count = math.ceil(
        len(thumbs)
        / columns
    )

    sheet = np.zeros(
        (
            row_count
            * cell_height,

            columns
            * cell_width,

            3,
        ),
        dtype=np.uint8,
    )

    for index, thumb in enumerate(
        thumbs
    ):

        row_index = (
            index // columns
        )

        column_index = (
            index % columns
        )

        y = (
            row_index
            * cell_height
        )

        x = (
            column_index
            * cell_width
        )

        sheet[
            y:y + thumb.shape[0],
            x:x + thumb.shape[1],
        ] = thumb

    output_path = (
        OUTPUT_DIR
        / (
            f"M20D_"
            f"{benchmark_split}_"
            f"contact_sheet.jpg"
        )
    )

    cv2.imwrite(
        str(output_path),
        sheet,
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("M20D — OCR BENCHMARK CROP SELECTION")
    print("=" * 72)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if DATASET_ROOT.exists():

        shutil.rmtree(
            DATASET_ROOT
        )

    print(
        "\nScanning NPDS validation..."
    )

    val_candidates = (
        collect_candidates(
            "val"
        )
    )

    val_unique = (
        unique_source_best(
            val_candidates
        )
    )

    print(
        "Usable unique val sources:",
        len(val_unique),
    )

    print(
        "\nScanning NPDS test..."
    )

    test_candidates = (
        collect_candidates(
            "test"
        )
    )

    test_unique = (
        unique_source_best(
            test_candidates
        )
    )

    print(
        "Usable unique test sources:",
        len(test_unique),
    )

    if len(val_unique) < DEV_COUNT:

        raise RuntimeError(
            "Not enough validation candidates."
        )

    if len(test_unique) < TEST_COUNT:

        raise RuntimeError(
            "Not enough test candidates."
        )


    # ========================================================
    # FIX THE OCR SPLITS BEFORE RUNNING ANY OCR
    # ========================================================

    dev_selected = (
        val_unique[
            :DEV_COUNT
        ]
    )

    test_selected = (
        test_unique[
            :TEST_COUNT
        ]
    )


    dev_rows = save_selected(
        dev_selected,
        "dev",
    )

    test_rows = save_selected(
        test_selected,
        "test",
    )

    all_rows = (
        dev_rows
        + test_rows
    )


    # ========================================================
    # MANIFEST
    # ========================================================

    manifest_path = (
        OUTPUT_DIR
        / "M20D_candidate_manifest.csv"
    )

    fieldnames = [
        "sample_id",
        "benchmark_split",
        "npds_split",
        "source_id",
        "crop_file",
        "crop_width",
        "crop_height",
        "sharpness",
        "contrast",
        "quality_score",
        "source_image",
    ]

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            all_rows
        )


    dev_sheet = (
        build_contact_sheet(
            dev_rows,
            "dev",
        )
    )

    test_sheet = (
        build_contact_sheet(
            test_rows,
            "test",
        )
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = f"""M20D — OCR BENCHMARK CROP SELECTION
============================================================

Purpose
-------
Create a manually transcribed OCR benchmark using
ground-truth NPDS plate boxes.

No OCR predictions were used for selection.

Source separation
-----------------
OCR DEV:
NPDS leakage-safe validation split

OCR TEST:
NPDS leakage-safe test split

DEV samples:
{len(dev_rows)}

TEST samples:
{len(test_rows)}

Total:
{len(all_rows)}

Source-ID rule
--------------
At most one selected image per original source ID.

This prevents multiple Roboflow-derived versions of the
same source image from appearing multiple times in the
OCR benchmark.

Selection criteria
------------------
Ground-truth plate crop
Minimum crop width: 60 px
Minimum crop height: 18 px
Quality ranking combines:
- crop size
- Laplacian sharpness
- grayscale contrast

Evaluation rule
---------------
DEV may be used for OCR method development.

TEST must remain untouched by OCR-method tuning and should
only be evaluated after the OCR pipeline is finalized.
"""

    summary_path = (
        OUTPUT_DIR
        / "M20D_summary.txt"
    )

    summary_path.write_text(
        summary,
        encoding="utf-8",
    )


    print("\n" + "=" * 72)
    print("M20D COMPLETE")
    print("=" * 72)

    print(
        f"\nDEV samples  : "
        f"{len(dev_rows)}"
    )

    print(
        f"TEST samples : "
        f"{len(test_rows)}"
    )

    print(
        f"Total        : "
        f"{len(all_rows)}"
    )

    print("\nManifest:")
    print(manifest_path)

    print("\nDEV contact sheet:")
    print(dev_sheet)

    print("\nTEST contact sheet:")
    print(test_sheet)

    print("\nSummary:")
    print(summary_path)


if __name__ == "__main__":
    main()