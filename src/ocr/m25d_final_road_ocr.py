from pathlib import Path
import csv
import statistics
import time

import cv2
from paddleocr import PaddleOCR

from m21_paddleocr_dev import (
    extract_prediction,
)

from m22a_plate_aware_parser import (
    clean_text,
    extract_indian_plate,
)

from m22c_preprocessing_rescue import (
    preprocessing_variants,
)

from m23b_test_evaluation import (
    removed_count,
    rotate_image,
    synchronize_gpu,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRACK_CSV = (
    ROOT
    / "outputs"
    / "M25_final_road_eval"
    / "M25B_tracking"
    / "track_summary.csv"
)

TRACK_MAP_CSV = (
    ROOT
    / "outputs"
    / "M25_final_road_eval"
    / "M25C_track_to_cluster.csv"
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

FRAME_CSV = (
    OUTPUT_DIR
    / "M25D_frame_predictions.csv"
)

ORIENTATION_CSV = (
    OUTPUT_DIR
    / "M25D_orientation_candidates.csv"
)

PREPROCESS_CSV = (
    OUTPUT_DIR
    / "M25D_preprocessing_candidates.csv"
)

SUMMARY_TXT = (
    OUTPUT_DIR
    / "M25D_ocr_summary.txt"
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


def round_optional(value, digits=6):

    if value is None:
        return ""

    return round(
        float(value),
        digits,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "M25D — FROZEN OCR ON UNSEEN ROAD VIDEO"
    )
    print("=" * 72)


    # ========================================================
    # LOAD INPUT METADATA
    # ========================================================

    track_rows = load_csv(
        TRACK_CSV
    )

    mapping_rows = load_csv(
        TRACK_MAP_CSV
    )


    track_to_cluster = {
        row["original_track_id"]:
            row["cluster_id"]

        for row in mapping_rows
    }


    print(
        "Plate crops:",
        len(track_rows),
    )

    print(
        "Tracker IDs:",
        len(
            {
                row["track_id"]
                for row in track_rows
            }
        ),
    )

    print(
        "Conservative clusters:",
        len(
            {
                row["cluster_id"]
                for row in mapping_rows
            }
        ),
    )


    # ========================================================
    # LOAD FROZEN PADDLEOCR CONFIGURATION
    # ========================================================

    print(
        "\nLoading frozen PaddleOCR configuration..."
    )


    ocr = PaddleOCR(

        text_detection_model_name=(
            "PP-OCRv5_server_det"
        ),

        text_recognition_model_name=(
            "PP-OCRv5_server_rec"
        ),

        use_doc_orientation_classify=False,

        use_doc_unwarping=False,

        use_textline_orientation=False,

        device="gpu:0",
    )


    # ========================================================
    # WARMUP
    #
    # Same practical timing principle used previously:
    # model initialization / warmup excluded.
    # ========================================================

    first_image = cv2.imread(
        str(
            CROP_DIR
            / track_rows[0][
                "filename"
            ]
        )
    )

    if first_image is None:

        raise RuntimeError(
            "Could not load warmup image."
        )


    print(
        "Running 3 untimed warmups..."
    )


    for _ in range(3):

        extract_prediction(
            ocr,
            first_image,
        )


    synchronize_gpu()


    # ========================================================
    # OUTPUT STORAGE
    # ========================================================

    orientation_rows = []

    preprocessing_rows = []

    final_rows = []


    orientation_triggered = 0

    orientation_selected = 0

    preprocessing_triggered = 0

    preprocessing_rescued = 0


    variant_priority = {
        "upscaled": 0,
        "clahe": 1,
        "sharpened": 2,
        "threshold": 3,
    }


    # ========================================================
    # PROCESS EACH SAVED M25B CROP
    # ========================================================

    for index, row in enumerate(
        track_rows,
        start=1,
    ):

        track_id = row[
            "track_id"
        ]

        cluster_id = (
            track_to_cluster.get(
                track_id,
                track_id,
            )
        )

        frame = int(
            row[
                "frame"
            ]
        )

        rank = int(
            row[
                "rank"
            ]
        )

        filename = row[
            "filename"
        ]


        image = cv2.imread(
            str(
                CROP_DIR
                / filename
            )
        )

        if image is None:

            raise RuntimeError(
                "Could not load crop: "
                + filename
            )


        # ====================================================
        # STAGE 1
        # ORIGINAL 0-DEGREE OCR
        # ====================================================

        synchronize_gpu()

        start = (
            time.perf_counter()
        )


        (
            raw_0,
            confidence_0,
            regions_0,
        ) = extract_prediction(
            ocr,
            image,
        )


        synchronize_gpu()


        base_latency = (
            time.perf_counter()
            - start
        ) * 1000.0


        raw_0_clean = clean_text(
            raw_0
        )


        (
            parsed_0,
            correction_0,
            parser_info_0,
        ) = extract_indian_plate(
            raw_0
        )


        removed_0 = removed_count(
            parser_info_0
        )


        # Frozen M23 behavior:
        #
        # parser-valid plate if available;
        # otherwise retain normalized raw OCR text
        # as diagnostic prediction.
        static_prediction = (
            parsed_0
            if parsed_0
            else raw_0_clean
        )


        static_valid_plate = (
            parsed_0 or ""
        )


        selected_rotation = 0

        static_confidence = (
            confidence_0
        )

        static_correction = (
            correction_0
        )

        static_removed = (
            removed_0
        )

        selected_static_raw = (
            raw_0_clean
        )


        orientation_extra_latency = (
            0.0
        )


        orientation_rows.append({
            "track_id":
                track_id,

            "cluster_id":
                cluster_id,

            "frame":
                frame,

            "rank":
                rank,

            "filename":
                filename,

            "rotation":
                0,

            "raw_prediction":
                raw_0_clean,

            "parsed_candidate":
                parsed_0 or "",

            "correction_cost":
                correction_0,

            "removed_characters":
                removed_0,

            "ocr_confidence":
                round_optional(
                    confidence_0
                ),

            "text_regions":
                regions_0,

            "latency_ms":
                round(
                    base_latency,
                    3,
                ),
        })


        # ====================================================
        # STAGE 2
        # FROZEN ADAPTIVE ORIENTATION RESCUE
        # ====================================================

        if not parsed_0:

            orientation_triggered += 1


            valid_orientation_candidates = []


            for angle in [
                90,
                180,
                270,
            ]:

                rotated = rotate_image(
                    image,
                    angle,
                )


                synchronize_gpu()

                rotation_start = (
                    time.perf_counter()
                )


                (
                    raw_rotation,
                    rotation_confidence,
                    rotation_regions,
                ) = extract_prediction(
                    ocr,
                    rotated,
                )


                synchronize_gpu()


                rotation_latency = (
                    time.perf_counter()
                    - rotation_start
                ) * 1000.0


                orientation_extra_latency += (
                    rotation_latency
                )


                raw_rotation_clean = (
                    clean_text(
                        raw_rotation
                    )
                )


                (
                    parsed_rotation,
                    correction_rotation,
                    parser_info_rotation,
                ) = extract_indian_plate(
                    raw_rotation
                )


                removed_rotation = (
                    removed_count(
                        parser_info_rotation
                    )
                )


                orientation_rows.append({
                    "track_id":
                        track_id,

                    "cluster_id":
                        cluster_id,

                    "frame":
                        frame,

                    "rank":
                        rank,

                    "filename":
                        filename,

                    "rotation":
                        angle,

                    "raw_prediction":
                        raw_rotation_clean,

                    "parsed_candidate":
                        parsed_rotation or "",

                    "correction_cost":
                        correction_rotation,

                    "removed_characters":
                        removed_rotation,

                    "ocr_confidence":
                        round_optional(
                            rotation_confidence
                        ),

                    "text_regions":
                        rotation_regions,

                    "latency_ms":
                        round(
                            rotation_latency,
                            3,
                        ),
                })


                if parsed_rotation:

                    valid_orientation_candidates.append({
                        "rotation":
                            angle,

                        "plate":
                            parsed_rotation,

                        "raw":
                            raw_rotation_clean,

                        "correction_cost":
                            correction_rotation,

                        "removed":
                            removed_rotation,

                        "confidence":
                            rotation_confidence,
                    })


            # =================================================
            # SAME M23 CANDIDATE RANKING
            # =================================================

            if valid_orientation_candidates:

                valid_orientation_candidates.sort(
                    key=lambda candidate: (

                        candidate[
                            "correction_cost"
                        ],

                        candidate[
                            "removed"
                        ],

                        -candidate[
                            "confidence"
                        ],

                        candidate[
                            "rotation"
                        ],
                    )
                )


                best = (
                    valid_orientation_candidates[
                        0
                    ]
                )


                static_valid_plate = (
                    best[
                        "plate"
                    ]
                )

                static_prediction = (
                    best[
                        "plate"
                    ]
                )

                selected_static_raw = (
                    best[
                        "raw"
                    ]
                )

                selected_rotation = (
                    best[
                        "rotation"
                    ]
                )

                static_confidence = (
                    best[
                        "confidence"
                    ]
                )

                static_correction = (
                    best[
                        "correction_cost"
                    ]
                )

                static_removed = (
                    best[
                        "removed"
                    ]
                )


                orientation_selected += 1


        # ====================================================
        # STAGE 3
        # FROZEN ROAD-DOMAIN PREPROCESSING FALLBACK
        #
        # Only if M23 static/orientation stages still have
        # NO parser-valid registration.
        # ====================================================

        final_valid_plate = (
            static_valid_plate
        )

        final_prediction = (
            static_prediction
        )

        final_raw = (
            selected_static_raw
        )

        final_confidence = (
            static_confidence
        )

        final_correction = (
            static_correction
        )

        final_removed = (
            static_removed
        )

        selected_variant = (
            "baseline"
        )


        preprocessing_extra_latency = (
            0.0
        )


        if not static_valid_plate:

            preprocessing_triggered += 1


            valid_preprocessing_candidates = []


            variants = preprocessing_variants(
                image
            )


            for (
                variant_name,
                variant_image
            ) in variants.items():

                synchronize_gpu()

                preprocess_start = (
                    time.perf_counter()
                )


                (
                    raw_variant,
                    variant_confidence,
                    variant_regions,
                ) = extract_prediction(
                    ocr,
                    variant_image,
                )


                synchronize_gpu()


                variant_latency = (
                    time.perf_counter()
                    - preprocess_start
                ) * 1000.0


                preprocessing_extra_latency += (
                    variant_latency
                )


                raw_variant_clean = clean_text(
                    raw_variant
                )


                (
                    parsed_variant,
                    correction_variant,
                    parser_info_variant,
                ) = extract_indian_plate(
                    raw_variant
                )


                removed_variant = removed_count(
                    parser_info_variant
                )


                preprocessing_rows.append({
                    "track_id":
                        track_id,

                    "cluster_id":
                        cluster_id,

                    "frame":
                        frame,

                    "rank":
                        rank,

                    "filename":
                        filename,

                    "variant":
                        variant_name,

                    "raw_prediction":
                        raw_variant_clean,

                    "parsed_candidate":
                        parsed_variant or "",

                    "correction_cost":
                        correction_variant,

                    "removed_characters":
                        removed_variant,

                    "ocr_confidence":
                        round_optional(
                            variant_confidence
                        ),

                    "text_regions":
                        variant_regions,

                    "latency_ms":
                        round(
                            variant_latency,
                            3,
                        ),
                })


                if parsed_variant:

                    valid_preprocessing_candidates.append({
                        "plate":
                            parsed_variant,

                        "raw":
                            raw_variant_clean,

                        "variant":
                            variant_name,

                        "correction_cost":
                            correction_variant,

                        "removed":
                            removed_variant,

                        "confidence":
                            variant_confidence,
                    })


            # =================================================
            # SAME M24B SELECTION RULE
            # =================================================

            if valid_preprocessing_candidates:

                valid_preprocessing_candidates.sort(
                    key=lambda candidate: (

                        candidate[
                            "correction_cost"
                        ],

                        candidate[
                            "removed"
                        ],

                        -candidate[
                            "confidence"
                        ],

                        variant_priority[
                            candidate[
                                "variant"
                            ]
                        ],
                    )
                )


                best = (
                    valid_preprocessing_candidates[
                        0
                    ]
                )


                final_valid_plate = (
                    best[
                        "plate"
                    ]
                )

                final_prediction = (
                    best[
                        "plate"
                    ]
                )

                final_raw = (
                    best[
                        "raw"
                    ]
                )

                final_confidence = (
                    best[
                        "confidence"
                    ]
                )

                final_correction = (
                    best[
                        "correction_cost"
                    ]
                )

                final_removed = (
                    best[
                        "removed"
                    ]
                )

                selected_variant = (
                    best[
                        "variant"
                    ]
                )


                preprocessing_rescued += 1


        # ====================================================
        # TOTAL LATENCY
        # ====================================================

        total_latency = (
            base_latency
            + orientation_extra_latency
            + preprocessing_extra_latency
        )


        final_rows.append({
            "track_id":
                track_id,

            "cluster_id":
                cluster_id,

            "frame":
                frame,

            "rank":
                rank,

            "filename":
                filename,

            "quality_score":
                row[
                    "quality_score"
                ],

            "plate_confidence":
                row[
                    "plate_confidence"
                ],

            "raw_0deg":
                raw_0_clean,

            "parsed_0deg":
                parsed_0 or "",

            "orientation_triggered":
                not bool(
                    parsed_0
                ),

            "selected_rotation":
                selected_rotation,

            "static_valid_plate":
                static_valid_plate,

            "static_prediction":
                static_prediction,

            "preprocessing_triggered":
                not bool(
                    static_valid_plate
                ),

            "selected_variant":
                selected_variant,

            "final_raw":
                final_raw,

            "final_valid_plate":
                final_valid_plate,

            "final_prediction":
                final_prediction,

            "final_confidence":
                round_optional(
                    final_confidence
                ),

            "correction_cost":
                final_correction,

            "removed_characters":
                final_removed,

            "base_latency_ms":
                round(
                    base_latency,
                    3,
                ),

            "orientation_extra_ms":
                round(
                    orientation_extra_latency,
                    3,
                ),

            "preprocessing_extra_ms":
                round(
                    preprocessing_extra_latency,
                    3,
                ),

            "total_latency_ms":
                round(
                    total_latency,
                    3,
                ),
        })


        print(
            f"[{index:02d}/{len(track_rows)}] "
            f"track={track_id} "
            f"cluster={cluster_id} "
            f"frame={frame} "
            f"rank={rank} "
            f"rot={selected_rotation} "
            f"variant={selected_variant} "
            f"plate="
            f"{final_valid_plate or 'NONE'}"
        )


    # ========================================================
    # WRITE CSV FILES
    # ========================================================

    with open(
        FRAME_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=(
                final_rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            final_rows
        )


    with open(
        ORIENTATION_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=(
                orientation_rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            orientation_rows
        )


    preprocess_fields = [
        "track_id",
        "cluster_id",
        "frame",
        "rank",
        "filename",
        "variant",
        "raw_prediction",
        "parsed_candidate",
        "correction_cost",
        "removed_characters",
        "ocr_confidence",
        "text_regions",
        "latency_ms",
    ]


    with open(
        PREPROCESS_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=preprocess_fields,
        )

        writer.writeheader()

        writer.writerows(
            preprocessing_rows
        )


    # ========================================================
    # DESCRIPTIVE SUMMARY
    #
    # NO ACCURACY IS CALCULATED HERE.
    # NO GROUND TRUTH EXISTS IN THIS STAGE.
    # ========================================================

    valid_static = sum(
        bool(
            row[
                "static_valid_plate"
            ]
        )
        for row in final_rows
    )


    valid_final = sum(
        bool(
            row[
                "final_valid_plate"
            ]
        )
        for row in final_rows
    )


    clusters_with_candidate = len({
        row[
            "cluster_id"
        ]

        for row in final_rows

        if row[
            "final_valid_plate"
        ]
    })


    unique_candidate_strings = len({
        row[
            "final_valid_plate"
        ]

        for row in final_rows

        if row[
            "final_valid_plate"
        ]
    })


    mean_latency = statistics.mean(
        float(
            row[
                "total_latency_ms"
            ]
        )
        for row in final_rows
    )


    median_latency = statistics.median(
        float(
            row[
                "total_latency_ms"
            ]
        )
        for row in final_rows
    )


    summary = f"""M25D — FROZEN OCR ON UNSEEN ROAD VIDEO
============================================================

Purpose
-------
Apply the already-developed OCR pipeline to the frozen
M25B road crops.

Ground truth used:
NO

OCR crops:
{len(final_rows)}

Raw tracker IDs:
{len(set(row["track_id"] for row in final_rows))}

Conservative M25C clusters:
{len(set(row["cluster_id"] for row in final_rows))}

Frozen static OCR
-----------------
PaddleOCR:
PP-OCRv5_server_det
PP-OCRv5_server_rec

Document orientation classifier:
disabled

Document unwarping:
disabled

Text-line orientation:
disabled

Device:
gpu:0

Static method:
0-degree OCR
-> Indian registration parser
-> adaptive 90/180/270-degree rescue when unresolved

Orientation rescue triggered:
{orientation_triggered}

Rotated candidate selected:
{orientation_selected}

Parser-valid crops after static/orientation stage:
{valid_static}

Road fallback
-------------
Only unresolved crops were tested with the previously
developed M24B road fallback:

1. 3x bicubic upscale
2. CLAHE
3. sharpening
4. Otsu thresholding

Preprocessing fallback triggered:
{preprocessing_triggered}

Crops rescued by preprocessing:
{preprocessing_rescued}

Final parser-valid crops:
{valid_final}

Clusters containing >=1 parser-valid candidate:
{clusters_with_candidate}

Unique parser-valid candidate strings:
{unique_candidate_strings}

Timing
------
Warmup/model initialization excluded.

Mean adaptive OCR latency per crop:
{mean_latency:.2f} ms

Median adaptive OCR latency per crop:
{median_latency:.2f} ms

Important
---------
These are prediction counts, NOT accuracy.

No ground truth was used in this stage.

No OCR threshold, parser rule, orientation rule,
preprocessing variant, or candidate-ranking rule was
changed after viewing M25 predictions.

Temporal consensus and fragment corroboration are
evaluated separately in the next milestone.
"""


    SUMMARY_TXT.write_text(
        summary,
        encoding="utf-8",
    )


    print()
    print("=" * 72)
    print("M25D COMPLETE")
    print("=" * 72)

    print(
        "Static parser-valid crops:",
        valid_static,
    )

    print(
        "Final parser-valid crops:",
        valid_final,
    )

    print(
        "Preprocessing rescues:",
        preprocessing_rescued,
    )

    print(
        "Clusters with candidate:",
        clusters_with_candidate,
    )

    print()
    print(
        "Frame predictions:",
        FRAME_CSV
    )

    print(
        "Orientation candidates:",
        ORIENTATION_CSV
    )

    print(
        "Preprocessing candidates:",
        PREPROCESS_CSV
    )

    print(
        "Summary:",
        SUMMARY_TXT
    )


if __name__ == "__main__":
    main()