from pathlib import Path
from collections import defaultdict
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
    synchronize_gpu,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

M24A_FRAME_CSV = (
    ROOT
    / "outputs"
    / "M24_temporal_ocr"
    / "M24A_frame_predictions.csv"
)

M24A_CONSENSUS_CSV = (
    ROOT
    / "outputs"
    / "M24_temporal_ocr"
    / "M24A_track_consensus.csv"
)

GT_CSV = (
    ROOT
    / "outputs"
    / "M20_ocr_ground_truth"
    / "track_ground_truth.csv"
)

CROP_DIR = (
    ROOT
    / "outputs"
    / "M09_ocr_video_tracking"
    / "best_plate_crops"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "M24_temporal_ocr"
)

CANDIDATE_CSV = (
    OUTPUT_DIR
    / "M24B_preprocessing_candidates.csv"
)

FRAME_CSV = (
    OUTPUT_DIR
    / "M24B_frame_predictions.csv"
)

CONSENSUS_CSV = (
    OUTPUT_DIR
    / "M24B_track_consensus.csv"
)

EVAL_CSV = (
    OUTPUT_DIR
    / "M24B_track_evaluation.csv"
)

SUMMARY_TXT = (
    OUTPUT_DIR
    / "M24B_road_preprocessing_summary.txt"
)


# ============================================================
# HELPERS
# ============================================================

def load_rows(path):

    with open(
        path,
        newline="",
        encoding="utf-8",
    ) as file:

        return list(
            csv.DictReader(file)
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "M24B — ROAD-DOMAIN PREPROCESSING PILOT"
    )
    print("=" * 72)


    # ========================================================
    # LOAD M24A FRAME RESULTS
    # ========================================================

    frame_rows = load_rows(
        M24A_FRAME_CSV
    )

    baseline_consensus = {
        row["track_id"]: row
        for row in load_rows(
            M24A_CONSENSUS_CSV
        )
    }


    # ========================================================
    # LOAD VERIFIED TRACK GT
    # ========================================================

    gt_rows = load_rows(
        GT_CSV
    )


    verified_gt = {
        row["track_id"]:
            clean_text(
                row["ground_truth"]
            )

        for row in gt_rows

        if row["status"]
        == "VERIFIED"
    }


    print(
        "\nVerified tracks:",
        len(verified_gt),
    )

    print(
        "Candidate crops:",
        len(frame_rows),
    )


    # ========================================================
    # LOAD PADDLEOCR
    # ========================================================

    print(
        "\nLoading PaddleOCR..."
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
    # ========================================================

    first_image = cv2.imread(
        str(
            CROP_DIR
            / frame_rows[0]["filename"]
        )
    )


    print(
        "Running GPU warmup..."
    )


    for _ in range(3):

        ocr.predict(
            first_image
        )


    synchronize_gpu()


    # ========================================================
    # PREPROCESSING RESCUE
    # ========================================================

    candidate_rows = []

    final_frame_rows = []


    variant_priority = {
        "upscaled": 0,
        "clahe": 1,
        "sharpened": 2,
        "threshold": 3,
    }


    preprocessing_attempted = 0

    rescued_frames = 0

    extra_latency_total = 0.0


    for index, row in enumerate(
        frame_rows,
        start=1,
    ):

        track_id = row[
            "track_id"
        ]

        frame = int(
            row[
                "frame"
            ]
        )


        existing_plate = (
            clean_text(
                row[
                    "valid_plate"
                ]
            )
        )


        final_plate = (
            existing_plate
        )


        selected_variant = (
            "baseline"
        )


        rescue_confidence = (
            float(
                row[
                    "ocr_confidence"
                ]
            )
        )


        rescue_cost = (
            int(
                float(
                    row[
                        "correction_cost"
                    ]
                )
            )
        )


        extra_latency = 0.0


        # ----------------------------------------------------
        # Preserve already valid M24A results.
        # ----------------------------------------------------

        if not existing_plate:

            preprocessing_attempted += 1


            image = cv2.imread(
                str(
                    CROP_DIR
                    / row[
                        "filename"
                    ]
                )
            )


            if image is None:

                raise RuntimeError(
                    "Could not load "
                    + row[
                        "filename"
                    ]
                )


            valid_candidates = []


            variants = (
                preprocessing_variants(
                    image
                )
            )


            for (
                variant_name,
                variant_image
            ) in variants.items():

                synchronize_gpu()

                start = (
                    time.perf_counter()
                )


                (
                    raw_prediction,
                    confidence,
                    regions,
                ) = extract_prediction(
                    ocr,
                    variant_image,
                )


                synchronize_gpu()


                latency_ms = (
                    time.perf_counter()
                    - start
                ) * 1000.0


                extra_latency += (
                    latency_ms
                )


                raw_prediction = (
                    clean_text(
                        raw_prediction
                    )
                )


                (
                    parsed_candidate,
                    correction_cost,
                    parser_info,
                ) = extract_indian_plate(
                    raw_prediction
                )


                removed = removed_count(
                    parser_info
                )


                candidate_rows.append({
                    "track_id":
                        track_id,

                    "frame":
                        frame,

                    "rank":
                        row[
                            "rank"
                        ],

                    "variant":
                        variant_name,

                    "raw_prediction":
                        raw_prediction,

                    "parsed_candidate":
                        parsed_candidate,

                    "correction_cost":
                        correction_cost,

                    "removed_characters":
                        removed,

                    "ocr_confidence":
                        round(
                            confidence,
                            6,
                        ),

                    "latency_ms":
                        round(
                            latency_ms,
                            3,
                        ),
                })


                if parsed_candidate:

                    valid_candidates.append({
                        "plate":
                            parsed_candidate,

                        "variant":
                            variant_name,

                        "correction_cost":
                            correction_cost,

                        "removed":
                            removed,

                        "confidence":
                            confidence,
                    })


            # ------------------------------------------------
            # Select without GT.
            # ------------------------------------------------

            if valid_candidates:

                valid_candidates.sort(
                    key=lambda item: (
                        item[
                            "correction_cost"
                        ],

                        item[
                            "removed"
                        ],

                        -item[
                            "confidence"
                        ],

                        variant_priority[
                            item[
                                "variant"
                            ]
                        ],
                    )
                )


                best = (
                    valid_candidates[0]
                )


                final_plate = (
                    best[
                        "plate"
                    ]
                )


                selected_variant = (
                    best[
                        "variant"
                    ]
                )


                rescue_confidence = (
                    best[
                        "confidence"
                    ]
                )


                rescue_cost = (
                    best[
                        "correction_cost"
                    ]
                )


                rescued_frames += 1


        extra_latency_total += (
            extra_latency
        )


        final_frame_rows.append({
            "track_id":
                track_id,

            "rank":
                row[
                    "rank"
                ],

            "frame":
                frame,

            "filename":
                row[
                    "filename"
                ],

            "quality_score":
                row[
                    "quality_score"
                ],

            "m24a_plate":
                existing_plate,

            "selected_variant":
                selected_variant,

            "final_valid_plate":
                final_plate,

            "ocr_confidence":
                round(
                    rescue_confidence,
                    6,
                ),

            "correction_cost":
                rescue_cost,

            "extra_preprocessing_latency_ms":
                round(
                    extra_latency,
                    3,
                ),
        })


        print(
            f"[{index:02d}/"
            f"{len(frame_rows):02d}] "
            f"track={track_id} "
            f"frame={frame} | "
            f"M24A="
            f"{existing_plate or '-'} | "
            f"VAR="
            f"{selected_variant} | "
            f"FINAL="
            f"{final_plate or '-'}"
        )


    # ========================================================
    # SAVE FRAME RESULTS
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
                final_frame_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            final_frame_rows
        )


    with open(
        CANDIDATE_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=(
                candidate_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            candidate_rows
        )


    # ========================================================
    # TEMPORAL VOTING
    # ========================================================

    votes = defaultdict(
        lambda: defaultdict(
            lambda: {
                "frames":
                    set(),

                "confidences":
                    [],

                "correction_costs":
                    [],

                "quality_scores":
                    [],
            }
        )
    )


    candidate_count = defaultdict(
        int
    )


    for row in final_frame_rows:

        track_id = row[
            "track_id"
        ]

        candidate_count[
            track_id
        ] += 1


        plate = clean_text(
            row[
                "final_valid_plate"
            ]
        )


        if not plate:
            continue


        frame = int(
            row[
                "frame"
            ]
        )


        vote = votes[
            track_id
        ][
            plate
        ]


        vote[
            "frames"
        ].add(
            frame
        )


        vote[
            "confidences"
        ].append(
            float(
                row[
                    "ocr_confidence"
                ]
            )
        )


        vote[
            "correction_costs"
        ].append(
            float(
                row[
                    "correction_cost"
                ]
            )
        )


        vote[
            "quality_scores"
        ].append(
            float(
                row[
                    "quality_score"
                ]
            )
        )


    consensus_rows = []


    for track_id in sorted(
        verified_gt,
        key=int,
    ):

        options = []


        for plate, data in (
            votes[
                track_id
            ].items()
        ):

            options.append({
                "plate":
                    plate,

                "frame_support":
                    len(
                        data[
                            "frames"
                        ]
                    ),

                "mean_confidence":
                    statistics.mean(
                        data[
                            "confidences"
                        ]
                    ),

                "mean_correction":
                    statistics.mean(
                        data[
                            "correction_costs"
                        ]
                    ),

                "mean_quality":
                    statistics.mean(
                        data[
                            "quality_scores"
                        ]
                    ),
            })


        if options:

            options.sort(
                key=lambda item: (
                    -item[
                        "frame_support"
                    ],

                    -item[
                        "mean_confidence"
                    ],

                    item[
                        "mean_correction"
                    ],

                    -item[
                        "mean_quality"
                    ],
                )
            )


            best = options[0]


            final_plate = (
                best[
                    "plate"
                ]
            )


            frame_support = (
                best[
                    "frame_support"
                ]
            )


            if frame_support >= 2:

                status = (
                    "VERIFIED"
                )

            else:

                status = (
                    "NEEDS_REVIEW"
                )


        else:

            final_plate = ""

            frame_support = 0

            status = (
                "REJECTED"
            )


        consensus_rows.append({
            "track_id":
                track_id,

            "candidate_frames":
                candidate_count[
                    track_id
                ],

            "final_plate":
                final_plate,

            "status":
                status,

            "frame_support":
                frame_support,
        })


    with open(
        CONSENSUS_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=(
                consensus_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            consensus_rows
        )


    # ========================================================
    # EVALUATION
    # ========================================================

    consensus_by_track = {
        row[
            "track_id"
        ]:
            row

        for row in consensus_rows
    }


    evaluation_rows = []

    baseline_correct = 0

    final_correct = 0


    for track_id in sorted(
        verified_gt,
        key=int,
    ):

        gt = verified_gt[
            track_id
        ]


        baseline_prediction = clean_text(
            baseline_consensus[
                track_id
            ][
                "final_plate"
            ]
        )


        final_prediction = clean_text(
            consensus_by_track[
                track_id
            ][
                "final_plate"
            ]
        )


        baseline_match = (
            baseline_prediction
            == gt
        )


        final_match = (
            final_prediction
            == gt
        )


        if baseline_match:
            baseline_correct += 1


        if final_match:
            final_correct += 1


        evaluation_rows.append({
            "track_id":
                track_id,

            "ground_truth":
                gt,

            "m24a_prediction":
                baseline_prediction,

            "m24a_exact":
                baseline_match,

            "m24b_prediction":
                final_prediction,

            "m24b_status":
                consensus_by_track[
                    track_id
                ][
                    "status"
                ],

            "frame_support":
                consensus_by_track[
                    track_id
                ][
                    "frame_support"
                ],

            "m24b_exact":
                final_match,
        })


    with open(
        EVAL_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=(
                evaluation_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            evaluation_rows
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    mean_extra_latency = (
        extra_latency_total
        / len(
            frame_rows
        )
    )


    summary = f"""M24B — ROAD-DOMAIN PREPROCESSING PILOT
============================================================

Purpose
-------
Test whether preprocessing that was rejected on the
static OCR DEV benchmark can still help specifically on
low-quality road-video crops.

This does NOT modify the frozen static OCR pipeline.

Road pilot
----------
Verified tracks:
{len(verified_gt)}

Unique registration strings:
{len(set(verified_gt.values()))}

Candidate crops:
{len(frame_rows)}

Baseline
--------
M24A temporal exact matches:

{baseline_correct}/{len(verified_gt)}

Road preprocessing rescue
-------------------------
Only frames without a parser-valid M24A plate were tested
with:

1. 3x bicubic upscale
2. CLAHE
3. sharpening
4. Otsu thresholding

Each resulting OCR string used the SAME Indian plate
parser.

Ground truth was NOT used for candidate selection.

Frames receiving preprocessing search:

{preprocessing_attempted}/{len(frame_rows)}

Frames where preprocessing created a valid plate:

{rescued_frames}

Temporal result after road preprocessing:

{final_correct}/{len(verified_gt)}

Mean additional preprocessing latency averaged across all
road crops:

{mean_extra_latency:.2f} ms

Interpretation rule
-------------------
This is an exploratory road-domain pilot.

These four tracks are development/analysis tracks and
must not be used to claim general road-video accuracy.

Any road-specific method retained from this pilot must be
evaluated later on a NEW unseen road video.
"""


    SUMMARY_TXT.write_text(
        summary,
        encoding="utf-8",
    )


    # ========================================================
    # PRINT
    # ========================================================

    print("\n" + "=" * 72)
    print("M24B COMPLETE")
    print("=" * 72)


    for row in evaluation_rows:

        print(
            f"Track "
            f"{row['track_id']} | "
            f"GT="
            f"{row['ground_truth']} | "
            f"M24A="
            f"{row['m24a_prediction'] or 'NONE'} | "
            f"M24B="
            f"{row['m24b_prediction'] or 'NONE'} | "
            f"support="
            f"{row['frame_support']} | "
            f"{'MATCH' if row['m24b_exact'] else 'MISS'}"
        )


    print(
        "\nM24A temporal exact:",
        f"{baseline_correct}/"
        f"{len(verified_gt)}",
    )


    print(
        "M24B temporal exact:",
        f"{final_correct}/"
        f"{len(verified_gt)}",
    )


    print(
        "\nSummary:"
    )

    print(
        SUMMARY_TXT
    )


if __name__ == "__main__":
    main()