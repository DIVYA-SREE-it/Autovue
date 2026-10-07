from pathlib import Path
from collections import defaultdict
import csv
import statistics
import time

import cv2
import paddle
from paddleocr import PaddleOCR

from m21_paddleocr_dev import (
    extract_prediction,
)

from m22a_plate_aware_parser import (
    clean_text,
    extract_indian_plate,
)

from m23b_test_evaluation import (
    rotate_image,
    removed_count,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRACK_CSV = (
    ROOT
    / "outputs"
    / "M09_ocr_video_tracking"
    / "track_summary.csv"
)

CROP_DIR = (
    ROOT
    / "outputs"
    / "M09_ocr_video_tracking"
    / "best_plate_crops"
)

GT_CSV = (
    ROOT
    / "outputs"
    / "M20_ocr_ground_truth"
    / "track_ground_truth.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "M24_temporal_ocr"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FRAME_CSV = (
    OUTPUT_DIR
    / "M24A_frame_predictions.csv"
)

CONSENSUS_CSV = (
    OUTPUT_DIR
    / "M24A_track_consensus.csv"
)

EVAL_CSV = (
    OUTPUT_DIR
    / "M24A_track_evaluation.csv"
)

SUMMARY_TXT = (
    OUTPUT_DIR
    / "M24A_temporal_pilot_summary.txt"
)


# ============================================================
# HELPERS
# ============================================================

def synchronize_gpu():

    if paddle.device.is_compiled_with_cuda():

        try:
            paddle.device.synchronize()

        except Exception:
            pass


def frozen_ocr(
    ocr,
    image,
):

    """
    Apply EXACTLY the frozen static OCR strategy:

    0-degree PaddleOCR
    -> Indian plate parser
    -> if unresolved:
       90 / 180 / 270
    -> same parser
    -> frozen candidate ranking

    Returns:
        final_output
        valid_plate
        selected_rotation
        confidence
        correction_cost
        total_latency_ms
    """

    # --------------------------------------------------------
    # 0 DEGREE
    # --------------------------------------------------------

    synchronize_gpu()

    start = time.perf_counter()

    (
        raw_0,
        confidence_0,
        regions_0,
    ) = extract_prediction(
        ocr,
        image,
    )

    synchronize_gpu()

    latency_0 = (
        time.perf_counter()
        - start
    ) * 1000.0


    raw_0 = clean_text(
        raw_0
    )


    (
        parsed_0,
        correction_0,
        parser_info_0,
    ) = extract_indian_plate(
        raw_0
    )


    if parsed_0:

        return {
            "raw_0":
                raw_0,

            "final_output":
                parsed_0,

            "valid_plate":
                parsed_0,

            "selected_rotation":
                0,

            "confidence":
                confidence_0,

            "correction_cost":
                correction_0,

            "removed_characters":
                removed_count(
                    parser_info_0
                ),

            "latency_ms":
                latency_0,

            "orientation_used":
                False,
        }


    # --------------------------------------------------------
    # ORIENTATION RESCUE
    # --------------------------------------------------------

    valid_candidates = []

    total_latency = (
        latency_0
    )


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

        start = time.perf_counter()


        (
            raw_prediction,
            confidence,
            regions,
        ) = extract_prediction(
            ocr,
            rotated,
        )


        synchronize_gpu()


        latency = (
            time.perf_counter()
            - start
        ) * 1000.0


        total_latency += latency


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


        if not parsed_candidate:
            continue


        valid_candidates.append({
            "plate":
                parsed_candidate,

            "rotation":
                angle,

            "confidence":
                confidence,

            "correction_cost":
                correction_cost,

            "removed_characters":
                removed_count(
                    parser_info
                ),
        })


    if valid_candidates:

        valid_candidates.sort(
            key=lambda item: (
                item[
                    "correction_cost"
                ],

                item[
                    "removed_characters"
                ],

                -item[
                    "confidence"
                ],

                item[
                    "rotation"
                ],
            )
        )


        best = valid_candidates[0]


        return {
            "raw_0":
                raw_0,

            "final_output":
                best[
                    "plate"
                ],

            "valid_plate":
                best[
                    "plate"
                ],

            "selected_rotation":
                best[
                    "rotation"
                ],

            "confidence":
                best[
                    "confidence"
                ],

            "correction_cost":
                best[
                    "correction_cost"
                ],

            "removed_characters":
                best[
                    "removed_characters"
                ],

            "latency_ms":
                total_latency,

            "orientation_used":
                True,
        }


    # --------------------------------------------------------
    # NO VALID PLATE
    #
    # Preserve raw text as diagnostic output.
    # It does NOT participate in temporal plate voting.
    # --------------------------------------------------------

    return {
        "raw_0":
            raw_0,

        "final_output":
            raw_0,

        "valid_plate":
            "",

        "selected_rotation":
            0,

        "confidence":
            confidence_0,

        "correction_cost":
            -1,

        "removed_characters":
            -1,

        "latency_ms":
            total_latency,

        "orientation_used":
            True,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "M24A — ROAD-TRACK TEMPORAL OCR PILOT"
    )
    print("=" * 72)


    # ========================================================
    # LOAD MANUAL TRACK GT
    # ========================================================

    with open(
        GT_CSV,
        newline="",
        encoding="utf-8",
    ) as file:

        gt_rows = list(
            csv.DictReader(file)
        )


    verified_gt = {
        row["track_id"]:
            clean_text(
                row["ground_truth"]
            )

        for row in gt_rows

        if row[
            "status"
        ] == "VERIFIED"
    }


    print(
        "\nVerified tracks:",
        len(
            verified_gt
        ),
    )


    print(
        "Track IDs:",
        sorted(
            verified_gt.keys(),
            key=int,
        ),
    )


    unique_registrations = sorted(
        set(
            verified_gt.values()
        )
    )


    print(
        "Unique registrations:",
        len(
            unique_registrations
        ),
    )


    for plate in unique_registrations:

        print(
            "  ",
            plate,
        )


    if not verified_gt:

        raise RuntimeError(
            "No VERIFIED track ground truth found."
        )


    # ========================================================
    # LOAD M09 CANDIDATE CROPS
    # ========================================================

    with open(
        TRACK_CSV,
        newline="",
        encoding="utf-8",
    ) as file:

        all_track_rows = list(
            csv.DictReader(file)
        )


    evaluation_rows = [
        row
        for row in all_track_rows
        if row[
            "track_id"
        ] in verified_gt
    ]


    print(
        "\nCandidate crops:",
        len(
            evaluation_rows
        ),
    )


    per_track_counts = defaultdict(
        int
    )


    for row in evaluation_rows:

        per_track_counts[
            row["track_id"]
        ] += 1


    for track_id in sorted(
        verified_gt,
        key=int,
    ):

        print(
            f"Track {track_id}: "
            f"{per_track_counts[track_id]} crops"
        )


    # ========================================================
    # LOAD FROZEN PADDLE OCR
    # ========================================================

    print(
        "\nLoading frozen PaddleOCR..."
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

    first_path = (
        CROP_DIR
        / evaluation_rows[0][
            "filename"
        ]
    )


    warmup_image = cv2.imread(
        str(
            first_path
        )
    )


    print(
        "Running GPU warmup..."
    )


    for _ in range(3):

        ocr.predict(
            warmup_image
        )


    synchronize_gpu()


    # ========================================================
    # OCR EVERY CANDIDATE FRAME
    # ========================================================

    frame_predictions = []

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


    total_crops = len(
        evaluation_rows
    )


    for index, row in enumerate(
        evaluation_rows,
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


        rank = int(
            row[
                "rank"
            ]
        )


        image_path = (
            CROP_DIR
            / row[
                "filename"
            ]
        )


        image = cv2.imread(
            str(
                image_path
            )
        )


        if image is None:

            raise RuntimeError(
                f"Could not read crop: "
                f"{image_path}"
            )


        result = frozen_ocr(
            ocr,
            image,
        )


        valid_plate = (
            result[
                "valid_plate"
            ]
        )


        quality_score = float(
            row[
                "quality_score"
            ]
        )


        frame_predictions.append({
            "track_id":
                track_id,

            "rank":
                rank,

            "frame":
                frame,

            "filename":
                row[
                    "filename"
                ],

            "quality_score":
                quality_score,

            "raw_0":
                result[
                    "raw_0"
                ],

            "final_output":
                result[
                    "final_output"
                ],

            "valid_plate":
                valid_plate,

            "selected_rotation":
                result[
                    "selected_rotation"
                ],

            "ocr_confidence":
                round(
                    result[
                        "confidence"
                    ],
                    6,
                ),

            "correction_cost":
                result[
                    "correction_cost"
                ],

            "latency_ms":
                round(
                    result[
                        "latency_ms"
                    ],
                    3,
                ),
        })


        # ----------------------------------------------------
        # TEMPORAL VOTING
        #
        # Only parser-validated registrations vote.
        #
        # Distinct FRAME support matters.
        # ----------------------------------------------------

        if valid_plate:

            vote = votes[
                track_id
            ][
                valid_plate
            ]


            vote[
                "frames"
            ].add(
                frame
            )


            vote[
                "confidences"
            ].append(
                result[
                    "confidence"
                ]
            )


            vote[
                "correction_costs"
            ].append(
                result[
                    "correction_cost"
                ]
            )


            vote[
                "quality_scores"
            ].append(
                quality_score
            )


        print(
            f"[{index:02d}/"
            f"{total_crops:02d}] "
            f"track={track_id} "
            f"frame={frame} "
            f"rank={rank} | "
            f"OUT="
            f"{result['final_output'] or 'NONE'} | "
            f"VALID="
            f"{valid_plate or '-'} | "
            f"ROT="
            f"{result['selected_rotation']}"
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
                frame_predictions[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            frame_predictions
        )


    # ========================================================
    # TEMPORAL CONSENSUS
    # ========================================================

    consensus_rows = []


    for track_id in sorted(
        verified_gt,
        key=int,
    ):

        candidates = []


        for plate, data in (
            votes[
                track_id
            ].items()
        ):

            frame_support = len(
                data[
                    "frames"
                ]
            )


            mean_confidence = (
                statistics.mean(
                    data[
                        "confidences"
                    ]
                )
            )


            mean_correction = (
                statistics.mean(
                    data[
                        "correction_costs"
                    ]
                )
            )


            mean_quality = (
                statistics.mean(
                    data[
                        "quality_scores"
                    ]
                )
            )


            candidates.append({
                "plate":
                    plate,

                "frame_support":
                    frame_support,

                "mean_confidence":
                    mean_confidence,

                "mean_correction":
                    mean_correction,

                "mean_quality":
                    mean_quality,
            })


        if candidates:

            # ------------------------------------------------
            # Ground truth is NOT used here.
            #
            # Priority:
            # 1. more distinct-frame support
            # 2. higher OCR confidence
            # 3. lower grammar correction
            # 4. higher crop quality
            # ------------------------------------------------

            candidates.sort(
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


            best = candidates[0]


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


            mean_confidence = (
                best[
                    "mean_confidence"
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

            mean_confidence = 0.0

            status = (
                "REJECTED"
            )


        consensus_rows.append({
            "track_id":
                track_id,

            "candidate_frames":
                per_track_counts[
                    track_id
                ],

            "final_plate":
                final_plate,

            "status":
                status,

            "frame_support":
                frame_support,

            "mean_ocr_confidence":
                round(
                    mean_confidence,
                    6,
                ),
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
    #
    # Compare:
    #
    # 1. rank-1 single frame
    # 2. temporal consensus
    #
    # This is only a 4-track PILOT.
    # ========================================================

    frame_by_track = defaultdict(
        list
    )


    for row in frame_predictions:

        frame_by_track[
            row[
                "track_id"
            ]
        ].append(
            row
        )


    consensus_by_track = {
        row["track_id"]:
            row

        for row in consensus_rows
    }


    eval_rows = []

    rank1_correct = 0

    temporal_correct = 0


    for track_id in sorted(
        verified_gt,
        key=int,
    ):

        gt = verified_gt[
            track_id
        ]


        track_frames = sorted(
            frame_by_track[
                track_id
            ],
            key=lambda row:
                int(
                    row[
                        "rank"
                    ]
                ),
        )


        rank1_prediction = (
            clean_text(
                track_frames[0][
                    "final_output"
                ]
            )
            if track_frames
            else ""
        )


        temporal_prediction = (
            clean_text(
                consensus_by_track[
                    track_id
                ][
                    "final_plate"
                ]
            )
        )


        rank1_match = (
            rank1_prediction
            == gt
        )


        temporal_match = (
            temporal_prediction
            == gt
        )


        if rank1_match:

            rank1_correct += 1


        if temporal_match:

            temporal_correct += 1


        eval_rows.append({
            "track_id":
                track_id,

            "ground_truth":
                gt,

            "rank1_prediction":
                rank1_prediction,

            "rank1_exact":
                rank1_match,

            "temporal_prediction":
                temporal_prediction,

            "temporal_status":
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

            "temporal_exact":
                temporal_match,
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
                eval_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            eval_rows
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = f"""M24A — ROAD-TRACK TEMPORAL OCR PILOT
============================================================

Purpose
-------
Evaluate whether multiple OCR observations from the same
tracked road vehicle provide more reliable plate
recognition than using only the highest-ranked single crop.

This is a PILOT experiment.

Ground truth
------------
Manually VERIFIED tracks:
{len(verified_gt)}

Unique registration strings:
{len(unique_registrations)}

Verified tracks:
{", ".join(sorted(verified_gt, key=int))}

This small set is NOT sufficient for a general road OCR
accuracy claim.

OCR method
----------
The frozen M23 static OCR pipeline was used independently
on every crop:

PaddleOCR
+
Indian plate parser
+
adaptive orientation rescue

No static OCR TEST images were reused.

Temporal voting
---------------
Only parser-validated Indian registrations participate in
temporal voting.

Votes are based on distinct video frames.

Candidate selection priority:

1. more distinct-frame support
2. higher mean OCR confidence
3. lower mean grammar correction cost
4. higher mean crop quality

Status rule:

>= 2 supporting frames
    VERIFIED

1 supporting frame
    NEEDS_REVIEW

0 valid frames
    REJECTED

Pilot comparison
----------------
Rank-1 single-frame exact matches:

{rank1_correct}/{len(verified_gt)}

Temporal-consensus exact matches:

{temporal_correct}/{len(verified_gt)}

Important
---------
Do NOT convert these counts into a general road-video OCR
accuracy percentage.

Three tracks share registration AP31AE7144, so the four
tracks represent only two unique plate strings.

This experiment is intended to study temporal behavior,
not estimate deployment accuracy.
"""


    SUMMARY_TXT.write_text(
        summary,
        encoding="utf-8",
    )


    # ========================================================
    # PRINT
    # ========================================================

    print("\n" + "=" * 72)
    print("M24A COMPLETE")
    print("=" * 72)


    for row in eval_rows:

        print(
            f"Track "
            f"{row['track_id']} | "
            f"GT="
            f"{row['ground_truth']} | "
            f"RANK1="
            f"{row['rank1_prediction'] or 'NONE'} "
            f"({'MATCH' if row['rank1_exact'] else 'MISS'}) | "
            f"TEMP="
            f"{row['temporal_prediction'] or 'NONE'} "
            f"[{row['temporal_status']}] "
            f"support="
            f"{row['frame_support']} "
            f"({'MATCH' if row['temporal_exact'] else 'MISS'})"
        )


    print(
        "\nRank-1 exact matches:",
        f"{rank1_correct}/"
        f"{len(verified_gt)}",
    )


    print(
        "Temporal exact matches:",
        f"{temporal_correct}/"
        f"{len(verified_gt)}",
    )


    print(
        "\nFrame predictions:"
    )

    print(
        FRAME_CSV
    )


    print(
        "\nConsensus:"
    )

    print(
        CONSENSUS_CSV
    )


    print(
        "\nEvaluation:"
    )

    print(
        EVAL_CSV
    )


    print(
        "\nSummary:"
    )

    print(
        SUMMARY_TXT
    )


if __name__ == "__main__":
    main()