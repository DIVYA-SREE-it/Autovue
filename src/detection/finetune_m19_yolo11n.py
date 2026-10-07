from pathlib import Path
import csv
import json

from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SOURCE_MODEL = (
    ROOT
    / "experiments"
    / "detector_benchmark"
    / "yolo11n"
    / "weights"
    / "best.pt"
)

DATASET_YAML = (
    ROOT
    / "data"
    / "processed"
    / "m19_mixed_train"
    / "dataset.yaml"
)

PROJECT_DIR = (
    ROOT
    / "experiments"
    / "domain_adaptation"
)

RUN_NAME = (
    "yolo11n_m19_safe"
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
# TRAINING CONFIGURATION
# ============================================================

EPOCHS = 15
IMAGE_SIZE = 640
BATCH = 8

LEARNING_RATE = 0.0003

SEED = 42

DEVICE = 0
WORKERS = 4

NMS_IOU = 0.70

VALIDATION_CONF = 0.001


# ============================================================
# HELPERS
# ============================================================

def metric_value(
    value,
):

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("M19F — YOLO11N DOMAIN-ADAPTATION FINE-TUNING")
    print("=" * 72)

    print("\nSource model:")
    print(SOURCE_MODEL)

    print("\nDataset:")
    print(DATASET_YAML)

    print("\nTraining configuration:")
    print(f"Epochs       : {EPOCHS}")
    print(f"Image size   : {IMAGE_SIZE}")
    print(f"Batch        : {BATCH}")
    print(f"Initial LR   : {LEARNING_RATE}")
    print(f"Seed         : {SEED}")
    print(f"Device       : CUDA:{DEVICE}")

    print(
        "\nImportant: NPDS test split is NOT used."
    )

    if not SOURCE_MODEL.exists():

        raise FileNotFoundError(
            SOURCE_MODEL
        )

    if not DATASET_YAML.exists():

        raise FileNotFoundError(
            DATASET_YAML
        )


    # ========================================================
    # LOAD EXISTING CUSTOM MODEL
    # ========================================================

    model = YOLO(
        str(SOURCE_MODEL)
    )


    # ========================================================
    # FINE-TUNE
    # ========================================================

    train_results = model.train(

        data=str(
            DATASET_YAML
        ),

        epochs=EPOCHS,

        imgsz=IMAGE_SIZE,

        batch=BATCH,

        device=DEVICE,

        workers=WORKERS,

        seed=SEED,

        deterministic=True,

        patience=5,

        optimizer="AdamW",
        
        lr0=LEARNING_RATE,

        lrf=0.1,

        warmup_epochs=1.0,

        amp=True,

        project=str(
            PROJECT_DIR
        ),

        name=RUN_NAME,

        exist_ok=True,

        verbose=True,
    )


    # ========================================================
    # BEST MODEL
    # ========================================================

    run_dir = (
        PROJECT_DIR
        / RUN_NAME
    )

    best_model_path = (
        run_dir
        / "weights"
        / "best.pt"
    )

    if not best_model_path.exists():

        raise FileNotFoundError(
            f"Best model not found: "
            f"{best_model_path}"
        )


    print("\n" + "=" * 72)
    print("FINE-TUNING COMPLETE")
    print("=" * 72)

    print("\nBest weights:")
    print(best_model_path)


    # ========================================================
    # ORIGINAL NPDS VALIDATION CHECK
    # ========================================================

    print("\n" + "=" * 72)
    print("NPDS VALIDATION — ADAPTED MODEL")
    print("=" * 72)

    adapted_model = YOLO(
        str(best_model_path)
    )

    metrics = adapted_model.val(

        data=str(
            DATASET_YAML
        ),

        split="val",

        imgsz=IMAGE_SIZE,

        batch=BATCH,

        conf=VALIDATION_CONF,

        iou=NMS_IOU,

        device=DEVICE,

        workers=WORKERS,

        plots=False,

        verbose=True,
    )


    precision = metric_value(
        metrics.box.mp
    )

    recall = metric_value(
        metrics.box.mr
    )

    map50 = metric_value(
        metrics.box.map50
    )

    map50_95 = metric_value(
        metrics.box.map
    )

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision is not None
            and recall is not None
            and precision + recall > 0
        )
        else 0.0
    )


    # ========================================================
    # PRE-ADAPTATION BASELINE
    # ========================================================

    baseline = {
        "precision":
            0.9654,

        "recall":
            0.9672,

        "f1":
            0.9663,

        "map50":
            0.9835,

        "map50_95":
            0.7109,
    }


    adapted = {
        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "map50":
            map50,

        "map50_95":
            map50_95,
    }


    # ========================================================
    # COMPARISON CSV
    # ========================================================

    comparison_path = (
        OUTPUT_DIR
        / "M19F_npds_validation_comparison.csv"
    )

    rows = []

    for metric_name in [
        "precision",
        "recall",
        "f1",
        "map50",
        "map50_95",
    ]:

        before = baseline[
            metric_name
        ]

        after = adapted[
            metric_name
        ]

        rows.append({
            "metric":
                metric_name,

            "before":
                round(
                    before,
                    6,
                ),

            "after":
                round(
                    after,
                    6,
                ),

            "delta":
                round(
                    after - before,
                    6,
                ),
        })


    with open(
        comparison_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "metric",
                "before",
                "after",
                "delta",
            ],
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


    # ========================================================
    # JSON RESULT
    # ========================================================

    result_json = {
        "training": {
            "source_model":
                str(
                    SOURCE_MODEL
                ),

            "dataset":
                str(
                    DATASET_YAML
                ),

            "epochs":
                EPOCHS,

            "imgsz":
                IMAGE_SIZE,

            "batch":
                BATCH,

            "lr0":
                LEARNING_RATE,

            "seed":
                SEED,

            "best_model":
                str(
                    best_model_path
                ),
        },

        "npds_validation_before":
            baseline,

        "npds_validation_after":
            adapted,
    }


    json_path = (
        OUTPUT_DIR
        / "M19F_results.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result_json,
            file,
            indent=2,
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = f"""M19F — YOLO11N DOMAIN ADAPTATION
============================================================

Training
--------
Source model:
{SOURCE_MODEL}

Mixed training instances:
2740

NPDS samples:
2500

Road adaptation instances:
240

Unique road images:
12

Epochs:
{EPOCHS}

Image size:
{IMAGE_SIZE}

Batch:
{BATCH}

Initial learning rate:
{LEARNING_RATE}

Seed:
{SEED}

NPDS validation — before adaptation
-----------------------------------
Precision:
{baseline["precision"]:.4f}

Recall:
{baseline["recall"]:.4f}

F1:
{baseline["f1"]:.4f}

mAP50:
{baseline["map50"]:.4f}

mAP50-95:
{baseline["map50_95"]:.4f}

NPDS validation — after adaptation
----------------------------------
Precision:
{precision:.4f}

Recall:
{recall:.4f}

F1:
{f1:.4f}

mAP50:
{map50:.4f}

mAP50-95:
{map50_95:.4f}

Change
------
Precision:
{precision - baseline["precision"]:+.4f}

Recall:
{recall - baseline["recall"]:+.4f}

F1:
{f1 - baseline["f1"]:+.4f}

mAP50:
{map50 - baseline["map50"]:+.4f}

mAP50-95:
{map50_95 - baseline["map50_95"]:+.4f}

Methodology
-----------
The NPDS test split was NOT used.

traffic_baseline.mp4:
Used for real-road adaptation.

traffic_ocr.mp4:
Not used during training.
Reserved for road-domain validation.

Next
----
Evaluate the adapted model on the same sampled
traffic_ocr frames and compare real-road precision,
recall and F1 against the M18 baseline.
"""

    summary_path = (
        OUTPUT_DIR
        / "M19F_summary.txt"
    )

    summary_path.write_text(
        summary,
        encoding="utf-8",
    )


    # ========================================================
    # PRINT FINAL
    # ========================================================

    print("\n" + "=" * 72)
    print("M19F NPDS VALIDATION RESULT")
    print("=" * 72)

    print(
        f"\nPrecision : "
        f"{precision:.4f}"
    )

    print(
        f"Recall    : "
        f"{recall:.4f}"
    )

    print(
        f"F1        : "
        f"{f1:.4f}"
    )

    print(
        f"mAP50     : "
        f"{map50:.4f}"
    )

    print(
        f"mAP50-95  : "
        f"{map50_95:.4f}"
    )

    print("\nChanges from original:")

    print(
        f"P          "
        f"{precision - baseline['precision']:+.4f}"
    )

    print(
        f"R          "
        f"{recall - baseline['recall']:+.4f}"
    )

    print(
        f"F1         "
        f"{f1 - baseline['f1']:+.4f}"
    )

    print(
        f"mAP50-95   "
        f"{map50_95 - baseline['map50_95']:+.4f}"
    )

    print("\nSummary:")
    print(summary_path)

    print("\nBest adapted model:")
    print(best_model_path)


if __name__ == "__main__":
    main()