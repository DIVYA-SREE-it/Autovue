from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

BENCHMARK_DIR = (
    ROOT
    / "experiments"
    / "detector_benchmark"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "M16_detector_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# MODELS USED IN THE COMPLETED BENCHMARK
# ============================================================

MODELS = {
    "yolov8n": "YOLOv8n",
    "yolo11n": "YOLO11n",
    "yolo11s": "YOLO11s",
}


# ============================================================
# HELPERS
# ============================================================

def read_json(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def read_csv(path):

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        rows = []

        for row in reader:

            clean_row = {
                key.strip(): value
                for key, value in row.items()
            }

            rows.append(clean_row)

    return rows


def get_float(row, column):

    value = row.get(
        column,
        ""
    )

    try:
        return float(value)
    except (
        ValueError,
        TypeError,
    ):
        return None


def get_column(rows, column):

    values = []

    for row in rows:

        value = get_float(
            row,
            column,
        )

        if value is not None:
            values.append(value)

    return values


def epochs_from_rows(rows):

    epochs = []

    for index, row in enumerate(rows):

        value = get_float(
            row,
            "epoch",
        )

        if value is None:
            value = index + 1

        epochs.append(value)

    return epochs


def write_csv(
    path,
    rows,
    fields,
):

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# LOAD BENCHMARK DATA
# ============================================================

summaries = {}
histories = {}

for model_key in MODELS:

    model_dir = (
        BENCHMARK_DIR
        / model_key
    )

    summary_path = (
        model_dir
        / "benchmark_summary.json"
    )

    results_path = (
        model_dir
        / "results.csv"
    )

    if not summary_path.exists():

        raise FileNotFoundError(
            f"Missing summary: "
            f"{summary_path}"
        )

    if not results_path.exists():

        raise FileNotFoundError(
            f"Missing results CSV: "
            f"{results_path}"
        )

    summaries[model_key] = (
        read_json(
            summary_path
        )
    )

    histories[model_key] = (
        read_csv(
            results_path
        )
    )


# ============================================================
# FINAL TEST COMPARISON TABLE
# ============================================================

comparison_rows = []

for model_key, display_name in MODELS.items():

    summary = summaries[
        model_key
    ]

    comparison_rows.append({
        "model": display_name,

        "precision":
            summary["precision"],

        "recall":
            summary["recall"],

        "f1":
            summary["f1"],

        "map50":
            summary["map50"],

        "map50_95":
            summary["map50_95"],

        "inference_ms":
            summary[
                "inference_ms_per_image"
            ],

        "fps":
            summary[
                "fps_theoretical"
            ],

        "parameters":
            summary["parameters"],

        "model_size_mb":
            summary[
                "model_size_mb"
            ],

        "peak_vram_gb":
            summary[
                "peak_reserved_vram_gb"
            ],

        "training_minutes":
            summary[
                "training_minutes"
            ],

        "best_epoch":
            summary[
                "best_epoch"
            ],
    })


comparison_fields = [
    "model",
    "precision",
    "recall",
    "f1",
    "map50",
    "map50_95",
    "inference_ms",
    "fps",
    "parameters",
    "model_size_mb",
    "peak_vram_gb",
    "training_minutes",
    "best_epoch",
]

write_csv(
    OUTPUT_DIR
    / "detector_comparison.csv",

    comparison_rows,
    comparison_fields,
)


# ============================================================
# FINAL TRAIN / VALIDATION LOSS TABLE
# ============================================================

loss_rows = []

for model_key, display_name in MODELS.items():

    rows = histories[
        model_key
    ]

    last = rows[-1]

    train_box = get_float(
        last,
        "train/box_loss",
    )

    val_box = get_float(
        last,
        "val/box_loss",
    )

    train_cls = get_float(
        last,
        "train/cls_loss",
    )

    val_cls = get_float(
        last,
        "val/cls_loss",
    )

    train_dfl = get_float(
        last,
        "train/dfl_loss",
    )

    val_dfl = get_float(
        last,
        "val/dfl_loss",
    )

    loss_rows.append({
        "model": display_name,

        "train_box_loss":
            train_box,

        "val_box_loss":
            val_box,

        "box_gap":
            (
                val_box
                - train_box
            ),

        "train_cls_loss":
            train_cls,

        "val_cls_loss":
            val_cls,

        "cls_gap":
            (
                val_cls
                - train_cls
            ),

        "train_dfl_loss":
            train_dfl,

        "val_dfl_loss":
            val_dfl,

        "dfl_gap":
            (
                val_dfl
                - train_dfl
            ),
    })


loss_fields = [
    "model",
    "train_box_loss",
    "val_box_loss",
    "box_gap",
    "train_cls_loss",
    "val_cls_loss",
    "cls_gap",
    "train_dfl_loss",
    "val_dfl_loss",
    "dfl_gap",
]

write_csv(
    OUTPUT_DIR
    / "final_loss_comparison.csv",

    loss_rows,
    loss_fields,
)


# ============================================================
# LOSS CURVES
# ============================================================

LOSS_CONFIGS = {
    "box": (
        "train/box_loss",
        "val/box_loss",
        "Box Loss",
        "box_loss_curves.png",
    ),

    "cls": (
        "train/cls_loss",
        "val/cls_loss",
        "Classification Loss",
        "cls_loss_curves.png",
    ),

    "dfl": (
        "train/dfl_loss",
        "val/dfl_loss",
        "Distribution Focal Loss",
        "dfl_loss_curves.png",
    ),
}


for (
    loss_name,
    config
) in LOSS_CONFIGS.items():

    (
        train_column,
        val_column,
        title,
        filename,
    ) = config

    plt.figure(
        figsize=(10, 6)
    )

    for (
        model_key,
        display_name
    ) in MODELS.items():

        rows = histories[
            model_key
        ]

        epochs = epochs_from_rows(
            rows
        )

        train_values = (
            get_column(
                rows,
                train_column,
            )
        )

        val_values = (
            get_column(
                rows,
                val_column,
            )
        )

        line, = plt.plot(
            epochs[
                :len(
                    train_values
                )
            ],
            train_values,
            label=(
                f"{display_name} Train"
            ),
            linewidth=2,
        )

        plt.plot(
            epochs[
                :len(
                    val_values
                )
            ],
            val_values,
            label=(
                f"{display_name} Val"
            ),
            linewidth=2,
            linestyle="--",
            color=line.get_color(),
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        f"Training vs Validation "
        f"{title}"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / filename,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# mAP CURVES
# ============================================================

MAP_CONFIGS = {
    "map50": (
        "metrics/mAP50(B)",
        "mAP@50",
        "map50_curves.png",
    ),

    "map50_95": (
        "metrics/mAP50-95(B)",
        "mAP@50-95",
        "map50_95_curves.png",
    ),
}


for (
    metric_name,
    config
) in MAP_CONFIGS.items():

    (
        column,
        title,
        filename,
    ) = config

    plt.figure(
        figsize=(10, 6)
    )

    for (
        model_key,
        display_name
    ) in MODELS.items():

        rows = histories[
            model_key
        ]

        epochs = epochs_from_rows(
            rows
        )

        values = get_column(
            rows,
            column,
        )

        if not values:
            continue

        plt.plot(
            epochs[
                :len(values)
            ],
            values,
            label=display_name,
            linewidth=2,
        )

    plt.xlabel("Epoch")
    plt.ylabel(title)

    plt.title(
        f"Validation {title} "
        f"Across Training"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / filename,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# HELD-OUT TEST METRICS BAR CHART
# ============================================================

model_names = [
    row["model"]
    for row in comparison_rows
]

metric_names = [
    "Precision",
    "Recall",
    "F1",
    "mAP50",
    "mAP50-95",
]

metric_keys = [
    "precision",
    "recall",
    "f1",
    "map50",
    "map50_95",
]


x_positions = list(
    range(
        len(model_names)
    )
)

bar_width = 0.15

plt.figure(
    figsize=(11, 6)
)

for metric_index, (
    metric_name,
    metric_key,
) in enumerate(
    zip(
        metric_names,
        metric_keys,
    )
):

    values = [
        row[
            metric_key
        ]
        for row in comparison_rows
    ]

    positions = [
        x
        + (
            metric_index
            - 2
        )
        * bar_width
        for x in x_positions
    ]

    plt.bar(
        positions,
        values,
        width=bar_width,
        label=metric_name,
    )

plt.xticks(
    x_positions,
    model_names,
)

plt.ylabel("Score")
plt.ylim(0.60, 1.00)

plt.title(
    "Held-Out Test Performance"
)

plt.grid(
    axis="y",
    alpha=0.25,
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "test_metrics_comparison.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# ACCURACY VS SPEED
# ============================================================

plt.figure(
    figsize=(9, 6)
)

for row in comparison_rows:

    x = row[
        "inference_ms"
    ]

    y = row[
        "map50_95"
    ]

    plt.scatter(
        x,
        y,
        s=100,
    )

    plt.annotate(
        row["model"],
        (
            x,
            y,
        ),
        xytext=(6, 6),
        textcoords="offset points",
    )

plt.xlabel(
    "Inference Time "
    "(ms / image)"
)

plt.ylabel(
    "mAP@50-95"
)

plt.title(
    "Accuracy–Efficiency Trade-Off"
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "accuracy_vs_speed.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# SUMMARY
# ============================================================

precision_winner = max(
    comparison_rows,
    key=lambda x: x["precision"],
)

recall_winner = max(
    comparison_rows,
    key=lambda x: x["recall"],
)

f1_winner = max(
    comparison_rows,
    key=lambda x: x["f1"],
)

map50_winner = max(
    comparison_rows,
    key=lambda x: x["map50"],
)

map5095_winner = max(
    comparison_rows,
    key=lambda x: x["map50_95"],
)

speed_winner = min(
    comparison_rows,
    key=lambda x: x["inference_ms"],
)

size_winner = min(
    comparison_rows,
    key=lambda x: x["model_size_mb"],
)

vram_winner = min(
    comparison_rows,
    key=lambda x: x["peak_vram_gb"],
)


summary_lines = [
    "M16 DETECTOR BENCHMARK ANALYSIS",
    "=" * 60,
    "",
    "Completed models:",
    "YOLOv8n, YOLO11n, YOLO11s",
    "",
    "Dataset:",
    "IURS-NPDS leakage-safe source-level split",
    "",
    "Held-out test winners:",
    (
        "Precision: "
        f"{precision_winner['model']} "
        f"({precision_winner['precision']:.4f})"
    ),
    (
        "Recall: "
        f"{recall_winner['model']} "
        f"({recall_winner['recall']:.4f})"
    ),
    (
        "F1: "
        f"{f1_winner['model']} "
        f"({f1_winner['f1']:.4f})"
    ),
    (
        "mAP50: "
        f"{map50_winner['model']} "
        f"({map50_winner['map50']:.4f})"
    ),
    (
        "mAP50-95: "
        f"{map5095_winner['model']} "
        f"({map5095_winner['map50_95']:.4f})"
    ),
    "",
    "Efficiency winners:",
    (
        "Fastest inference: "
        f"{speed_winner['model']} "
        f"({speed_winner['inference_ms']:.3f} ms/image)"
    ),
    (
        "Smallest model: "
        f"{size_winner['model']} "
        f"({size_winner['model_size_mb']:.2f} MB)"
    ),
    (
        "Lowest peak VRAM: "
        f"{vram_winner['model']} "
        f"({vram_winner['peak_vram_gb']:.3f} GB)"
    ),
    "",
    "Project model selection:",
    (
        "YOLO11n is selected as the primary detector "
        "for the next ANPR optimization stage."
    ),
    (
        "It provides the strongest overall "
        "accuracy-efficiency balance: highest "
        "Precision, F1 and mAP50, strong mAP50-95, "
        "and substantially lower compute cost "
        "than YOLO11s."
    ),
    "",
    (
        "YOLO11s achieved the highest mAP50-95, "
        "but the localization improvement is small "
        "relative to its increase in parameters, "
        "model size and inference latency."
    ),
    "",
    (
        "RT-DETR-L is not included in this completed "
        "comparison because its full training run was "
        "deferred due to computational and time constraints."
    ),
]


summary_path = (
    OUTPUT_DIR
    / "M16_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "\n".join(
            summary_lines
        )
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 72)
print("M16 DETECTOR COMPARISON COMPLETE")
print("=" * 72)

print(
    "\nFinal held-out test comparison:\n"
)

for row in comparison_rows:

    print(
        f"{row['model']:10s} | "
        f"P={row['precision']:.4f} | "
        f"R={row['recall']:.4f} | "
        f"F1={row['f1']:.4f} | "
        f"mAP50={row['map50']:.4f} | "
        f"mAP50-95={row['map50_95']:.4f} | "
        f"{row['inference_ms']:.2f} ms"
    )


print(
    "\nFinal train/validation loss gaps:\n"
)

for row in loss_rows:

    print(
        f"{row['model']:10s} | "
        f"Box gap={row['box_gap']:+.4f} | "
        f"CLS gap={row['cls_gap']:+.4f} | "
        f"DFL gap={row['dfl_gap']:+.4f}"
    )


print(
    "\nSelected primary detector: "
    "YOLO11n"
)

print(
    "\nOutputs saved to:"
)

print(
    OUTPUT_DIR
)