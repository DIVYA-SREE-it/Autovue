from pathlib import Path
import csv
import html
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs


ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    ROOT
    / "outputs"
    / "M20_ocr_dataset"
    / "M20D_candidate_manifest.csv"
)

DATA_ROOT = (
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

GT_CSV = (
    OUTPUT_DIR
    / "M20E_ground_truth.csv"
)

PORT = 8767


FIELDNAMES = [
    "sample_id",
    "benchmark_split",
    "source_id",
    "ground_truth",
    "status",
    "annotation_source",
]


# ============================================================
# LOAD MANIFEST
# ============================================================

with open(
    MANIFEST,
    newline="",
    encoding="utf-8",
) as f:

    manifest_rows = list(
        csv.DictReader(f)
    )


samples = {
    row["sample_id"]: row
    for row in manifest_rows
}


# DEV first, TEST second.
SAMPLE_IDS = sorted(
    samples.keys(),
    key=lambda sample_id: (
        0
        if samples[sample_id][
            "benchmark_split"
        ] == "dev"
        else 1,
        sample_id,
    )
)


# ============================================================
# GT HELPERS
# ============================================================

def normalize_plate(text):

    return re.sub(
        r"[^A-Z0-9]",
        "",
        text.upper(),
    )


def load_gt():

    if not GT_CSV.exists():
        return {}

    with open(
        GT_CSV,
        newline="",
        encoding="utf-8",
    ) as f:

        return {
            row["sample_id"]: row
            for row in csv.DictReader(f)
        }


def save_gt(data):

    with open(
        GT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()

        for sample_id in SAMPLE_IDS:

            if sample_id in data:
                writer.writerow(
                    data[sample_id]
                )


def next_unresolved_index():

    data = load_gt()

    for index, sample_id in enumerate(
        SAMPLE_IDS
    ):

        if sample_id not in data:
            return index

    return len(SAMPLE_IDS) - 1


def completed_count():

    return len(
        load_gt()
    )


# ============================================================
# HTML
# ============================================================

PAGE = r"""
<!DOCTYPE html>
<html>

<head>

<meta charset="utf-8">
<title>M20 OCR Ground Truth</title>

<style>

body {
    margin: 0;
    background: #111;
    color: #eee;
    font-family: Arial, sans-serif;
}

.container {
    max-width: 1100px;
    margin: auto;
    padding: 25px;
}

.header {
    display: flex;
    justify-content: space-between;
    font-size: 18px;
}

.split {
    margin-top: 12px;
    padding: 8px;
    background: #263238;
    border-radius: 6px;
}

.image-area {
    margin-top: 25px;
    height: 430px;
    background: #050505;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}

.image-area img {
    max-width: 850px;
    max-height: 360px;
    transform-origin: center center;
    image-rendering: auto;
}

.controls {
    margin-top: 15px;
}

.entry {
    margin-top: 20px;
    background: #222;
    padding: 18px;
    border-radius: 8px;
}

input {
    font-size: 24px;
    width: 370px;
    padding: 12px;
    text-transform: uppercase;
}

button {
    padding: 11px 16px;
    margin: 8px 5px 0 0;
    font-size: 15px;
    cursor: pointer;
}

.verify {
    background: #1976d2;
    color: white;
    border: 0;
}

.unreadable {
    background: #b26a00;
    color: white;
    border: 0;
}

.notplate {
    background: #8b2c2c;
    color: white;
    border: 0;
}

.nav {
    background: #555;
    color: white;
    border: 0;
}

.rotate {
    background: #455a64;
    color: white;
    border: 0;
}

.existing {
    margin-top: 15px;
    background: #183c24;
    padding: 10px;
}

.instructions {
    margin-top: 15px;
    line-height: 1.5;
    color: #ccc;
}

</style>

</head>


<body>

<div class="container">

<div class="header">

<div>
<b>__SAMPLE_ID__</b>
</div>

<div>
__INDEX__ / __TOTAL__
&nbsp; | &nbsp;
Completed: __COMPLETED__
</div>

</div>


<div class="split">

Benchmark split:
<b>__SPLIT__</b>

&nbsp; | &nbsp;

Source:
__SOURCE_ID__

</div>


<div class="instructions">

Read the registration directly from the image.

Do not use OCR predictions and do not guess uncertain characters.

Use rotation buttons only to make the plate easier to read.

</div>


__EXISTING__


<div class="image-area">

<img
    id="plateImage"
    src="/image/__SAMPLE_ID__"
>

</div>


<div class="controls">

<button
    class="rotate"
    onclick="rotateLeft()"
>
↺ Rotate Left
</button>

<button
    class="rotate"
    onclick="rotateRight()"
>
↻ Rotate Right
</button>

<button
    class="rotate"
    onclick="resetRotation()"
>
Reset
</button>

</div>


<div class="entry">

<input
    id="plateInput"
    placeholder="Example: GJ01AB1234"
    autocomplete="off"
>

<br>

<button
    class="verify"
    onclick="saveVerified()"
>
VERIFIED & Next
</button>

<button
    class="unreadable"
    onclick="saveStatus('UNREADABLE')"
>
UNREADABLE
</button>

<button
    class="notplate"
    onclick="saveStatus('NOT_A_PLATE')"
>
NOT A PLATE
</button>

<br>

<button
    class="nav"
    onclick="goPrevious()"
>
Previous
</button>

<button
    class="nav"
    onclick="goNext()"
>
Skip / Next
</button>

</div>

</div>


<script>

const sampleId =
    __SAMPLE_ID_JSON__;

const index =
    __INDEX_ZERO__;

const total =
    __TOTAL__;

let rotation = 0;


function applyRotation() {

    document
        .getElementById(
            "plateImage"
        )
        .style
        .transform =
            "rotate("
            + rotation
            + "deg)";
}


function rotateLeft() {

    rotation -= 90;

    applyRotation();
}


function rotateRight() {

    rotation += 90;

    applyRotation();
}


function resetRotation() {

    rotation = 0;

    applyRotation();
}


async function submit(
    groundTruth,
    status
) {

    const response =
        await fetch(
            "/save",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body:
                    JSON.stringify({
                        sample_id:
                            sampleId,

                        ground_truth:
                            groundTruth,

                        status:
                            status
                    })
            }
        );


    if (!response.ok) {

        alert(
            await response.text()
        );

        return;
    }


    if (index >= total - 1) {

        alert(
            "All 45 samples reached."
        );

        location.href =
            "/?i=" + index;

        return;
    }


    location.href =
        "/?i=" + (index + 1);
}


function saveVerified() {

    let value =
        document
        .getElementById(
            "plateInput"
        )
        .value
        .toUpperCase()
        .replace(
            /[^A-Z0-9]/g,
            ""
        );


    if (value.length < 5) {

        alert(
            "Enter the complete readable plate, or choose UNREADABLE."
        );

        return;
    }


    submit(
        value,
        "VERIFIED"
    );
}


function saveStatus(status) {

    submit(
        "",
        status
    );
}


function goPrevious() {

    location.href =
        "/?i="
        + Math.max(
            0,
            index - 1
        );
}


function goNext() {

    location.href =
        "/?i="
        + Math.min(
            total - 1,
            index + 1
        );
}


document
    .getElementById(
        "plateInput"
    )
    .addEventListener(
        "keydown",
        function(event) {

            if (
                event.key === "Enter"
            ) {
                saveVerified();
            }
        }
    );

</script>

</body>
</html>
"""


# ============================================================
# PAGE
# ============================================================

def make_page(index):

    index = max(
        0,
        min(
            index,
            len(SAMPLE_IDS) - 1,
        ),
    )

    sample_id = SAMPLE_IDS[
        index
    ]

    row = samples[
        sample_id
    ]

    gt = load_gt()

    existing = ""

    if sample_id in gt:

        record = gt[
            sample_id
        ]

        value = (
            record[
                "ground_truth"
            ]
            or "-"
        )

        existing = f"""
        <div class="existing">

        Existing:
        <b>{html.escape(record["status"])}</b>

        &nbsp; | &nbsp;

        GT:
        <b>{html.escape(value)}</b>

        </div>
        """


    page = PAGE

    replacements = {
        "__SAMPLE_ID__":
            sample_id,

        "__SAMPLE_ID_JSON__":
            json.dumps(
                sample_id
            ),

        "__INDEX__":
            str(
                index + 1
            ),

        "__INDEX_ZERO__":
            str(index),

        "__TOTAL__":
            str(
                len(SAMPLE_IDS)
            ),

        "__COMPLETED__":
            str(
                completed_count()
            ),

        "__SPLIT__":
            html.escape(
                row[
                    "benchmark_split"
                ].upper()
            ),

        "__SOURCE_ID__":
            html.escape(
                row[
                    "source_id"
                ]
            ),

        "__EXISTING__":
            existing,
    }

    for old, new in replacements.items():

        page = page.replace(
            old,
            new,
        )

    return page


# ============================================================
# SERVER
# ============================================================

class Handler(
    BaseHTTPRequestHandler
):

    def do_GET(self):

        parsed = urlparse(
            self.path
        )


        if parsed.path.startswith(
            "/image/"
        ):

            sample_id = (
                parsed.path
                .split(
                    "/image/",
                    1,
                )[1]
            )

            if sample_id not in samples:

                self.send_error(404)
                return


            row = samples[
                sample_id
            ]

            path = (
                DATA_ROOT
                / row[
                    "benchmark_split"
                ]
                / "images"
                / row[
                    "crop_file"
                ]
            )


            if not path.exists():

                self.send_error(
                    404,
                    "Image not found",
                )

                return


            data = path.read_bytes()

            self.send_response(200)

            self.send_header(
                "Content-Type",
                "image/jpeg",
            )

            self.send_header(
                "Content-Length",
                str(
                    len(data)
                ),
            )

            self.end_headers()

            self.wfile.write(
                data
            )

            return


        query = parse_qs(
            parsed.query
        )

        try:

            index = int(
                query.get(
                    "i",
                    [
                        str(
                            next_unresolved_index()
                        )
                    ],
                )[0]
            )

        except ValueError:

            index = (
                next_unresolved_index()
            )


        body = make_page(
            index
        ).encode(
            "utf-8"
        )

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "text/html; charset=utf-8",
        )

        self.send_header(
            "Content-Length",
            str(
                len(body)
            ),
        )

        self.end_headers()

        self.wfile.write(
            body
        )


    def do_POST(self):

        if self.path != "/save":

            self.send_error(404)
            return


        length = int(
            self.headers.get(
                "Content-Length",
                "0",
            )
        )

        payload = json.loads(
            self.rfile.read(
                length
            ).decode(
                "utf-8"
            )
        )


        sample_id = str(
            payload[
                "sample_id"
            ]
        )

        status = str(
            payload[
                "status"
            ]
        )

        ground_truth = (
            normalize_plate(
                str(
                    payload.get(
                        "ground_truth",
                        "",
                    )
                )
            )
        )


        if sample_id not in samples:

            self.send_error(
                400,
                "Unknown sample",
            )

            return


        if status not in {
            "VERIFIED",
            "UNREADABLE",
            "NOT_A_PLATE",
        }:

            self.send_error(
                400,
                "Invalid status",
            )

            return


        if (
            status == "VERIFIED"
            and len(
                ground_truth
            ) < 5
        ):

            self.send_error(
                400,
                "Plate text too short",
            )

            return


        if status != "VERIFIED":

            ground_truth = ""


        row = samples[
            sample_id
        ]

        gt = load_gt()

        gt[
            sample_id
        ] = {
            "sample_id":
                sample_id,

            "benchmark_split":
                row[
                    "benchmark_split"
                ],

            "source_id":
                row[
                    "source_id"
                ],

            "ground_truth":
                ground_truth,

            "status":
                status,

            "annotation_source":
                "manual_visual",
        }


        save_gt(
            gt
        )


        self.send_response(200)

        self.end_headers()

        self.wfile.write(
            b"saved"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("M20E — OCR BENCHMARK GROUND-TRUTH ANNOTATOR")
    print("=" * 72)

    print(
        "\nTotal samples :",
        len(SAMPLE_IDS),
    )

    print(
        "DEV           :",
        sum(
            1
            for x in manifest_rows
            if x[
                "benchmark_split"
            ] == "dev"
        ),
    )

    print(
        "TEST          :",
        sum(
            1
            for x in manifest_rows
            if x[
                "benchmark_split"
            ] == "test"
        ),
    )

    print(
        "Completed     :",
        completed_count(),
    )

    print(
        f"\nOpen:\n"
        f"http://localhost:{PORT}"
    )

    print(
        "\nOCR predictions are intentionally hidden."
    )

    print(
        "Press Ctrl+C when complete."
    )


    server = HTTPServer(
        (
            "0.0.0.0",
            PORT,
        ),
        Handler,
    )


    try:

        server.serve_forever()

    except KeyboardInterrupt:

        print(
            "\nAnnotator stopped."
        )


if __name__ == "__main__":
    main()