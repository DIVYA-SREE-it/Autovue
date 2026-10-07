from pathlib import Path
import csv
import html
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs


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

PORT = 8765

LABEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD ANNOTATION TASKS
# ============================================================

with open(
    MANIFEST,
    newline="",
    encoding="utf-8",
) as f:

    manifest_rows = list(
        csv.DictReader(f)
    )


TASKS = [
    row
    for row in manifest_rows
    if row["annotation_status"]
    == "needs_plate_boxes"
]


# ============================================================
# YOLO LABEL HELPERS
# ============================================================

def label_path(sample_id):

    return (
        LABEL_DIR
        / f"{sample_id}.txt"
    )


def load_existing_boxes(sample_id):

    path = label_path(
        sample_id
    )

    if not path.exists():
        return []

    boxes = []

    text = path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return []

    for line in text.splitlines():

        parts = line.split()

        if len(parts) != 5:
            continue

        class_id, xc, yc, w, h = parts

        if class_id != "0":
            continue

        boxes.append({
            "xc": float(xc),
            "yc": float(yc),
            "w": float(w),
            "h": float(h),
        })

    return boxes


def save_boxes(
    sample_id,
    boxes,
):

    path = label_path(
        sample_id
    )

    lines = []

    for box in boxes:

        xc = float(
            box["xc"]
        )

        yc = float(
            box["yc"]
        )

        w = float(
            box["w"]
        )

        h = float(
            box["h"]
        )

        lines.append(
            f"0 "
            f"{xc:.6f} "
            f"{yc:.6f} "
            f"{w:.6f} "
            f"{h:.6f}"
        )

    path.write_text(
        "\n".join(lines)
        + ("\n" if lines else ""),
        encoding="utf-8",
    )


# ============================================================
# PAGE
# ============================================================

def make_page(
    index,
    message="",
):

    index = max(
        0,
        min(
            index,
            len(TASKS) - 1,
        ),
    )

    row = TASKS[index]

    sample_id = row[
        "sample_id"
    ]

    expected = int(
        row[
            "visible_plates"
        ]
    )

    existing = (
        load_existing_boxes(
            sample_id
        )
    )

    completed = sum(
        1
        for task in TASKS
        if label_path(
            task["sample_id"]
        ).exists()
    )

    message_html = ""

    if message:

        message_html = (
            f'<div class="message">'
            f'{html.escape(message)}'
            f'</div>'
        )

    return f"""
<!DOCTYPE html>
<html>

<head>

<meta charset="utf-8">

<title>M19 Plate Annotation</title>

<style>

body {{
    background: #111;
    color: #eee;
    font-family: Arial, sans-serif;
    margin: 0;
}}

.container {{
    max-width: 1500px;
    margin: auto;
    padding: 18px;
}}

.header {{
    display: flex;
    justify-content: space-between;
    margin-bottom: 12px;
    font-size: 18px;
}}

.instructions {{
    background: #222;
    padding: 12px;
    margin-bottom: 12px;
    border-radius: 8px;
}}

.canvas-wrap {{
    width: 100%;
    text-align: center;
}}

canvas {{
    max-width: 100%;
    max-height: 72vh;
    border: 2px solid #555;
    cursor: crosshair;
}}

button {{
    padding: 12px 22px;
    margin: 10px 6px 0 0;
    font-size: 17px;
    cursor: pointer;
}}

.save {{
    background: #1976d2;
    color: white;
    border: none;
}}

.clear {{
    background: #8b2c2c;
    color: white;
    border: none;
}}

.undo {{
    background: #555;
    color: white;
    border: none;
}}

.message {{
    background: #7d2929;
    padding: 10px;
    margin-bottom: 10px;
}}

.status {{
    margin-top: 12px;
    font-size: 18px;
}}

</style>

</head>

<body>

<div class="container">

<div class="header">

<div>
<b>{sample_id}</b>
&nbsp; | &nbsp;
{html.escape(row["video"])}
&nbsp; | &nbsp;
{row["timestamp_sec"]} sec
</div>

<div>
Image {index + 1}/{len(TASKS)}
&nbsp; | &nbsp;
Label files: {completed}/{len(TASKS)}
</div>

</div>


<div class="instructions">

<b>Expected real plates in this image: {expected}</b>

<br><br>

Drag one rectangle around each real registration plate.

<br>

Annotate the <b>physical number plate</b>, not the whole vehicle.

<br>

Do not annotate advertisements, shop signs, stickers,
logos, or other rectangular text.

</div>

{message_html}


<div class="canvas-wrap">

<canvas id="canvas"></canvas>

</div>


<div class="status">

Boxes drawn:
<b id="count">0</b>
&nbsp; / &nbsp;
Expected:
<b>{expected}</b>

</div>


<button
    class="undo"
    onclick="undoBox()"
>
Undo last box
</button>


<button
    class="clear"
    onclick="clearBoxes()"
>
Clear all
</button>


<button
    class="save"
    onclick="saveAndNext()"
>
Save & Next
</button>


<script>

const sampleId =
    {json.dumps(sample_id)};

const expected =
    {expected};

const index =
    {index};

const existingNormalized =
    {json.dumps(existing)};


const canvas =
    document.getElementById(
        "canvas"
    );

const ctx =
    canvas.getContext(
        "2d"
    );

const image =
    new Image();


let boxes = [];

let dragging = false;

let startX = 0;
let startY = 0;
let currentX = 0;
let currentY = 0;


image.onload = function() {{

    canvas.width =
        image.naturalWidth;

    canvas.height =
        image.naturalHeight;

    boxes =
        existingNormalized.map(
            box => ({{

                x:
                    (
                        box.xc
                        - box.w / 2
                    )
                    * canvas.width,

                y:
                    (
                        box.yc
                        - box.h / 2
                    )
                    * canvas.height,

                w:
                    box.w
                    * canvas.width,

                h:
                    box.h
                    * canvas.height
            }})
        );

    redraw();
}};


image.src =
    "/image/"
    + sampleId
    + ".jpg";


function canvasPoint(event) {{

    const rect =
        canvas.getBoundingClientRect();

    const scaleX =
        canvas.width
        / rect.width;

    const scaleY =
        canvas.height
        / rect.height;

    return {{

        x:
            (
                event.clientX
                - rect.left
            )
            * scaleX,

        y:
            (
                event.clientY
                - rect.top
            )
            * scaleY
    }};
}}


canvas.addEventListener(
    "mousedown",
    function(event) {{

        const p =
            canvasPoint(
                event
            );

        startX = p.x;
        startY = p.y;

        currentX = p.x;
        currentY = p.y;

        dragging = true;
    }}
);


canvas.addEventListener(
    "mousemove",
    function(event) {{

        if (!dragging)
            return;

        const p =
            canvasPoint(
                event
            );

        currentX = p.x;
        currentY = p.y;

        redraw();
    }}
);


canvas.addEventListener(
    "mouseup",
    function(event) {{

        if (!dragging)
            return;

        dragging = false;

        const p =
            canvasPoint(
                event
            );

        currentX = p.x;
        currentY = p.y;

        const x =
            Math.min(
                startX,
                currentX
            );

        const y =
            Math.min(
                startY,
                currentY
            );

        const w =
            Math.abs(
                currentX
                - startX
            );

        const h =
            Math.abs(
                currentY
                - startY
            );

        if (
            w >= 4
            && h >= 4
        ) {{

            boxes.push({{
                x: x,
                y: y,
                w: w,
                h: h
            }});
        }}

        redraw();
    }}
);


function redraw() {{

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    ctx.drawImage(
        image,
        0,
        0
    );

    ctx.lineWidth = 5;
    ctx.strokeStyle = "lime";
    ctx.font = "28px Arial";
    ctx.fillStyle = "lime";


    boxes.forEach(
        (box, i) => {{

            ctx.strokeRect(
                box.x,
                box.y,
                box.w,
                box.h
            );

            ctx.fillText(
                "Plate "
                + (i + 1),

                box.x,

                Math.max(
                    30,
                    box.y - 8
                )
            );
        }}
    );


    if (dragging) {{

        const x =
            Math.min(
                startX,
                currentX
            );

        const y =
            Math.min(
                startY,
                currentY
            );

        const w =
            Math.abs(
                currentX
                - startX
            );

        const h =
            Math.abs(
                currentY
                - startY
            );

        ctx.strokeStyle =
            "yellow";

        ctx.strokeRect(
    x,
    y,
    w,
    h
);

}}

document.getElementById(
    "count"
).textContent =
    boxes.length;
}}


function undoBox() {{

    boxes.pop();

    redraw();
}}


function clearBoxes() {{

    boxes = [];

    redraw();
}}


async function saveAndNext() {{

    if (
        boxes.length
        !== expected
    ) {{

        alert(
            "Expected "
            + expected
            + " plate box(es), but you drew "
            + boxes.length
            + "."
        );

        return;
        
        "Expected "
        + expected
        + " plate box(es), but you drew "
        + boxes.length
        + "."
    
    }}

    const normalized =
        boxes.map(
            box => ({{

                xc:
                    (
                        box.x
                        + box.w / 2
                    )
                    / canvas.width,

                yc:
                    (
                        box.y
                        + box.h / 2
                    )
                    / canvas.height,

                w:
                    box.w
                    / canvas.width,

                h:
                    box.h
                    / canvas.height
            }})
        );


    const response =
        await fetch(
            "/save",

            {{
                method: "POST",

                headers: {{
                    "Content-Type":
                        "application/json"
                }},

                body:
                    JSON.stringify({{
                        index: index,
                        sample_id:
                            sampleId,
                        boxes:
                            normalized
                    }})
            }}
        );


    if (!response.ok) {{

    const message =
        await response.text();

    alert(
        message
    );

    return;
}}

    const next =
        Math.min(
            index + 1,
            {len(TASKS) - 1}
        );


    if (
    index === {len(TASKS) - 1}
) {{

    alert(
        "All M19 positive frames are annotated!"
    );

    window.location.href =
        "/?i="
        + index;

    return;
}}

window.location.href =
    "/?i="
    + next;

}}

</script>

</div>

</body>

</html>
"""


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

            filename = (
                parsed.path
                .split(
                    "/image/",
                    1
                )[1]
            )

            path = (
                IMAGE_DIR
                / filename
            )

            if not path.exists():

                self.send_error(
                    404,
                    "Image not found"
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
                str(len(data)),
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
                    ["0"]
                )[0]
            )

        except ValueError:

            index = 0


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
            str(len(body)),
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
                "0"
            )
        )

        raw = self.rfile.read(
            length
        ).decode(
            "utf-8"
        )


        try:

            payload = json.loads(
                raw
            )

            index = int(
                payload[
                    "index"
                ]
            )

            sample_id = str(
                payload[
                    "sample_id"
                ]
            )

            boxes = payload[
                "boxes"
            ]

        except Exception:

            self.send_error(
                400,
                "Invalid annotation payload"
            )

            return


        if not (
            0 <= index < len(TASKS)
        ):

            self.send_error(
                400,
                "Invalid task index"
            )

            return


        expected_id = (
            TASKS[index][
                "sample_id"
            ]
        )


        if sample_id != expected_id:

            self.send_error(
                400,
                "Sample ID mismatch"
            )

            return


        expected_count = int(
            TASKS[index][
                "visible_plates"
            ]
        )


        if len(boxes) != expected_count:

            self.send_error(
                400,
                (
                    f"Expected "
                    f"{expected_count} boxes"
                ),
            )

            return


        for box in boxes:

            for key in [
                "xc",
                "yc",
                "w",
                "h",
            ]:

                value = float(
                    box[key]
                )

                if not (
                    0.0 <= value <= 1.0
                ):

                    self.send_error(
                        400,
                        "Invalid normalized box"
                    )

                    return


            if (
                float(box["w"]) <= 0
                or float(box["h"]) <= 0
            ):

                self.send_error(
                    400,
                    "Invalid box size"
                )

                return


        save_boxes(
            sample_id,
            boxes,
        )


        self.send_response(200)

        self.send_header(
            "Content-Type",
            "text/plain",
        )

        self.end_headers()

        self.wfile.write(
            b"saved"
        )


def main():

    print("=" * 72)

    print(
        "M19C — REAL-ROAD PLATE ANNOTATOR"
    )

    print("=" * 72)

    print(
        f"\nPositive frames to annotate: "
        f"{len(TASKS)}"
    )

    print(
        "\nOpen:"
    )

    print(
        f"http://localhost:{PORT}"
    )

    print(
        "\nDraw boxes around every real plate."
    )

    print(
        "Press Ctrl+C when finished."
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