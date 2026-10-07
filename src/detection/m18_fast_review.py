from pathlib import Path
import csv
import html
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs


ROOT = Path(__file__).resolve().parents[2]

M18_DIR = (
    ROOT
    / "outputs"
    / "M18_real_road_eval"
)

CSV_PATH = (
    M18_DIR
    / "manual_review.csv"
)

PANEL_DIR = (
    M18_DIR
    / "review_panels"
)

PORT = 8765


# ============================================================
# LOAD CSV
# ============================================================

def load_rows():

    with open(
        CSV_PATH,
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        rows = list(reader)
        fields = list(reader.fieldnames)

    required = [
        "visible_plate_count",
        "correct_detection_count",
        "true_positives",
        "false_positives",
        "false_negatives",
        "reviewed",
        "exclude_from_metrics",
    ]

    for field in required:

        if field not in fields:
            fields.append(field)

    for row in rows:

        for field in required:
            row.setdefault(field, "")

    return rows, fields


ROWS, FIELDS = load_rows()


def save_csv():

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS,
        )

        writer.writeheader()
        writer.writerows(ROWS)


# ============================================================
# HTML
# ============================================================

def button_group(
    name,
    values,
):

    buttons = []

    for value in values:

        buttons.append(
            f"""
            <button
                type="button"
                class="choice"
                onclick="
                    document.getElementById('{name}').value='{value}';
                    updateSelection('{name}', this);
                "
            >
                {value}
            </button>
            """
        )

    return "\n".join(buttons)


def page(
    index,
    error="",
):

    index = max(
        0,
        min(
            index,
            len(ROWS) - 1,
        ),
    )

    row = ROWS[index]

    sample_id = row["sample_id"]

    detections = int(
        row.get(
            "detections",
            "0"
        )
        or 0
    )

    reviewed = sum(
        1
        for r in ROWS
        if r.get("reviewed") == "1"
    )

    error_html = ""

    if error:

        error_html = (
            f'<div class="error">'
            f'{html.escape(error)}'
            f'</div>'
        )

    return f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="utf-8">

<title>M18 Fast Review</title>

<style>

body {{
    margin: 0;
    background: #111;
    color: white;
    font-family: Arial, sans-serif;
}}

.container {{
    max-width: 1500px;
    margin: auto;
    padding: 15px;
}}

.top {{
    display: flex;
    justify-content: space-between;
    margin-bottom: 10px;
    font-size: 18px;
}}

img {{
    width: 100%;
    max-height: 68vh;
    object-fit: contain;
    background: black;
}}

.question {{
    background: #222;
    margin-top: 12px;
    padding: 14px;
    border-radius: 8px;
}}

.choice {{
    font-size: 22px;
    padding: 14px 28px;
    margin: 7px;
    cursor: pointer;
}}

.save {{
    font-size: 22px;
    padding: 16px 35px;
    margin-top: 15px;
    background: #1976d2;
    color: white;
    border: 0;
    border-radius: 7px;
    cursor: pointer;
}}

.quick {{
    font-size: 20px;
    padding: 14px 28px;
    background: #388e3c;
    color: white;
    border: 0;
    margin: 8px;
    cursor: pointer;
}}

.error {{
    background: #8b2525;
    padding: 12px;
    margin: 10px 0;
}}

small {{
    color: #bbb;
}}

</style>

<script>

function updateSelection(name, button) {{

    const buttons =
        button.parentElement.querySelectorAll(".choice");

    buttons.forEach(
        b => b.style.outline = "none"
    );

    button.style.outline =
        "4px solid yellow";
}}


function quickEmpty() {{

    document.getElementById(
        "visible_plate_count"
    ).value = "0";

    document.getElementById(
        "correct_detection_count"
    ).value = "0";

    document.getElementById(
        "reviewForm"
    ).submit();
}}


document.addEventListener(
    "keydown",
    function(event) {{

        if (
            event.key === "n"
            || event.key === "N"
        ) {{

            quickEmpty();
        }}

    }}
);

</script>

</head>


<body>

<div class="container">

<div class="top">

<div>
<b>{sample_id}</b>
&nbsp; | &nbsp;
{html.escape(row["video"])}
&nbsp; | &nbsp;
{row["timestamp_sec"]} sec
</div>

<div>
Frame {index + 1}/{len(ROWS)}
&nbsp; | &nbsp;
Reviewed {reviewed}/{len(ROWS)}
</div>

</div>


<div>

Detector boxes:
<b>{detections}</b>

</div>


{error_html}


<img
    src="/image/{sample_id}.jpg"
>


<form
    id="reviewForm"
    method="POST"
    action="/save"
>

<input
    type="hidden"
    name="index"
    value="{index}"
>

<input
    type="hidden"
    id="visible_plate_count"
    name="visible_plate_count"
>

<input
    type="hidden"
    id="correct_detection_count"
    name="correct_detection_count"
>


<div class="question">

<h2>
1. How many clearly visible real plates?
</h2>

{button_group(
    "visible_plate_count",
    [0, 1, 2, 3, 4, 5]
)}

</div>


<div class="question">

<h2>
2. How many YOLO boxes are actually correct?
</h2>

{button_group(
    "correct_detection_count",
    list(
        range(
            detections + 1
        )
    )
)}

</div>


<button
    class="save"
    type="submit"
>
SAVE & NEXT
</button>


<button
    class="quick"
    type="button"
    onclick="quickEmpty()"
>
NO PLATES HERE
</button>

<small>
Keyboard shortcut: press N if there are no real plates.
</small>


</form>

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
                PANEL_DIR
                / filename
            )

            if not path.exists():

                self.send_error(404)
                return

            data = path.read_bytes()

            self.send_response(200)

            self.send_header(
                "Content-Type",
                "image/jpeg",
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


        body = page(
            index
        ).encode(
            "utf-8"
        )

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "text/html; charset=utf-8",
        )

        self.end_headers()

        self.wfile.write(
            body
        )


    def do_POST(self):

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

        form = parse_qs(
            raw,
            keep_blank_values=True,
        )

        index = int(
            form[
                "index"
            ][0]
        )

        try:

            visible = int(
                form[
                    "visible_plate_count"
                ][0]
            )

            correct = int(
                form[
                    "correct_detection_count"
                ][0]
            )

        except (
            KeyError,
            ValueError,
        ):

            body = page(
                index,
                "Select both answers first.",
            ).encode(
                "utf-8"
            )

            self.send_response(400)

            self.end_headers()

            self.wfile.write(
                body
            )

            return


        row = ROWS[index]

        detections = int(
            row.get(
                "detections",
                "0"
            )
            or 0
        )


        if correct > detections:

            body = page(
                index,
                "Correct detections cannot exceed YOLO boxes.",
            ).encode(
                "utf-8"
            )

            self.send_response(400)

            self.end_headers()

            self.wfile.write(
                body
            )

            return


        if correct > visible:

            body = page(
                index,
                "Correct detections cannot exceed visible plates.",
            ).encode(
                "utf-8"
            )

            self.send_response(400)

            self.end_headers()

            self.wfile.write(
                body
            )

            return


        tp = correct

        fp = (
            detections
            - correct
        )

        fn = (
            visible
            - correct
        )


        row[
            "visible_plate_count"
        ] = str(
            visible
        )

        row[
            "correct_detection_count"
        ] = str(
            correct
        )

        row[
            "true_positives"
        ] = str(
            tp
        )

        row[
            "false_positives"
        ] = str(
            fp
        )

        row[
            "false_negatives"
        ] = str(
            fn
        )

        row[
            "reviewed"
        ] = "1"

        save_csv()


        next_index = min(
            index + 1,
            len(ROWS) - 1,
        )

        self.send_response(303)

        self.send_header(
            "Location",
            f"/?i={next_index}",
        )

        self.end_headers()


def main():

    print("=" * 72)

    print(
        "M18 FAST REVIEW"
    )

    print("=" * 72)

    print(
        "\nOnly answer TWO questions."
    )

    print(
        "\nOpen:"
    )

    print(
        f"http://localhost:{PORT}"
    )

    print(
        "\nPress Ctrl+C when finished."
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
            "\nStopped."
        )


if __name__ == "__main__":
    main()