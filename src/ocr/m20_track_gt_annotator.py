from pathlib import Path
import csv
import html
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

M09_SUMMARY = (
    ROOT
    / "outputs"
    / "M09_ocr_video_tracking"
    / "track_summary.csv"
)

M09_CROPS = (
    ROOT
    / "outputs"
    / "M09_ocr_video_tracking"
    / "best_plate_crops"
)

M12_GT = (
    ROOT
    / "outputs"
    / "M12_manual_ground_truth"
    / "verified_plates.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "M20_ocr_ground_truth"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

GT_CSV = (
    OUTPUT_DIR
    / "track_ground_truth.csv"
)

PORT = 8766


# ============================================================
# LOAD M09 TRACKS
# ============================================================

with open(
    M09_SUMMARY,
    newline="",
    encoding="utf-8",
) as f:

    rows = list(
        csv.DictReader(f)
    )


tracks = {}

for row in rows:

    track_id = int(
        row["track_id"]
    )

    tracks.setdefault(
        track_id,
        []
    ).append(row)


for track_id in tracks:

    tracks[track_id].sort(
        key=lambda r: int(
            r["rank"]
        )
    )


TRACK_IDS = sorted(
    tracks.keys()
)


# ============================================================
# LOAD EXISTING M12 VERIFIED GT
# ============================================================

existing_m12 = {}

if M12_GT.exists():

    with open(
        M12_GT,
        newline="",
        encoding="utf-8",
    ) as f:

        for row in csv.DictReader(f):

            track_id = int(
                row["track_id"]
            )

            existing_m12[
                track_id
            ] = row[
                "ground_truth"
            ].strip().upper()


# ============================================================
# LOAD / INITIALIZE M20 GT
# ============================================================

FIELDNAMES = [
    "track_id",
    "ground_truth",
    "status",
    "source",
]


def load_gt():

    data = {}

    if GT_CSV.exists():

        with open(
            GT_CSV,
            newline="",
            encoding="utf-8",
        ) as f:

            for row in csv.DictReader(f):

                data[
                    int(
                        row["track_id"]
                    )
                ] = row

    return data


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

        for track_id in sorted(
            data.keys()
        ):

            writer.writerow(
                data[
                    track_id
                ]
            )


gt_data = load_gt()


# Import M12 manually verified plates once.
for track_id, plate in existing_m12.items():

    if (
        track_id in tracks
        and track_id not in gt_data
    ):

        gt_data[
            track_id
        ] = {
            "track_id":
                track_id,

            "ground_truth":
                plate,

            "status":
                "VERIFIED",

            "source":
                "M12_manual_visual",
        }


save_gt(
    gt_data
)


# ============================================================
# HELPERS
# ============================================================

def normalize_plate(text):

    return re.sub(
        r"[^A-Z0-9]",
        "",
        text.upper(),
    )


def completed_count():

    data = load_gt()

    return sum(
        1
        for track_id in TRACK_IDS
        if track_id in data
    )


def next_unresolved_index():

    data = load_gt()

    for index, track_id in enumerate(
        TRACK_IDS
    ):

        if track_id not in data:
            return index

    return max(
        0,
        len(TRACK_IDS) - 1,
    )


# ============================================================
# HTML TEMPLATE
#
# Deliberately NOT an f-string.
# This avoids Python/JavaScript brace problems.
# ============================================================

PAGE_TEMPLATE = r"""
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
    max-width: 1500px;
    margin: auto;
    padding: 20px;
}

.header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 15px;
    font-size: 18px;
}

.info {
    background: #222;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 15px;
}

.crops {
    display: flex;
    flex-wrap: wrap;
    gap: 15px;
    align-items: flex-start;
}

.crop-card {
    background: #1c1c1c;
    padding: 10px;
    border-radius: 8px;
}

.crop-card img {
    max-width: 420px;
    max-height: 230px;
    image-rendering: auto;
}

.caption {
    margin-top: 5px;
    font-size: 14px;
    color: #bbb;
}

.entry {
    margin-top: 22px;
    padding: 15px;
    background: #222;
    border-radius: 8px;
}

input {
    padding: 12px;
    font-size: 22px;
    width: 350px;
    text-transform: uppercase;
}

button {
    padding: 12px 18px;
    margin: 10px 8px 0 0;
    font-size: 16px;
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

.existing {
    background: #163a21;
    padding: 10px;
    margin-top: 12px;
    border-radius: 6px;
}

</style>

</head>


<body>

<div class="container">

<div class="header">

<div>
<b>Track __TRACK_ID__</b>
&nbsp; | &nbsp;
Vehicle: __VEHICLE__
</div>

<div>
Track __INDEX__ / __TOTAL__
&nbsp; | &nbsp;
Completed: __COMPLETED__
</div>

</div>


<div class="info">

Look at all crops from the same vehicle track.

<br><br>

If the registration number is readable, type it without spaces or hyphens.

<br>

Example:
<b>TN09AB1234</b>

<br><br>

If it is clearly a real plate but cannot be read confidently,
choose <b>UNREADABLE</b>.

<br>

If the crop is not actually a number plate,
choose <b>NOT A PLATE</b>.

<br><br>

<b>Do not guess characters.</b>

</div>


__EXISTING_HTML__


<div class="crops">

__CROP_HTML__

</div>


<div class="entry">

<input
    id="plateInput"
    placeholder="Enter plate number"
    autocomplete="off"
>

<br>

<button
    class="verify"
    onclick="saveVerified()"
>
Save VERIFIED & Next
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
    onclick="previousTrack()"
>
Previous
</button>

<button
    class="nav"
    onclick="nextTrack()"
>
Skip / Next
</button>

</div>

</div>


<script>

const trackId = __TRACK_ID_JSON__;
const index = __INDEX_ZERO__;
const total = __TOTAL__;


async function submitAnnotation(
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
                        track_id:
                            trackId,

                        ground_truth:
                            groundTruth,

                        status:
                            status
                    })
            }
        );


    if (!response.ok) {

        const message =
            await response.text();

        alert(message);

        return;
    }


    if (index >= total - 1) {

        alert(
            "All available tracks reached."
        );

        window.location.href =
            "/?i=" + index;

        return;
    }


    window.location.href =
        "/?i=" + (index + 1);
}


function saveVerified() {

    let value =
        document.getElementById(
            "plateInput"
        ).value;

    value =
        value
        .toUpperCase()
        .replace(
            /[^A-Z0-9]/g,
            ""
        );


    if (value.length < 4) {

        alert(
            "Please enter a readable plate number, or mark UNREADABLE."
        );

        return;
    }


    submitAnnotation(
        value,
        "VERIFIED"
    );
}


function saveStatus(status) {

    submitAnnotation(
        "",
        status
    );
}


function previousTrack() {

    const previous =
        Math.max(
            0,
            index - 1
        );

    window.location.href =
        "/?i=" + previous;
}


function nextTrack() {

    const next =
        Math.min(
            total - 1,
            index + 1
        );

    window.location.href =
        "/?i=" + next;
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
# PAGE GENERATION
# ============================================================

def make_page(index):

    index = max(
        0,
        min(
            index,
            len(TRACK_IDS) - 1,
        ),
    )

    track_id = TRACK_IDS[
        index
    ]

    crop_rows = tracks[
        track_id
    ]

    vehicle = crop_rows[
        0
    ]["vehicle"]

    crop_html = []

    for row in crop_rows:

        filename = row[
            "filename"
        ]

        rank = row[
            "rank"
        ]

        quality = row[
            "quality_score"
        ]

        crop_html.append(
            f"""
            <div class="crop-card">

                <img
                    src="/crop/{html.escape(filename)}"
                >

                <div class="caption">
                    Rank {html.escape(rank)}
                    &nbsp; | &nbsp;
                    Quality {html.escape(quality)}
                </div>

            </div>
            """
        )


    data = load_gt()

    existing_html = ""

    if track_id in data:

        existing = data[
            track_id
        ]

        gt = (
            existing[
                "ground_truth"
            ]
            or "-"
        )

        existing_html = f"""
        <div class="existing">

        Existing annotation:
        <b>{html.escape(existing["status"])}</b>

        &nbsp; | &nbsp;

        GT:
        <b>{html.escape(gt)}</b>

        &nbsp; | &nbsp;

        Source:
        {html.escape(existing["source"])}

        </div>
        """


    page = PAGE_TEMPLATE

    replacements = {
        "__TRACK_ID__":
            str(track_id),

        "__TRACK_ID_JSON__":
            json.dumps(
                track_id
            ),

        "__VEHICLE__":
            html.escape(
                vehicle
            ),

        "__INDEX__":
            str(
                index + 1
            ),

        "__INDEX_ZERO__":
            str(index),

        "__TOTAL__":
            str(
                len(TRACK_IDS)
            ),

        "__COMPLETED__":
            str(
                completed_count()
            ),

        "__EXISTING_HTML__":
            existing_html,

        "__CROP_HTML__":
            "\n".join(
                crop_html
            ),
    }


    for key, value in replacements.items():

        page = page.replace(
            key,
            value,
        )


    return page


# ============================================================
# HTTP SERVER
# ============================================================

class Handler(
    BaseHTTPRequestHandler
):

    def do_GET(self):

        parsed = urlparse(
            self.path
        )


        # ----------------------------------------------------
        # SERVE CROP
        # ----------------------------------------------------

        if parsed.path.startswith(
            "/crop/"
        ):

            filename = (
                parsed.path
                .split(
                    "/crop/",
                    1,
                )[1]
            )

            path = (
                M09_CROPS
                / filename
            )


            if not path.exists():

                self.send_error(
                    404,
                    "Crop not found",
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


        # ----------------------------------------------------
        # MAIN PAGE
        # ----------------------------------------------------

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

        raw = self.rfile.read(
            length
        ).decode(
            "utf-8"
        )


        try:

            payload = json.loads(
                raw
            )

            track_id = int(
                payload[
                    "track_id"
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

        except Exception:

            self.send_error(
                400,
                "Invalid payload",
            )

            return


        if track_id not in tracks:

            self.send_error(
                400,
                "Unknown track",
            )

            return


        allowed_status = {
            "VERIFIED",
            "UNREADABLE",
            "NOT_A_PLATE",
        }


        if status not in allowed_status:

            self.send_error(
                400,
                "Invalid status",
            )

            return


        if (
            status == "VERIFIED"
            and len(
                ground_truth
            ) < 4
        ):

            self.send_error(
                400,
                "Verified plate is too short",
            )

            return


        if status != "VERIFIED":

            ground_truth = ""


        data = load_gt()

        data[
            track_id
        ] = {
            "track_id":
                track_id,

            "ground_truth":
                ground_truth,

            "status":
                status,

            "source":
                "M20_manual_track_review",
        }


        save_gt(
            data
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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("M20B — TRACK-LEVEL OCR GROUND TRUTH")
    print("=" * 72)

    print(
        f"\nM09 crop rows       : "
        f"{len(rows)}"
    )

    print(
        f"Unique tracks       : "
        f"{len(TRACK_IDS)}"
    )

    print(
        f"Existing M12 GT     : "
        f"{len(existing_m12)}"
    )

    print(
        f"Already completed   : "
        f"{completed_count()}"
    )

    print(
        "\nOpen:"
    )

    print(
        f"http://localhost:{PORT}"
    )

    print(
        "\nOCR predictions are intentionally hidden."
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