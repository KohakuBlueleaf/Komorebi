"""Draws the README's top figure, docs/images/teaser-{light,dark}.svg: what a user
writes, the card running it by itself, what comes out.

usage: python docs/images/teaser.py    (needs Pillow, for the four thumbnails)

Left, one card for each way of talking the centre draws, its lines taken from the file
the card names (experiments/language/dense_class/linear_class.py, examples/03_broadcast
and 05_forward as classes, projects/raster/thread and examples/10_x280_ports) and marked
with the arrow the centre draws for it. The numbers are copied from committed results:
the frame's passes and the pictures' frame times from projects/raster and
projects/raytrace, the decode rows from projects/decode/README.md (whole tokens) beside
tt-metal's tt_transformers demo on the same card, the kernel rows from docs/measured.md
(a loop of 32 calls) beside TT-NN on the same card.
"""

import base64
import io
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image

W, H = 1280, 690


THEMES = {
    "light": {
        "bg": "#ffffff",
        "fg": "#1f2328",
        "mute": "#59636e",
        "box": "#f6f8fa",
        "line": "#d1d9e0",
        "dram": "#d5dde6",
        "a": "#bc4c00",
        "afill": "#ffe3cc",
        "aline": "#f2b98a",
        "asoft": "#fff1e5",
        "aink": "#ffffff",
        "w": "#0969da",
        "wsoft": "#ddf4ff",
        "rule": "#818b98",
        "g": "#1a7f37",
        "gsoft": "#dafbe1",
        "m": "#8250df",
        "msoft": "#f1e8ff",
        "b": "#1b7c83",
        "bsoft": "#dff5f5",
        "r": "#bf3989",
        "rsoft": "#ffeff7",
        "d": "#9a6700",
        "dsoft": "#fff8c5",
    },
    "dark": {
        "bg": "#0d1117",
        "fg": "#e6edf3",
        "mute": "#9198a1",
        "box": "#161b22",
        "line": "#3d444d",
        "dram": "#2d3540",
        "a": "#f0883e",
        "afill": "#3a2412",
        "aline": "#7a4a22",
        "asoft": "#2d1a0c",
        "aink": "#0d1117",
        "w": "#4493f8",
        "wsoft": "#0c2d6b",
        "rule": "#6e7681",
        "g": "#3fb950",
        "gsoft": "#12261a",
        "m": "#a371f7",
        "msoft": "#2a1a4d",
        "b": "#39c5cf",
        "bsoft": "#0b2f33",
        "r": "#db61a2",
        "rsoft": "#3a1128",
        "d": "#e3b341",
        "dsoft": "#2e2508",
    },
}

SANS = "-apple-system, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'DejaVu Sans Mono', monospace"

# One settled frame of 10^6 triangles on one card, microseconds: (name, time, on the
# Tensix cores); HIST is its two commands together. The step is the X280 thread's own
# work. The V100's whole frame.
PASSES = [
    ("COUNT", 1157, True),
    ("SUM", 60, True),
    ("HIST", 153, True),
    ("step", 633, False),
    ("PLACE", 101, True),
    ("SETUP", 1994, True),
    ("FILL", 2104, True),
]
FRAME_MICROSECONDS = 6202
V100_MICROSECONDS = 4420

PICTURES = [
    ("raster-sponza", "Sponza, rasterized", "3.7 ms · V100 2.3 ms"),
    ("raster-bistro", "Bistro, rasterized", "6.3 ms · V100 6.0 ms"),
    ("raytrace-sponza", "Sponza, ray traced", "298 ms · V100 102 ms"),
    ("raytrace-bistro", "Bistro, ray traced", "470 ms · V100 211 ms"),
]
# (program, its condition, Komorebi, beside it): whole tokens for the decode rows, a
# loop of 32 calls for the kernels, each side at its fastest passing setting.
ROWS = [
    ("Qwen3-1.7B decode", "bf16, whole tokens", "101.9 tokens/s", "official stack 88.3"),
    (
        "Qwen3-1.7B decode",
        "bfp8/bfp4 weights",
        "157.2 tokens/s",
        "official stack 103.0",
    ),
    ("Matrix product", "1024 cubed", "43.2 µs", "TT-NN 92.4 µs"),
    ("SwiGLU", "256 × 256 × 512", "32.2 µs", "TT-NN 39.4 µs"),
    ("Attention", "2 × 8 × 1024 × 128", "147.6 µs", "TT-NN 429.2 µs"),
]

# (title, colour, mark, language, the file its lines are from, lines); a line is
# (text, highlight colour or "skip", note, note colour).
CARDS = [
    (
        "a command runs a handler",
        "a",
        "solid",
        "Python",
        "language/dense_class/linear_class.py",
        [
            ("@km.handler", None, "", None),
            (
                "def split(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):",
                None,
                "",
                None,
            ),
            ("    ...", "skip", "this core's part of K: k0, k1, first", "mute"),
            ("    for xb, wb in zip(", "d", "two streams of tiles from DRAM", "d"),
            (
                "        km.Stream(x.tiles[0:1, k0:k1], block=self.CHUNK),",
                "d",
                "",
                None,
            ),
            (
                "        km.Stream(w.tiles[k0:k1, first : first + self.WIDTH],",
                "d",
                "",
                None,
            ),
            ("                  block=self.W_BLOCK),", "d", "", None),
            ("    ):", "d", "", None),
            (
                "        self.sums += self.project(xb, wb)",
                None,
                "added into this core's field",
                "mute",
            ),
        ],
    ),
    (
        "cores add their parts up a tree",
        "r",
        "solid",
        "Python",
        "the same handler",
        [
            (
                '    whole = km.reduce(self.sums, form="tree", group=self.SPLIT)',
                "r",
                "",
                None,
            ),
            ("    if self.rank == self.leader:", None, "", None),
            (
                "        y.tiles[0:1, first : first + self.WIDTH].write(whole)",
                None,
                "",
                None,
            ),
        ],
    ),
    (
        "a core casts to its row or column",
        "b",
        "solid",
        "Python",
        "03_broadcast/as_class",
        [
            (
                'row = km.Broadcast(self, along="row", window=self.WINDOW)',
                None,
                "",
                None,
            ),
            ("...", "skip", "", None),
            ("if row.sender():", None, "", None),
            (
                "    along_row = row.cast(rows.pages(first_row, self.PAGES).read())",
                "b",
                "",
                None,
            ),
            ("else:", None, "", None),
            ("    along_row = row.arrive()", "b", "", None),
        ],
    ),
    (
        "a message carries a thread to another core",
        "m",
        "dash",
        "Python",
        "05_forward/as_class",
        [
            ("for message in self.arrive():", "m", "", None),
            ("    ...", "skip", "", None),
            ("    self.forward(message.state % self.size, message)", "m", "", None),
        ],
    ),
    (
        "a thread on the card; a core asks it",
        "w",
        "both",
        "C, C++",
        "raster thread, 10_x280_ports",
        [
            (
                "int error = kmx_submit(command, handle);",
                "a",
                "the thread submits a command",
                "a",
            ),
            (
                "const uint32_t count = port.next(first);",
                "w",
                "a core asks an X280 core",
                "w",
            ),
        ],
    ),
]
STEP, HEADER = 18, 27


def jpeg_uri(path, side):
    image = Image.open(path).convert("RGB").resize((side, side), Image.LANCZOS)
    data = io.BytesIO()
    image.save(data, "JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(data.getvalue()).decode()


class Svg:
    def __init__(self, c):
        self.c, self.out = c, []

    def add(self, text):
        self.out.append(text)

    def rect(self, x, y, w, h, fill, stroke="none", rx=4, sw=1):
        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def text(self, x, y, s, size=13, fill=None, weight=400, mono=False, anchor="start"):
        family = MONO if mono else SANS
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" '
            f'font-size="{size}" font-weight="{weight}" '
            f'fill="{fill or self.c["fg"]}" text-anchor="{anchor}" '
            f'xml:space="preserve">{escape(s)}</text>'
        )

    def line(self, x1, y1, x2, y2, stroke, sw=1.5, dash=None, head=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        extra += f' marker-end="url(#{head})"' if head else ""
        self.add(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}"{extra}/>'
        )

    def path(self, points, stroke, sw=1.5, head=None):
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in points)
        extra = f' marker-end="url(#{head})"' if head else ""
        self.add(
            f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}" '
            f'stroke-linejoin="round"{extra}/>'
        )


HEAD = {"a": "ha", "b": "hb", "m": "hp", "w": "hw", "g": "hg", "r": "hr", "d": "hd"}


def arrow(s, x1, y1, x2, y2, colour, kind="solid", sw=2.0):
    extra = ' stroke-dasharray="4 3"' if kind == "dash" else ""
    extra += f' marker-end="url(#{HEAD[colour]})"'
    if kind == "both":
        extra += f' marker-start="url(#{HEAD[colour]})"'
    s.add(
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{s.c[colour]}" stroke-width="{sw}"{extra}/>'
    )


PW, PH, TW, TH, XW = 14, 22, 12, 19, 34
GAP_L, GAP_R, GAP_D = 18, 10, 14  # panel 0 to the X280 column; to its DRAM; to panel 1
GRID_W, GRID_H = 266, 10 * PH


def grid(s, x, y):
    """Card 0, every way a core talks drawn once, no arrow over another or over a
    label; returns the link cores' right edge and middle."""
    c = s.c
    p0 = x + PW  # panel 0's first column
    xb = p0 + 5 * PW + TW + GAP_L  # the X280 column
    dram1 = xb + XW + GAP_R
    p1 = dram1 + TW + GAP_D  # panel 1's first column

    def at(index):
        row, k = divmod(index, 12)
        return (p0 if k < 6 else p1) + (k % 6) * PW + TW / 2, y + row * PH + TH / 2

    for row in range(10):
        ry = y + row * PH
        for px in (x, dram1):
            s.rect(px, ry, TW, TH, c["dram"], rx=2)
        for k in range(12):
            px = at(row * 12 + k)[0] - TW / 2
            if row * 12 + k >= 118:
                s.rect(px, ry, TW, TH, c["gsoft"], c["g"], rx=2, sw=1.3)
            else:
                s.rect(px, ry, TW, TH, c["afill"], c["aline"], rx=2, sw=0.8)
    x280 = {}
    for row, name, kind in ((1, "0", "a"), (3, "2", "w"), (5, "3", "w"), (7, "1", "w")):
        ry = y + row * PH - 1
        soft, edge = (c["asoft"], c["a"]) if kind == "a" else (c["wsoft"], c["w"])
        s.rect(xb, ry, XW, TH + 2, soft, edge, rx=3, sw=1.4)
        s.text(xb + XW / 2, ry + 14, f"X280 {name}", 8.5, edge, 700, anchor="middle")
        x280[name] = ry + (TH + 2) / 2
    # the threads' rings to the dispatcher: a line of their own, right of the column
    bus, top = xb + XW + GAP_R / 2, x280["0"] + 7
    s.line(bus, x280["1"], bus, top, c["w"], 1.4)
    for name in ("2", "3", "1"):
        s.line(xb + XW, x280[name], bus, x280[name], c["w"], 1.4)
    s.line(xb + XW, top, bus, top, c["w"], 1.4)
    # a command to both panels; the right one passes over the DRAM column
    arrow(s, xb - 1, x280["0"], p0 + 5 * PW + TW + 2, x280["0"], "a", sw=2.2)
    s.line(xb + XW + 1, x280["0"] - 3, p1 - 3, x280["0"] - 3, c["bg"], 6)
    arrow(s, xb + XW + 1, x280["0"] - 3, p1 - 2, x280["0"] - 3, "a", sw=2.2)
    # streams: DRAM to a core, in both panels
    for px, cores in ((x, (1, 25)), (dram1, (7, 31))):
        for index in cores:
            ax, ay = at(index)
            arrow(s, px + TW / 2, ay, ax + 2, ay, "d")
    # a core and an X280 core, both ways
    for name, index in (("2", 40), ("1", 88)):
        arrow(s, at(index)[0], x280[name], xb - 2, x280[name], "w", "both", 1.8)
    # a message forwarded twice
    for a, b in ((48, 63), (63, 73)):
        arrow(s, *at(a), *at(b), "m", "dash")
    # a broadcast along a row and down a column
    arrow(s, at(96)[0], at(96)[1], at(101)[0] + 4, at(101)[1], "b")
    arrow(s, at(35)[0], at(35)[1], at(107)[0], at(107)[1] + 6, "b")
    # a reduce: three cores' parts into their root
    for sender in (78, 80, 82):
        ax, ay = at(sender)
        bx, by = at(56)
        arrow(s, ax, ay - 3, bx + (ax - bx) * 0.2, by + 9, "r", sw=1.8)
    return p1 + 5 * PW + TW, at(118)[1]


def second_card(s, x, y, w, edge, ly):
    c = s.c
    s.add(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{GRID_H - 3:.1f}" '
        f'rx="5" fill="{c["box"]}" stroke="{c["rule"]}" stroke-width="1.2" '
        'stroke-dasharray="5 4"/>'
    )
    mid = x + w / 2
    s.text(mid, y + 28, "card 1", 13.5, weight=700, anchor="middle")
    s.text(mid, y + 50, "ranks", 12, c["mute"], anchor="middle")
    s.text(mid, y + 66, "118 to 235", 12, c["mute"], anchor="middle")
    s.text(mid, y + 112, "4 links", 12.5, c["g"], 700, anchor="middle")
    s.text(mid, y + 130, "72.5 GB/s", 12, c["g"], 600, anchor="middle")
    s.text(mid, y + 146, "each way", 12, c["g"], 600, anchor="middle")
    arrow(s, edge + 2, ly, x - 2, ly, "g", "both")


def key(s, x, y, w):
    c = s.c
    rows = [
        [
            ("a", "solid", "command"),
            ("d", "solid", "stream"),
            ("r", "solid", "reduce"),
            ("b", "solid", "broadcast"),
        ],
        [("m", "dash", "message"), ("w", "both", "port"), ("g", "both", "link")],
    ]
    for i, items in enumerate(rows):
        widths = [32 + 7.4 * len(name) for _, _, name in items]
        px = x + (w - sum(widths) - 14 * (len(items) - 1)) / 2
        for (colour, kind, name), width in zip(items, widths):
            arrow(s, px, y + 22 * i + 6, px + 23, y + 22 * i + 6, colour, kind)
            s.text(px + 30, y + 22 * i + 10.5, name, 12.5, c["mute"], 600)
            px += width + 14


def timeline(s, x, y, w):
    c = s.c
    label, pre, post = 84, 40, 40
    x0 = x + label + pre + 6
    span = w - label - pre - post - 12
    scale = span / FRAME_MICROSECONDS
    lane = 26
    ys = [y, y + 46, y + 108]
    for name, ly in zip(["host", "X280 thread", "Tensix cores"], ys):
        s.text(x, ly + 18, name, 13, weight=600)
        s.line(x + label, ly + lane / 2, x + w, ly + lane / 2, c["line"], 1)
    s.line(x0, y - 8, x0, ys[2] + lane + 4, c["rule"], 1, dash="2 3")
    s.line(x0 + span, y - 8, x0 + span, ys[2] + lane + 4, c["rule"], 1, dash="2 3")
    s.rect(x + label, ys[0], pre, lane, c["box"], c["rule"], rx=3)
    s.text(x + label + pre / 2, ys[0] + 17.5, "scene", 12, anchor="middle")
    s.rect(x0 + span + 6, ys[0], post, lane, c["box"], c["rule"], rx=3)
    s.text(x0 + span + 6 + post / 2, ys[0] + 17.5, "image", 12, anchor="middle")
    s.rect(x0 + span / 2 - 76, ys[0] + 2, 152, lane - 4, c["bg"], rx=2)
    s.text(
        x0 + span / 2,
        ys[0] + 18,
        "0 round trips, 0 bytes",
        12.5,
        c["mute"],
        anchor="middle",
    )
    s.rect(x0, ys[1] + 9, span, lane - 18, c["wsoft"], rx=2)
    t = 0
    for name, us, tensix in PASSES:
        px, pw = x0 + t * scale, us * scale
        if not tensix:
            s.rect(px, ys[1], pw, lane, c["w"], rx=2)
        else:
            s.rect(
                px + 0.5, ys[2], max(pw - 1, 1.4), lane, c["a"], rx=2 if pw > 6 else 0
            )
            if pw > 30:
                mid = px + pw / 2
                s.text(
                    mid,
                    ys[2] + 17.5,
                    f"{us / 1000:.1f}",
                    12,
                    c["aink"],
                    700,
                    anchor="middle",
                )
                s.text(
                    mid, ys[2] + lane + 14, name, 11.5, c["mute"], 600, anchor="middle"
                )
        t += us
    s.rect(x0 + 2, ys[1] + lane + 4, 236, 17, c["bg"], rx=2)
    step = next(us for _, us, tensix in PASSES if not tensix)
    s.text(
        x0 + 4,
        ys[1] + lane + 17,
        f"7 commands; sort and merge {step / 1000:.2f} ms",
        12,
        c["w"],
        600,
    )
    vy = ys[2] + lane + 28
    s.text(x, vy + 18, "V100 (CUDA)", 13, weight=600)
    s.rect(x0 + 0.5, vy + 2, V100_MICROSECONDS * scale, lane - 4, c["rule"], rx=2)
    s.text(
        x0 + V100_MICROSECONDS * scale + 8,
        vy + 18,
        f"{V100_MICROSECONDS / 1000:.1f} ms",
        13,
        weight=600,
    )


def card(s, x, y, w, title, colour, kind, language, source, lines):
    c = s.c
    h = HEADER + STEP * len(lines) + 7
    s.rect(x, y, w, h, c["box"], c["line"])
    arrow(s, x + 12, y + 14, x + 37, y + 14, colour, kind)
    s.text(x + 46, y + 18.5, title, 13.5, weight=700)
    bw = 14 + 7.4 * len(language)
    s.rect(x + w - 10 - bw, y + 5, bw, 18, c["bg"], c["rule"], rx=9)
    s.text(x + w - 10 - bw / 2, y + 18, language, 11, weight=600, anchor="middle")
    s.text(x + w - 18 - bw, y + 18, source, 11, c["mute"], anchor="end")
    for i, (text, tint, note, ink) in enumerate(lines):
        ly = y + HEADER + STEP * i
        if tint and tint != "skip":
            s.rect(x + 1, ly, w - 2, STEP, c[tint + "soft"], rx=0)
            s.rect(x + 1, ly, 3, STEP, c[tint], rx=0)
        s.text(
            x + 14,
            ly + 13,
            text,
            12.5,
            c["mute"] if tint == "skip" else c["fg"],
            mono=True,
        )
        if note:
            s.text(x + w - 12, ly + 13, note, 11.5, c[ink], 600, anchor="end")
    return y + h


def draw(theme, repo):
    c = THEMES[theme]
    s = Svg(c)
    s.add(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img">'
        "<title>Komorebi: the lines that say each way cores talk, the card doing each "
        "of them and running a frame by itself, and what comes out</title>"
    )
    s.add("<defs>")
    heads = (
        ("ha", "a"),
        ("hb", "b"),
        ("hp", "m"),
        ("hw", "w"),
        ("hg", "g"),
        ("hr", "r"),
        ("hd", "d"),
        ("hm", "rule"),
    )
    for name, colour in heads:
        s.add(
            f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{c[colour]}"/></marker>'
        )
    s.add("</defs>")
    s.rect(0, 0, W, H, c["bg"], rx=8)

    lx, lw = 16, 552
    mx, mw = 588, 408
    rx_, rw = 1016, 248
    top = 56
    s.text(lx, 36, "you write", 22, weight=700)
    s.text(mx, 36, "the card runs it by itself", 22, c["a"], 700)
    s.text(rx_, 36, "what comes out", 22, weight=700)

    y = top
    for spec in CARDS:
        y = card(s, lx, y, lw, *spec) + 6
    s.line(lx + lw + 4, 360, mx - 6, 360, c["rule"], 2, head="hm")
    s.line(mx + mw + 4, 360, rx_ - 6, 360, c["rule"], 2, head="hm")

    s.rect(mx, top, mw, H - top - 16, c["bg"], c["a"], rx=6, sw=1.8)
    gx, gy = mx + 16, top + 42
    s.text(gx, gy - 13, "card 0", 13.5, weight=700)
    s.text(gx + 54, gy - 13, "120 Tensix cores, 4 X280 clusters", 12.5, c["mute"])
    edge, ly = grid(s, gx, gy)
    second_card(s, edge + 28, gy, mx + mw - 16 - edge - 28, edge, ly)
    key(s, mx + 12, gy + GRID_H + 14, mw - 24)
    sy = gy + GRID_H + 14 + 54
    s.text(mx + 16, sy, "the same class on", 12.5, c["g"], 600)
    s.text(mx + 136, sy, "km.Session(cards=[0, 1])", 12.5, c["g"], 700, mono=True)
    ty = gy + GRID_H + 100
    s.line(mx + 16, ty - 12, mx + mw - 16, ty - 12, c["line"], 1)
    s.text(mx + 16, ty + 14, "one rasterizer frame, 10⁶ triangles", 14.5, None, 700)
    s.text(
        mx + mw - 16,
        ty + 14,
        f"{FRAME_MICROSECONDS / 1000:.1f} ms",
        14.5,
        c["a"],
        700,
        anchor="end",
    )
    timeline(s, mx + 16, ty + 40, mw - 32)

    side, cap = 120, 38
    for i, (name, line1, line2) in enumerate(PICTURES):
        px, py = rx_ + (i % 2) * (side + 8), top + (i // 2) * (side + cap + 6)
        uri = jpeg_uri(repo / "docs" / "images" / f"{name}.jpg", 2 * side)
        s.add(f'<image x="{px}" y="{py}" width="{side}" height="{side}" href="{uri}"/>')
        s.rect(px, py, side, side, "none", c["line"], rx=0)
        s.text(px, py + side + 16, line1, 12.5, weight=600)
        s.text(px, py + side + 32, line2, 11.5, c["mute"])
    ny = top + 2 * (side + cap + 6) + 8
    for prog, shape, value, beside in ROWS:
        s.rect(rx_, ny, rw, 44, c["box"], c["line"])
        s.text(rx_ + 10, ny + 18, prog, 13, weight=600)
        s.text(rx_ + 10, ny + 35, shape, 11.5, c["mute"])
        s.text(rx_ + rw - 10, ny + 18, value, 13, c["g"], 700, anchor="end")
        s.text(rx_ + rw - 10, ny + 35, beside, 11.5, c["a"], 600, anchor="end")
        ny += 49
    s.text(rx_, H - 31, "Sponza © 2010 Frank Meinl, Crytek, CC BY 3.0", 10.5, c["mute"])
    s.text(rx_, H - 17, "Bistro © 2017 Amazon Lumberyard, CC BY 4.0", 10.5, c["mute"])
    s.add("</svg>")
    return "\n".join(s.out)


def main():
    repo = Path(__file__).resolve().parents[2]
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else repo / "docs" / "images"
    for theme in THEMES:
        (out / f"teaser-{theme}.svg").write_text(draw(theme, repo) + "\n")


if __name__ == "__main__":
    main()
