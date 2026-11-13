"""Draws the README's top figure, docs/images/teaser-{light,dark}.svg, from one
description: what a user writes, the card running one rasterizer frame, what comes out.

usage: python docs/images/teaser.py    (needs Pillow, for the four thumbnails)

The numbers are copied from committed results: the frame's passes and the pictures'
frame times from projects/raster/README.md and projects/raytrace/README.md (cold
frames), the neural rows from projects/dense and projects/attention against the TT-NN
result files beside them. The code cards are shortened from lines of
examples/01_mapcollect, projects/attention, examples/05_forward, examples/10_x280_ports
and projects/raster/thread: one card for each thing a program says, and the grid draws
each card's traffic in the card's colour.
"""

import base64
import io
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image

W, H = 1280, 676

THEMES = {
    "light": {
        "bg": "#ffffff",
        "fg": "#1f2328",
        "mute": "#59636e",
        "box": "#f6f8fa",
        "line": "#d1d9e0",
        "dram": "#d5dde6",
        "a": "#bc4c00",
        "afill": "#ffc999",
        "asoft": "#fff1e5",
        "aink": "#ffffff",
        "w": "#0969da",
        "wsoft": "#ddf4ff",
        "rule": "#818b98",
        "g": "#1a7f37",
        "b": "#1b7c83",
        "bsoft": "#dff5f5",
        "m": "#8250df",
        "msoft": "#f1e8ff",
    },
    "dark": {
        "bg": "#0d1117",
        "fg": "#e6edf3",
        "mute": "#9198a1",
        "box": "#161b22",
        "line": "#3d444d",
        "dram": "#2d3540",
        "a": "#f0883e",
        "afill": "#6b3a14",
        "asoft": "#2d1a0c",
        "aink": "#0d1117",
        "w": "#4493f8",
        "wsoft": "#0c2d6b",
        "rule": "#6e7681",
        "g": "#3fb950",
        "b": "#39c5cf",
        "bsoft": "#0b2f33",
        "m": "#a371f7",
        "msoft": "#2a1a4d",
    },
}

SANS = "-apple-system, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'DejaVu Sans Mono', monospace"

# What a program says, one card each: (title, language, colour, mark, lines). A line's
# flag: 1 the line the card is about, 2 lines left out. The mark is the arrow the grid
# draws for the card.
CARDS = [
    (
        "a command runs a handler",
        "C++",
        "a",
        "solid",
        [
            ("km::Reader reader;", 0),
            ("reader.run([&](const km_command_t& command) {", 1),
            ("    switch (km::handler(command)) {", 0),
        ],
    ),
    (
        "broadcast to a row or column",
        "C++",
        "b",
        "solid",
        [
            ("km::Broadcast broadcast(", 0),
            ("    reader, command, kArgBroadcast);", 0),
            ("if (broadcast.sender()) broadcast.cast(tiles);", 1),
            ("else broadcast.arrive(tiles);", 1),
        ],
    ),
    (
        "a message finds its core",
        "C++",
        "m",
        "dash",
        [
            ("km::Dyn messages(compute, command);", 0),
            ("messages.send(rank, message);", 1),
            ("messages.forward(next_core, message);", 1),
            ("messages.consume();", 0),
        ],
    ),
    (
        "a core asks an X280 core",
        "C++",
        "w",
        "both",
        [
            ("km::X280Port port(reader);", 0),
            ("const uint32_t count = port.next(first);", 1),
            ("run(first, count);", 0),
        ],
    ),
    (
        "an X280 thread decides",
        "C",
        "w",
        "box",
        [
            ("int error = kmx_submit(command, &handle);", 1),
            ("kmx_par(sort_partition, &partitions);", 0),
            ("...", 2),
            ("error = kmx_pull_run(command, &pull, &event);", 1),
        ],
    ),
]
SOFT = {"a": "asoft", "b": "bsoft", "m": "msoft", "w": "wsoft"}
HEAD = {"a": "ha", "b": "hb", "m": "hp", "w": "hw"}

# One cold frame of 10^6 triangles, microseconds: (name, time, on the Tensix cores,
# commands it takes). The step is the X280 thread's own work. The V100's whole frame.
PASSES = [
    ("COUNT", 3825, True, 1),
    ("SUM", 59, True, 1),
    ("HIST", 160, True, 2),
    ("step", 804, False, 0),
    ("PLACE", 108, True, 1),
    ("SETUP", 5060, True, 1),
    ("FILL", 2108, True, 1),
]
FRAME_US = 12125
V100_US = 4420

PICTURES = [
    ("raster-sponza", "Sponza, rasterized", "7.2 ms · 0.32× V100"),
    ("raster-bistro", "Bistro, rasterized", "15.4 ms · 0.39× V100"),
    ("raytrace-sponza", "Sponza, ray traced", "360 ms · 0.28× V100"),
    ("raytrace-bistro", "Bistro, ray traced", "597 ms · 0.35× V100"),
]

NUMBERS = [
    ("Matrix product", "128 × 896 × 128", "6.3 µs", "7.7× TT-NN"),
    ("Matrix product", "8192 cubed", "7.99 ms", "1.5× TT-NN"),
    ("SwiGLU", "256 × 256 × 512", "42.6 µs", "6.2× TT-NN"),
    ("Attention", "2 × 8 × 1024 × 128", "163 µs", "1.3× TT-NN"),
]


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

    def line(self, x1, y1, x2, y2, stroke, sw=1.5, dash=None, head=None, tail=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        extra += f' marker-end="url(#{head})"' if head else ""
        extra += f' marker-start="url(#{tail})"' if tail else ""
        self.add(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}"{extra}/>'
        )


def mark(s, x, y, colour, kind):
    """The card's arrow, as the grid draws it."""
    c = s.c
    if kind == "box":
        s.rect(x, y - 8, 26, 16, c[SOFT[colour]], c[colour], rx=3, sw=1.5)
        return
    dash = "4 3" if kind == "dash" else None
    both = kind == "both"
    s.line(
        x,
        y,
        x + (26 if both else 24),
        y,
        c[colour],
        1.7 if both else 2.2,
        dash=dash,
        head=HEAD[colour],
        tail=HEAD[colour] if both else None,
    )


def card(s, x, y, w, title, language, colour, kind, lines):
    """One card: its mark, what it says, its language, the lines; returns its bottom."""
    c = s.c
    step = 19
    h = 36 + step * len(lines) + 8
    s.rect(x, y, w, h, c["box"], c["line"])
    mark(s, x + 14, y + 19, colour, kind)
    s.text(x + 50, y + 24, title, 15.5, weight=700)
    bw = 16 + 8.2 * len(language)
    s.rect(x + w - 12 - bw, y + 8, bw, 21, c["bg"], c["rule"], rx=10)
    s.text(x + w - 12 - bw / 2, y + 23, language, 12.5, weight=600, anchor="middle")
    for i, (text, flag) in enumerate(lines):
        ly = y + 36 + step * i
        if flag == 1:
            s.rect(x + 1, ly, w - 2, step, c[SOFT[colour]], rx=0)
            s.rect(x + 1, ly, 3, step, c[colour], rx=0)
        fill = c["mute"] if flag == 2 else c["fg"]
        s.text(x + 14, ly + 14, text, 13, fill, mono=True)
    return y + h


GRID_W, GRID_H = 24 * 14 + 16 + 52, 170


def grid(s, x, y):
    """The card as used: two panels of 6 x 10 Tensix cores, the X280 column, DRAM."""
    c = s.c
    pw, ph, cw, ch = 24, 17, 22, 15
    cols = {}
    cx = x
    for col in [0, 1, 2, 3, 4, 5, 6, "gap", 8, 9, "gap", 11, 12, 13, 14, 15, 16]:
        if col == "gap":
            cx += 8
            continue
        cols[col] = cx
        cx += 52 if col == 8 else pw
    for row in range(10):
        ry = y + row * ph
        for col, px in cols.items():
            if col in (0, 9):
                s.rect(px, ry, cw, ch, c["dram"], rx=2)
            elif col != 8:
                s.rect(px, ry, cw, ch, c["afill"], c["a"], rx=2, sw=0.8)
    xs = cols[8]
    for row, name, kind in ((1, "0", "a"), (3, "2", "w"), (5, "3", "w"), (7, "1", "w")):
        ry = y + row * ph - 2
        soft, edge = (c["asoft"], c["a"]) if kind == "a" else (c["wsoft"], c["w"])
        s.rect(xs + 3, ry, 44, ch + 4, soft, edge, rx=3, sw=1.5)
        s.text(xs + 25, ry + 14, f"X280 {name}", 10, edge, 700, anchor="middle")
    # the threads' rings to the dispatcher
    for row in (3, 5, 7):
        s.line(xs - 1, y + row * ph + 7, xs - 1, y + 1 * ph + 20, c["w"], 1.6)
        s.line(xs - 1, y + row * ph + 7, xs + 3, y + row * ph + 7, c["w"], 1.6)
    # a command to both panels
    ym = y + 1 * ph + 7
    s.line(xs + 3, ym, cols[6] + cw + 3, ym, c["a"], 2.2, head="ha")
    s.line(xs + 47, ym, cols[11] - 3, ym, c["a"], 2.2, head="ha")
    # a broadcast along a row of panel 0 and down a column of panel 1
    yb = y + 8 * ph + 7
    s.line(cols[1] + 11, yb, cols[6] + cw - 2, yb, c["b"], 2.2, head="hb")
    xb = cols[16] + 11
    s.line(xb, y + 2 * ph + 7, xb, y + 9 * ph + 9, c["b"], 2.2, head="hb")
    # a routed message: sent, forwarded, consumed
    hops = [
        (cols[2] + 11, y + 3 * ph + 7),
        (cols[5] + 11, y + 5 * ph + 7),
        (cols[3] + 11, y + 6 * ph + 7),
    ]
    for (x1, y1), (x2, y2) in zip(hops, hops[1:]):
        s.line(x1, y1, x2, y2, c["m"], 2, dash="4 3", head="hp")
    hops = [
        (cols[14] + 11, y + 7 * ph + 7),
        (cols[12] + 11, y + 5 * ph + 7),
        (cols[14] + 11, y + 4 * ph + 7),
    ]
    for (x1, y1), (x2, y2) in zip(hops, hops[1:]):
        s.line(x1, y1, x2, y2, c["m"], 2, dash="4 3", head="hp")
    # Tensix cores ask the threads' X280 cores, and are answered
    s.line(
        xs + 50,
        y + 3 * ph + 7,
        cols[13] + 8,
        y + 3 * ph + 7,
        c["w"],
        2,
        head="hw",
        tail="hw",
    )
    s.line(
        xs,
        y + 5 * ph + 9,
        cols[4] + 14,
        y + 6 * ph + 9,
        c["w"],
        2,
        head="hw",
        tail="hw",
    )


def legend(s, x, y, w):
    c = s.c
    rows = [
        ("t", "Tensix cores, 120"),
        ("a", "dispatcher, X280 cluster 0"),
        ("d", "DRAM"),
        ("w", "threads, X280 clusters 1 to 3"),
    ]
    for i, (kind, head) in enumerate(rows):
        px, py = x + (i % 2) * (w / 2 - 14), y + (i // 2) * 30
        if kind == "t":
            s.rect(px, py + 1, 22, 15, c["afill"], c["a"], rx=2, sw=0.8)
        elif kind == "a":
            s.rect(px, py + 1, 22, 15, c["asoft"], c["a"], rx=3, sw=1.5)
        elif kind == "w":
            s.rect(px, py + 1, 22, 15, c["wsoft"], c["w"], rx=3, sw=1.5)
        else:
            s.rect(px, py + 1, 22, 15, c["dram"], rx=2)
        s.text(px + 32, py + 14, head, 14, weight=600)


def timeline(s, x, y, w):
    c = s.c
    label, pre, post = 118, 48, 48
    x0 = x + label + pre + 6
    span = w - label - pre - post - 12
    scale = span / FRAME_US
    lane = 26
    names = ["host", "X280 thread", "dispatcher", "Tensix cores"]
    ys = [y, y + 44, y + 106, y + 150]
    for name, ly in zip(names, ys):
        s.text(x, ly + 18, name, 14, weight=600)
        s.line(x + label, ly + lane / 2, x + w, ly + lane / 2, c["line"], 1)
    # the frame's extent
    s.line(x0, y - 8, x0, ys[3] + lane + 4, c["rule"], 1, dash="2 3")
    s.line(x0 + span, y - 8, x0 + span, ys[3] + lane + 4, c["rule"], 1, dash="2 3")
    # host: before and after only
    s.rect(x + label, ys[0], pre, lane, c["box"], c["rule"], rx=3)
    s.text(x + label + pre / 2, ys[0] + 18, "scene", 13, anchor="middle")
    s.rect(x0 + span + 6, ys[0], post, lane, c["box"], c["rule"], rx=3)
    s.text(x0 + span + 6 + post / 2, ys[0] + 18, "image", 13, anchor="middle")
    s.rect(x0 + span / 2 - 104, ys[0] + 2, 208, lane - 4, c["bg"], rx=2)
    s.text(
        x0 + span / 2,
        ys[0] + 18,
        "0 round trips, 0 bytes",
        14,
        c["mute"],
        anchor="middle",
    )
    # the thread lives through the frame; its own work is the step
    s.rect(x0, ys[1] + 9, span, lane - 18, c["wsoft"], rx=2)
    t = 0
    commands = 0
    for name, us, tensix, count in PASSES:
        px, pw = x0 + t * scale, us * scale
        for k in range(count):
            s.line(px + 3 * k, ys[2] + 3, px + 3 * k, ys[2] + lane - 3, c["a"], 2)
        commands += count
        if not tensix:
            s.rect(px, ys[1], pw, lane, c["w"], rx=2)
            s.text(
                x0 + 4,
                ys[1] + lane + 17,
                f"{us / 1000:.2f} ms on 4 cores: sort, merge",
                14,
                c["w"],
                600,
            )
        else:
            round_ = 2 if pw > 6 else 0
            s.rect(px + 0.5, ys[3], max(pw - 1, 1.2), lane, c["a"], rx=round_)
            if pw > 40:
                label_ = f"{name} {us / 1000:.1f}" if pw > 70 else f"{us / 1000:.1f}"
                s.text(
                    px + pw / 2,
                    ys[3] + 18,
                    label_,
                    13.5,
                    c["aink"],
                    700,
                    anchor="middle",
                )
            if 40 < pw <= 70:
                s.text(px + pw / 2, ys[3] + lane + 16, name, 13, anchor="middle")
        t += us
    # the commands, written over the lane where COUNT leaves room
    mid = x0 + 3825 * scale / 2
    s.rect(mid - 48, ys[2] + 2, 96, lane - 4, c["bg"], rx=2)
    s.text(mid, ys[2] + 18, f"{commands} commands", 14, c["a"], 600, anchor="middle")
    # the reference's whole frame, at the same scale
    vy = ys[3] + lane + 36
    s.text(x, vy + 18, "V100 (CUDA)", 14, weight=600)
    s.rect(x0 + 0.5, vy + 2, V100_US * scale, lane - 4, c["rule"], rx=2)
    s.text(
        x0 + V100_US * scale + 10, vy + 18, f"{V100_US / 1000:.1f} ms", 14, weight=600
    )
    return vy + lane


def draw(theme, repo):
    c = THEMES[theme]
    s = Svg(c)
    s.add(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img">'
        "<title>Komorebi: kernels and threads you write, a card that runs the whole "
        "program by itself, and what comes out</title>"
    )
    s.add("<defs>")
    heads = (("ha", "a"), ("hb", "b"), ("hp", "m"), ("hw", "w"), ("hm", "rule"))
    for name, colour in ((n, c[k]) for n, k in heads):
        s.add(
            f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{colour}"/></marker>'
        )
    s.add("</defs>")
    s.rect(0, 0, W, H, c["bg"], rx=8)

    lx, lw = 16, 376
    mx, mw = 428, 536
    rx_, rw = 1000, 264
    top = 56
    s.text(lx, 36, "you write", 22, weight=700)
    s.text(mx, 36, "the card runs it by itself", 22, c["a"], 700)
    s.text(rx_, 36, "what comes out", 22, weight=700)

    # left: what a program says, one card each
    y = top
    for title, language, colour, kind, lines in CARDS:
        y = card(s, lx, y, lw, title, language, colour, kind, lines) + 10

    # arrows between the columns
    s.line(lx + lw + 6, 330, mx - 8, 330, c["rule"], 2, head="hm")
    s.line(mx + mw + 6, 330, rx_ - 8, 330, c["rule"], 2, head="hm")

    # centre: the card, then one frame over time
    s.rect(mx, top, mw, H - top - 16, c["bg"], c["a"], rx=6, sw=1.8)
    grid(s, mx + (mw - GRID_W) / 2, top + 18)
    legend(s, mx + 34, top + 18 + GRID_H + 14, mw - 30)
    ty = top + 18 + GRID_H + 14 + 60 + 22
    s.line(mx + 16, ty - 14, mx + mw - 16, ty - 14, c["line"], 1)
    s.text(mx + 16, ty + 14, "one frame, 10⁶ triangles", 16, weight=700)
    s.text(
        mx + mw - 16,
        ty + 14,
        f"{FRAME_US / 1000:.1f} ms",
        16,
        c["a"],
        700,
        anchor="end",
    )
    timeline(s, mx + 16, ty + 44, mw - 32)

    # right: pictures and numbers
    side, cap = 128, 38
    for i, (name, line1, line2) in enumerate(PICTURES):
        px, py = rx_ + (i % 2) * (side + 8), top + (i // 2) * (side + cap + 6)
        uri = jpeg_uri(repo / "docs" / "images" / f"{name}.jpg", 2 * side)
        s.add(f'<image x="{px}" y="{py}" width="{side}" height="{side}" href="{uri}"/>')
        s.rect(px, py, side, side, "none", c["line"], rx=0)
        s.text(px, py + side + 16, line1, 13, weight=600)
        s.text(px, py + side + 32, line2, 12, c["mute"])
    ny = top + 2 * (side + cap + 6) + 8
    for prog, shape, value, ratio in NUMBERS:
        s.rect(rx_, ny, rw, 46, c["box"], c["line"])
        s.text(rx_ + 10, ny + 19, prog, 13.5, weight=600)
        s.text(rx_ + 10, ny + 37, shape, 12, c["mute"])
        s.text(rx_ + rw - 10, ny + 20, value, 15, c["a"], 700, mono=True, anchor="end")
        s.text(rx_ + rw - 10, ny + 38, ratio, 13, c["g"], 700, anchor="end")
        ny += 51
    s.text(rx_, H - 30, "Sponza © 2010 Frank Meinl, Crytek, CC BY 3.0", 10.5, c["mute"])
    s.text(rx_, H - 15, "Bistro © 2017 Amazon Lumberyard, CC BY 4.0", 10.5, c["mute"])
    s.add("</svg>")
    return "\n".join(s.out)


def main():
    repo = Path(__file__).resolve().parents[2]
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else repo / "docs" / "images"
    for theme in THEMES:
        (out / f"teaser-{theme}.svg").write_text(draw(theme, repo) + "\n")


if __name__ == "__main__":
    main()
