"""Draws the README's top figure, docs/images/teaser-{light,dark}.svg, from one
description: what a user writes, the card running one rasterizer frame, what comes out.

usage: python docs/images/teaser.py    (needs Pillow, for the four thumbnails)

The numbers are copied from committed results: the frame's passes and the pictures'
frame times from projects/raster/README.md and projects/raytrace/README.md (first
frames), the neural rows from projects/dense and projects/attention against the TT-NN
result files beside them. The code panes quote the rasterizer's files, lines left out.
"""

import base64
import io
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image

W, H = 1280, 700

THEMES = {
    "light": dict(
        bg="#ffffff",
        fg="#1f2328",
        mute="#59636e",
        box="#f6f8fa",
        line="#d1d9e0",
        cell="#eaeef2",
        dram="#d5dde6",
        a="#bc4c00",
        afill="#ffc999",
        asoft="#fff1e5",
        aink="#ffffff",
        w="#0969da",
        wsoft="#ddf4ff",
        wink="#ffffff",
        rule="#818b98",
        g="#1a7f37",
    ),
    "dark": dict(
        bg="#0d1117",
        fg="#e6edf3",
        mute="#9198a1",
        box="#161b22",
        line="#3d444d",
        cell="#21262d",
        dram="#2d3540",
        a="#f0883e",
        afill="#6b3a14",
        asoft="#2d1a0c",
        aink="#0d1117",
        w="#4493f8",
        wsoft="#0c2d6b",
        wink="#0d1117",
        rule="#6e7681",
        g="#3fb950",
    ),
}

SANS = "-apple-system, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'DejaVu Sans Mono', monospace"

KERNEL = [
    ("void kernel_main() {", 0),
    ("    km::Compute cp;", 0),
    ("    cp.run([&](const km_record_t& r) {", 1),
    ("        if (km::seg(r) >= pass_k::kSegLanes) {", 0),
    ("            pass_k::setup_compute_record(cp, r);", 0),
    ("        } else {", 0),
    ("            fill_k::fill_compute_record(r);", 0),
    ("        }", 0),
    ("    });", 0),
    ("}", 0),
]
THREAD = [
    ("record(&r, RF_SEG_COUNT, f);", 1),
    ("r.a0 = f->w[RF_W_MESH];", 0),
    ("uint32_t code = run(&r);", 0),
    ("...", 2),
    ("if (kmx_par(sort_part, &p) != KMX_OK ||", 1),
    ("    kmx_par(merge_part, &p) != KMX_OK) {", 0),
    ("...", 2),
    ("record(&r, RF_SEG_FILL, f);", 1),
]
HOST = [
    ("prog = km.Program(", 1),
    ('    "raster_frame",', 0),
    ("    kernels={...}, buffers=[...],", 0),
    (")", 0),
    ("prog.thread(", 1),
    ("    cluster=CLUSTER,", 0),
    ('    sources=[THREAD / "frame.c", ...],', 0),
    ("    regions=regions(scene, workers),", 0),
    ("    deadline=10.0,", 0),
    (")", 0),
]

# One frame of 10^6 triangles, first frame, microseconds; the V100's whole frame.
PASSES = [
    ("COUNT", 3825, "t"),
    ("SUM", 59, "t"),
    ("HIST", 160, "t"),
    ("step", 804, "x"),
    ("PLACE", 108, "t"),
    ("SETUP", 5060, "t"),
    ("FILL", 2108, "t"),
]
FRAME_US = 12125
V100_US = 4420

PICTURES = [
    ("raster-sponza", "Sponza, rasterized", "7.2 ms \u00b7 0.32 of a V100"),
    ("raster-bistro", "Bistro, rasterized", "15.4 ms \u00b7 0.39 of a V100"),
    ("raytrace-sponza", "Sponza, ray traced", "360 ms \u00b7 0.28 of a V100"),
    ("raytrace-bistro", "Bistro, ray traced", "597 ms \u00b7 0.35 of a V100"),
]

NUMBERS = [
    ("Matrix product", "128 \u00d7 896 \u00d7 128", "6.3 \u00b5s", "7.7\u00d7 TT-NN"),
    ("Matrix product", "8192 cubed", "7.99 ms", "1.5\u00d7 TT-NN"),
    ("SwiGLU", "256 \u00d7 256 \u00d7 512", "42.6 \u00b5s", "6.2\u00d7 TT-NN"),
    (
        "Attention",
        "2 \u00d7 8 \u00d7 1024 \u00d7 128",
        "163 \u00b5s",
        "1.3\u00d7 TT-NN",
    ),
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

    def rect(self, x, y, w, h, fill, stroke="none", rx=4, sw=1, dash=None, op=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        extra += f' opacity="{op}"' if op else ""
        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{extra}/>'
        )

    def text(self, x, y, s, size=13, fill=None, weight=400, mono=False, anchor="start"):
        family = MONO if mono else SANS
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill or self.c["fg"]}" '
            f'text-anchor="{anchor}" xml:space="preserve">{escape(s)}</text>'
        )

    def line(self, x1, y1, x2, y2, stroke, sw=1.5, dash=None, head=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        extra += f' marker-end="url(#{head})"' if head else ""
        self.add(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}"{extra}/>'
        )


def code(s, x, y, w, title, path, lines):
    """A code pane; returns its bottom. Flags: 1 a line the model shows, 2 left out."""
    c = s.c
    h = 44 + 15 * len(lines) + 8
    s.rect(x, y, w, h, c["box"], c["line"])
    s.text(x + 14, y + 19, title, 13, weight=600)
    s.text(x + 14, y + 34, path, 10.5, c["mute"], mono=True)
    for i, (text, flag) in enumerate(lines):
        ly = y + 44 + 15 * i
        if flag == 1:
            s.rect(x + 1, ly, w - 2, 15, c["asoft"], rx=0)
            s.rect(x + 1, ly, 3, 15, c["a"], rx=0)
        fill = c["mute"] if flag == 2 else c["fg"]
        s.text(x + 14, ly + 11.5, text, 11, fill, mono=True)
    return y + h


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
    width = cx - x
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
    # the three threads' rings to the dispatch kernel
    for row in (3, 5, 7):
        s.line(xs - 1, y + row * ph + 7, xs - 1, y + 1 * ph + 20, c["w"], 1.6)
    for row in (3, 5, 7):
        s.line(xs - 1, y + row * ph + 7, xs + 3, y + row * ph + 7, c["w"], 1.6)
    # the record to both panels
    ym = y + 1 * ph + 7
    s.line(xs + 3, ym, cols[6] + cw + 3, ym, c["a"], 2.2, head="ha")
    s.line(xs + 47, ym, cols[11] - 3, ym, c["a"], 2.2, head="ha")
    # a value crossing core to core
    s.line(
        cols[2] + 11,
        y + 4 * ph + 8,
        cols[5] + 6,
        y + 7 * ph + 5,
        c["fg"],
        1.8,
        dash="4 3",
        head="hf",
    )
    s.line(
        cols[15] + 11,
        y + 7 * ph + 8,
        cols[12] + 16,
        y + 4 * ph + 11,
        c["fg"],
        1.8,
        dash="4 3",
        head="hf",
    )
    return width, 10 * ph


def legend(s, x, y, w):
    c = s.c
    rows = [
        ("t", "120 Tensix cores", "resident kernels; a record names a team"),
        ("a", "X280 cluster 0", "the dispatch kernel issues the records"),
        ("v", "values cross core to core", "lanes, ports, items that move"),
        ("w", "X280 clusters 1 to 3", "the program's C threads, 12 cores"),
    ]
    for i, (kind, head, body) in enumerate(rows):
        px, py = x + (i % 2) * (w / 2), y + (i // 2) * 42
        if kind == "t":
            s.rect(px, py + 1, 20, 14, c["afill"], c["a"], rx=2, sw=0.8)
        elif kind == "a":
            s.rect(px, py + 1, 20, 14, c["asoft"], c["a"], rx=3, sw=1.5)
        elif kind == "w":
            s.rect(px, py + 1, 20, 14, c["wsoft"], c["w"], rx=3, sw=1.5)
        else:
            s.line(px, py + 8, px + 18, py + 8, c["fg"], 1.8, dash="4 3", head="hf")
        s.text(px + 30, py + 12, head, 12.5, weight=600)
        s.text(px + 30, py + 28, body, 11.5, c["mute"])


def timeline(s, x, y, w):
    c = s.c
    label, pre, post = 112, 42, 42
    x0 = x + label + pre + 6
    span = w - label - pre - post - 12
    scale = span / FRAME_US
    lane = 24
    names = ["host", "X280 thread", "dispatch: 7 records", "120 Tensix cores"]
    ys = [y, y + 34, y + 84, y + 118]
    for name, ly in zip(names, ys):
        s.text(x, ly + 16, name, 12, c["mute"])
        s.line(x + label, ly + lane / 2, x + w, ly + lane / 2, c["line"], 1)
    # the frame's extent
    s.line(x0, y - 8, x0, ys[3] + lane + 4, c["rule"], 1, dash="2 3")
    s.line(x0 + span, y - 8, x0 + span, ys[3] + lane + 4, c["rule"], 1, dash="2 3")
    # host: before and after only
    s.rect(x + label, ys[0], pre, lane, c["box"], c["rule"], rx=3)
    s.text(x + label + pre / 2, ys[0] + 16, "scene", 11, anchor="middle")
    s.rect(x0 + span + 6, ys[0], post, lane, c["box"], c["rule"], rx=3)
    s.text(x0 + span + 6 + post / 2, ys[0] + 16, "image", 11, anchor="middle")
    s.rect(x0 + span / 2 - 116, ys[0] + 2, 232, lane - 4, c["bg"], rx=2)
    s.text(
        x0 + span / 2,
        ys[0] + 16,
        "not in the frame: 0 round trips, 0 bytes",
        11.5,
        c["mute"],
        anchor="middle",
    )
    # the thread lives through the frame; its own work is the step
    s.rect(x0, ys[1] + 8, span, lane - 16, c["wsoft"], rx=2)
    t = 0
    for name, us, kind in PASSES:
        px, pw = x0 + t * scale, us * scale
        s.line(px, ys[2] + 3, px, ys[2] + lane - 3, c["a"], 2)
        if kind == "x":
            s.rect(px, ys[1], pw, lane, c["w"], rx=2)
            s.text(
                x0 + 4,
                ys[1] + lane + 15,
                "the step, 0.80 ms on four cores: tile costs, sort, merge",
                11.5,
                c["w"],
                600,
            )
        else:
            s.rect(
                px + 0.5, ys[3], max(pw - 1, 1.2), lane, c["a"], rx=2 if pw > 6 else 0
            )
            if pw > 46:
                s.text(
                    px + pw / 2,
                    ys[3] + 16,
                    f"{name} {us / 1000:.1f}",
                    11,
                    c["aink"],
                    600,
                    anchor="middle",
                )
        t += us
    mid = x0 + (3825 + 59 + 160 + 804 / 2) * scale
    s.text(
        mid, ys[3] + lane + 15, "SUM, HIST, PLACE: 0.33", 11, c["mute"], anchor="middle"
    )
    s.text(x + w, ys[3] + lane + 15, "times in ms", 11, c["mute"], anchor="end")
    # the reference's whole frame, at the same scale
    vy = ys[3] + lane + 30
    s.text(x, vy + 16, "V100, same frame", 12, c["mute"])
    s.rect(x0 + 0.5, vy + 3, V100_US * scale, lane - 6, c["rule"], rx=2)
    s.text(x0 + V100_US * scale + 8, vy + 16, "4.4 (CUDA)", 11, c["mute"])
    return vy + lane + 6


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
    for name, colour in (
        ("ha", c["a"]),
        ("hf", c["fg"]),
        ("hw", c["w"]),
        ("hm", c["rule"]),
    ):
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
    top = 78
    heads = [
        (
            lx,
            "you write",
            "kernels in C++, threads in C, a descriptor in Python",
            c["fg"],
        ),
        (
            mx,
            "the card runs it by itself",
            "one resident program on 120 Tensix and 16 X280 cores",
            c["a"],
        ),
        (rx_, "what comes out", "whole frames, read back from the card", c["fg"]),
    ]
    for x, head, sub, colour in heads:
        s.text(x, 34, head, 21, colour, 700)
        s.text(x, 56, sub, 13, c["mute"])

    # left: the three things a user writes
    y = code(
        s,
        lx,
        top,
        lw,
        "a kernel on every Tensix core",
        "projects/raster/kernels/frame_compute.cpp",
        KERNEL,
    )
    y = code(
        s,
        lx,
        y + 12,
        lw,
        "a thread on the X280",
        "projects/raster/thread/frame.c",
        THREAD,
    )
    y = code(
        s,
        lx,
        y + 12,
        lw,
        "the program, from Python",
        "projects/raster/card_thread.py",
        HOST,
    )
    s.text(lx, H - 10, "the rasterizer's own files, lines left out", 11, c["mute"])

    # arrows between the columns
    s.line(lx + lw + 6, 330, mx - 8, 330, c["rule"], 2, head="hm")
    s.line(mx + mw + 6, 330, rx_ - 8, 330, c["rule"], 2, head="hm")

    # centre: the card, then one frame over time
    s.rect(mx, top, mw, H - top - 16, c["bg"], c["a"], rx=6, sw=1.8)
    gw, gh = 24 * 14 + 16 + 52, 170
    grid(s, mx + (mw - gw) / 2, top + 18)
    legend(s, mx + 22, top + 18 + gh + 16, mw - 30)
    ty = top + 18 + gh + 16 + 84 + 12
    s.line(mx + 16, ty - 12, mx + mw - 16, ty - 12, c["line"], 1)
    s.text(mx + 16, ty + 12, "one frame, over time", 14, weight=700)
    s.text(
        mx + mw - 16,
        ty + 12,
        "10\u2076 triangles, 1024 \u00d7 1024: 12.1 ms",
        12.5,
        c["mute"],
        anchor="end",
    )
    end = timeline(s, mx + 16, ty + 36, mw - 32)
    s.text(
        mx + 16,
        end + 20,
        "The thread issues every pass and does the irregular step itself; the",
        12,
    )
    s.text(
        mx + 16,
        end + 36,
        "Tensix team does the per-triangle and per-pixel work. Image bit-exact.",
        12,
    )

    # right: pictures and numbers
    side, cap = 128, 34
    for i, (name, line1, line2) in enumerate(PICTURES):
        px, py = rx_ + (i % 2) * (side + 8), top + (i // 2) * (side + cap + 6)
        uri = jpeg_uri(repo / "docs" / "images" / f"{name}.jpg", 2 * side)
        s.add(f'<image x="{px}" y="{py}" width="{side}" height="{side}" href="{uri}"/>')
        s.rect(px, py, side, side, "none", c["line"], rx=0)
        s.text(px, py + side + 14, line1, 11.5, weight=600)
        s.text(px, py + side + 29, line2, 10.5, c["mute"])
    ny = top + 2 * (side + cap + 6) + 12
    s.text(rx_, ny + 6, "the same model, neural work", 13, weight=700)
    ny += 16
    for prog, shape, value, ratio in NUMBERS:
        s.rect(rx_, ny, rw, 44, c["box"], c["line"])
        s.text(rx_ + 10, ny + 18, prog, 12, weight=600)
        s.text(rx_ + 10, ny + 34, shape, 10.5, c["mute"])
        s.text(rx_ + rw - 10, ny + 19, value, 14, c["a"], 700, mono=True, anchor="end")
        s.text(rx_ + rw - 10, ny + 35, ratio, 11, c["g"], 600, anchor="end")
        ny += 49
    s.text(
        rx_, H - 28, "Sponza \u00a9 2010 Frank Meinl, Crytek, CC BY 3.0", 9.5, c["mute"]
    )
    s.text(
        rx_, H - 15, "Bistro \u00a9 2017 Amazon Lumberyard, CC BY 4.0", 9.5, c["mute"]
    )
    s.add("</svg>")
    return "\n".join(s.out)


def main():
    repo = Path(__file__).resolve().parents[2]
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else repo / "docs" / "images"
    for theme in THEMES:
        (out / f"teaser-{theme}.svg").write_text(draw(theme, repo) + "\n")


if __name__ == "__main__":
    main()
