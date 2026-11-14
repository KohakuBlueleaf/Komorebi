"""Draws two figures of the programming model from one description each:
docs/images/breakpoints-{light,dark}.svg ("A thread has a program counter and a
place": six ways cores talk, each on a piece of the P150A tile grid with the lines of
the example that does it) and docs/images/placement-{light,dark}.svg ("Code goes where
the work is": an overlay loaded and called, a handler loaded where a message arrives,
and the code another party decides).

usage: python docs/images/patterns.py [output folder]    (needs Pillow; Noto Sans, if
it is installed, measures the labels)

Every code line is read from the example file it names (path and line number below)
when the figure is drawn, so a line cannot drift from its example; only its
indentation, and where a row says so its trailing comment, are left out. The numbers
quoted in a panel are the page's: overlays.md for the costs of load() and call().
"""

import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import ImageFont

REPO = Path(__file__).resolve().parents[2]
W = 830
BOX_X, BOX_W = 16, 798

THEMES = {
    "light": {
        "bg": "#ffffff",
        "fg": "#1f2328",
        "mute": "#59636e",
        "box": "#f6f8fa",
        "line": "#d1d9e0",
        "cell": "#eaeef2",
        "dram": "#d5dde6",
        "rule": "#818b98",
        "ink": "#ffffff",
        "a": ("#bc4c00", "#ffc999", "#fff1e5"),
        "b": ("#1b7c83", "#a6e0e3", "#dff5f5"),
        "m": ("#8250df", "#d8c5fb", "#f1e8ff"),
        "w": ("#0969da", "#a5d4ff", "#ddf4ff"),
        "r": ("#1a7f37", "#a6e3b5", "#dafbe1"),
        "p": ("#bf3989", "#f7c0de", "#ffeff7"),
        "y": ("#9a6700", "#ffe27a", "#fff8c5"),
        "ys": ("#9a6700", "#fff8c5", "#fff8c5"),
    },
    "dark": {
        "bg": "#0d1117",
        "fg": "#e6edf3",
        "mute": "#9198a1",
        "box": "#161b22",
        "line": "#3d444d",
        "cell": "#21262d",
        "dram": "#2d3540",
        "rule": "#6e7681",
        "ink": "#0d1117",
        "a": ("#f0883e", "#6b3a14", "#2d1a0c"),
        "b": ("#39c5cf", "#14555a", "#0b2f33"),
        "m": ("#a371f7", "#4a2f85", "#2a1a4d"),
        "w": ("#4493f8", "#1d4b99", "#0c2d6b"),
        "r": ("#3fb950", "#1b5e2e", "#0f2d1a"),
        "p": ("#db61a2", "#7a2a57", "#3a1229"),
        "y": ("#d29922", "#6b4e0a", "#3a2d0a"),
        "ys": ("#d29922", "#3a2d0a", "#3a2d0a"),
    },
}

SANS = "-apple-system, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'DejaVu Sans Mono', monospace"
MONO_WIDTH = 0.602  # of the font size: DejaVu Sans Mono and Menlo
ROW = 19  # a line of code

_FONTS = {}


def font(bold):
    """Noto Sans, found by name in the system's font folders; None when it is not
    installed."""
    if bold not in _FONTS:
        name = "NotoSans-Bold.ttf" if bold else "NotoSans-Regular.ttf"
        try:
            _FONTS[bold] = ImageFont.truetype(name, 100)
        except OSError:
            _FONTS[bold] = None
    return _FONTS[bold]


def width(text, size=12, bold=False, mono=False):
    """Width of a text in pixels, a few per cent generous for other fonts. The
    positions of the labels after a text follow from it."""
    if mono:
        return MONO_WIDTH * size * len(text)
    face = font(bold)
    if face is None:
        return (0.6 if bold else 0.56) * size * len(text)
    return face.getlength(text) * size / 100 * 1.03


# The tile grid of the P150A in NoC0 coordinates (the card figure's).
DRAM_BANK = {
    0: "D0 D0 D1 D1 D2 D3 D3 D3 D2 D2 D1 D0",
    9: "D4 D4 D5 D5 D6 D7 D7 D7 D6 D6 D5 D4",
}
X280_ROW = {3: "0", 5: "2", 7: "3", 9: "1"}


def core_index(col, row):
    """Komorebi's core index of the Tensix core at (col, row), or None."""
    if 1 <= col <= 6:
        offset = col - 1
    elif 11 <= col <= 16:
        offset = 6 + col - 11
    else:
        return None
    return (row - 2) * 12 + offset if 2 <= row <= 11 else None


def dram_name(col, row):
    return DRAM_BANK[col].split()[row]


# Code lines, read from the examples.
ELLIPSIS = None


def lines_of(path, spec, drop_comment=(), joint=False):
    """The lines of examples/<path> named in spec (numbers, or ELLIPSIS for a line
    left out), as the texts of their rows with the indentation removed: together
    (joint, for lines that follow each other) or each by itself. A line in
    drop_comment loses its trailing comment."""
    source = (REPO / "examples" / path).read_text().splitlines()
    real = [source[n - 1] for n in spec if n is not ELLIPSIS]
    common = min(len(line) - len(line.lstrip()) for line in real)
    out = []
    for n in spec:
        if n is ELLIPSIS:
            out.append(None)
            continue
        line = source[n - 1]
        text = line[common:] if joint else line.lstrip()
        if n in drop_comment:
            text = re.sub(r"\s+(//|#)\s.*$", "", text)
        out.append(text)
    return out


class Svg:
    def __init__(self, theme, title, markers="abmwrp"):
        self.c = THEMES[theme]
        self.title = title
        self.markers = markers  # the arrowheads this figure uses
        self.layers = [[], [], []]  # panel boxes, tiles and text, arrows and chips
        self.geom = []  # what the upper layer draws, to keep tile numbers clear of it

    def add(self, text, layer=1):
        self.layers[layer].append(text)

    def finish(self, height):
        c = self.c
        defs = "".join(
            f'<marker id="h{name}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{c[name][0]}"/></marker>'
            for name in self.markers
        )
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
            f'viewBox="0 0 {W} {height}" role="img">\n'
            f"<title>{escape(self.title)}</title>\n<defs>{defs}</defs>\n"
            f'<rect width="{W}" height="{height}" fill="{c["bg"]}"/>'
        )
        body = [line for layer in self.layers for line in layer]
        return "\n".join([head, *body, "</svg>"])

    def rect(self, x, y, w, h, fill, stroke="none", rx=4, sw=1, dash=None, layer=1):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{extra}/>',
            layer,
        )

    def text(self, x, y, s, size=12, fill=None, bold=False, mono=False, anchor="start"):
        family = MONO if mono else SANS
        weight = ' font-weight="600"' if bold else ""
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" '
            f'font-size="{size}"{weight} fill="{fill or self.c["fg"]}" '
            f'text-anchor="{anchor}" xml:space="preserve">{escape(s)}</text>'
        )

    def line(self, x1, y1, x2, y2, colour, sw=2.2, dash=None, head=True, tail=False):
        stroke = self.c[colour][0] if colour in self.c else colour
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        extra += (
            f' marker-end="url(#h{colour})"' if head and colour in self.markers else ""
        )
        extra += f' marker-start="url(#h{colour})"' if tail else ""
        self.add(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}"{extra}/>',
            2,
        )
        self.geom.append(
            [(x1 + (x2 - x1) * k / 40, y1 + (y2 - y1) * k / 40) for k in range(41)]
        )

    def curve(self, x1, y1, qx, qy, x2, y2, colour, sw=2.2, dash=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<path d="M{x1:.1f},{y1:.1f} Q{qx:.1f},{qy:.1f} {x2:.1f},{y2:.1f}" '
            f'fill="none" stroke="{self.c[colour][0]}" stroke-width="{sw}" '
            f'marker-end="url(#h{colour})"{extra}/>',
            2,
        )
        points = []
        for k in range(41):
            t = k / 40
            points.append(
                (
                    (1 - t) ** 2 * x1 + 2 * t * (1 - t) * qx + t * t * x2,
                    (1 - t) ** 2 * y1 + 2 * t * (1 - t) * qy + t * t * y2,
                )
            )
        self.geom.append(points)

    def chip(self, cx, cy, s, colour):
        """A pill: white text on the colour, over an arrow."""
        w = width(s, 12, True) + 12
        self.rect(cx - w / 2, cy - 9, w, 18, self.c[colour][0], rx=9, layer=2)
        self.add(
            f'<text x="{cx:.1f}" y="{cy + 4:.1f}" font-family="{SANS}" font-size="12" '
            f'font-weight="600" fill="{self.c["ink"]}" text-anchor="middle" '
            f'xml:space="preserve">{escape(s)}</text>',
            2,
        )
        self.geom.append(
            [
                (cx - w / 2 + dx * w / 10, cy + dy)
                for dx in range(11)
                for dy in (-9, 0, 9)
            ]
        )

    def clear_of_marks(self, box, margin=1.5):
        """True when nothing in the upper layer touches box (x, y, w, h)."""
        x, y, w, h = box
        for points in self.geom:
            for px, py in points:
                if (
                    x - margin <= px <= x + w + margin
                    and y - margin <= py <= y + h + margin
                ):
                    return False
        return True


class Frag:
    """A piece of the tile grid: columns and rows of NoC0 coordinates, numbered."""

    def __init__(self, s, x, y, cols, rows, pitch=38, tile=34, heads=True):
        self.s, self.cols, self.pitch, self.tile = s, cols, pitch, tile
        self.x, self.y = x, y
        self.ypos = {}
        cursor = y
        for row in rows:
            if isinstance(row, tuple):  # ("gap", height, label)
                _, height, label = row
                if label:
                    s.text(x, cursor + height / 2 + 4, label, 12, s.c["mute"])
                cursor += height
                continue
            self.ypos[row] = cursor
            cursor += pitch
        self.bottom = cursor - (pitch - tile)
        self.right = x + len(cols) * pitch - (pitch - tile)
        if heads is True:
            for col in cols:
                s.text(self.cx(col), y - 6, str(col), 12, s.c["mute"], anchor="middle")
        if heads:
            for row in self.ypos:
                s.text(x - 8, self.cy(row) + 4, str(row), 12, s.c["mute"], anchor="end")

    def box(self, col, row):
        return (
            self.x + self.cols.index(col) * self.pitch,
            self.ypos[row],
            self.tile,
            self.tile,
        )

    def cx(self, col):
        return self.x + self.cols.index(col) * self.pitch + self.tile / 2

    def cy(self, row):
        return self.ypos[row] + self.tile / 2

    def draw(self, hot=None):
        """Every tile: plain, or in a colour (hot: {(col, row): colour}). Call it
        after the arrows and chips of the panel are placed: a tile's number is left
        out where one of them lies over it."""
        s, c = self.s, self.s.c
        hot = hot or {}
        for row in self.ypos:
            for col in self.cols:
                px, py, w, h = self.box(col, row)
                colour = hot.get((col, row))
                if col in (7, 10):
                    s.rect(px, py, w, h, "none", c["line"], rx=3, dash="3 3")
                    continue
                if col in DRAM_BANK:
                    s.rect(px, py, w, h, c["dram"], rx=3)
                    s.text(
                        px + w / 2,
                        py + h / 2 + 4,
                        dram_name(col, row),
                        12,
                        c["mute"],
                        anchor="middle",
                    )
                    continue
                if col == 8:
                    name = X280_ROW.get(row)
                    if name is None:
                        continue
                    edge, _, soft = c[colour or "w"]
                    s.rect(px, py, w, h, soft, edge, rx=3, sw=1.5)
                    s.text(
                        px + w / 2,
                        py + h / 2 - 2,
                        "X280",
                        12,
                        edge,
                        True,
                        anchor="middle",
                    )
                    s.text(
                        px + w / 2,
                        py + h / 2 + 12,
                        name,
                        12,
                        edge,
                        True,
                        anchor="middle",
                    )
                    continue
                index = core_index(col, row)
                if index is None:
                    continue
                if colour:
                    edge, fill, _ = c[colour]
                    s.rect(px, py, w, h, fill, edge, rx=3)
                    ink = c["fg"]
                else:
                    s.rect(px, py, w, h, c["cell"], rx=3)
                    ink = c["mute"]
                cx, cy = px + w / 2, py + h / 2
                if s.clear_of_marks((cx - 12, cy - 7, 24, 13)):
                    s.text(cx, cy + 4, str(index), 12, ink, anchor="middle")


def edge_point(rect, other, gap=2.0):
    """Where the line from the centre of rect towards the point other leaves it."""
    x, y, w, h = rect
    cx, cy = x + w / 2, y + h / 2
    dx, dy = other[0] - cx, other[1] - cy
    scale = min(
        (w / 2 + gap) / abs(dx) if dx else 1e9, (h / 2 + gap) / abs(dy) if dy else 1e9
    )
    return cx + dx * scale, cy + dy * scale


def centre(rect):
    return rect[0] + rect[2] / 2, rect[1] + rect[3] / 2


def arrow(s, a, b, colour, dash=None, sw=2.2):
    """An arrow between two tiles, from edge to edge; returns its two ends."""
    ca, cb = centre(a), centre(b)
    start, end = edge_point(a, cb), edge_point(b, ca)
    s.line(*start, *end, colour, sw, dash=dash)
    return start, end


def check(right, limit, what):
    if right > limit:
        print(f"warning: {what!r} ends at {right:.0f}, past {limit:.0f}")


def head_row(s, x, y, limit, colour, tag=None, label=None, gloss=None):
    """A muted tag (file and lines), a bold label in the panel's colour, a gloss."""
    cursor = x
    if tag:
        s.text(cursor, y, tag, 12, s.c["mute"])
        cursor += width(tag) + 12
    if label:
        s.text(cursor, y, label, 12, s.c[colour][0], True)
        cursor += width(label, bold=True) + 8
    if gloss:
        s.text(cursor, y, gloss, 12, s.c["mute"])
        cursor += width(gloss)
    check(cursor, limit, tag or label)


LIMIT = BOX_X + BOX_W - 6
RIGHT = BOX_X + BOX_W


def plan_rows(rows, x, limit):
    """Rows of code as they are drawn. A row is ("head", tag, label, gloss), ("ell",),
    or ("code", text, hot[, label, gloss[, tag]]): a code row with a tag puts the tag
    at its end when it fits, else on a head row above it."""
    out = []
    for row in rows:
        if row[0] != "code":
            out.append(row)
            continue
        _, text, hot, *rest = row
        label, gloss, tag = (rest + [None, None, None])[:3]
        need = x + width(text, mono=True)
        if label:
            need += 14 + width(label, bold=True)
            need += 8 + width(gloss) if gloss else 0
        if tag and need + 24 + width(tag) > limit:
            out.append(("head", tag, label, gloss))
            out.append(("code", text, hot, None, None, None))
        else:
            out.append(("code", text, hot, label, gloss, tag))
    return out


def code_height(rows, x=BOX_X + 322):
    return 34 + len(plan_rows(rows, x, LIMIT)) * ROW + 12


def code_rows(s, x, y, colour, rows, limit=LIMIT, box_right=RIGHT):
    """Draws the rows of plan_rows at (x, y); returns the y after the last."""
    c = s.c
    for row in plan_rows(rows, x, limit):
        if row[0] == "head":
            head_row(s, x, y + 13, limit, colour, *row[1:])
        elif row[0] == "ell":
            s.text(x, y + 14, "...", 12, c["mute"], mono=True)
        else:
            _, text, hot, label, gloss, tag = row
            if hot:
                s.rect(x - 10, y, box_right - (x - 10) - 1, ROW, c[colour][2], rx=0)
                s.rect(x - 10, y, 3, ROW, c[colour][0], rx=0)
            s.text(x, y + 14, text, 12, c["fg"], mono=True)
            cursor = x + width(text, mono=True)
            check(cursor, limit, text)
            if label:
                cursor += 14
                s.text(cursor, y + 14, label, 12, c[colour][0], True)
                cursor += width(label, bold=True) + 8
                if gloss:
                    s.text(cursor, y + 14, gloss, 12, c["mute"])
                    cursor += width(gloss)
            if tag:
                s.text(limit - 4, y + 14, tag, 12, c["mute"], anchor="end")
        y += ROW
    return y


def panel_frame(s, y, height, title, examples, colour):
    c = s.c
    s.rect(BOX_X, y, BOX_W, height, c["box"], c["line"], layer=0)
    s.text(BOX_X + 14, y + 22, title, 14, c[colour][0], True)
    s.text(BOX_X + BOX_W - 14, y + 22, examples, 12, c["mute"], anchor="end")


# ---------------------------------------------------------------------------
# Figure 1: six ways cores talk.
# ---------------------------------------------------------------------------


def panel_move(s, y):
    """1. A thread moves: the message goes to the core it names."""
    c, col = s.c, "m"
    forward = lines_of(
        "05_forward/kernels/compute.cpp", [76, ELLIPSIS, 78, 79], joint=True
    )
    handler = lines_of("14_message_handlers/kernels/compute.cpp", [39])
    rows = [
        ("head", "05_forward · compute.cpp 76-79", "forward", "to the core it names")
    ]
    rows += [
        ("ell",) if text is None else ("code", text, i == 0)
        for i, text in enumerate(forward)
    ]
    rows += [
        ("head", "14_message_handlers · compute.cpp 39", "forward", "with a handler"),
        ("code", handler[0], True),
    ]
    frag = Frag(s, BOX_X + 34, y + 52, [1, 2, 3, 4, 5, 6], [4, 5, 6])
    height = max(code_height(rows), frag.bottom + 30 - y)
    panel_frame(
        s, y, height, "the thread moves", "05_forward, 14_message_handlers", col
    )
    a, b, d = frag.box(1, 6), frag.box(3, 4), frag.box(6, 6)
    arrow(s, a, b, col, dash="5 3")
    arrow(s, b, d, col, dash="5 3")
    s.chip(
        (centre(a)[0] + centre(b)[0]) / 2,
        (centre(a)[1] + centre(b)[1]) / 2,
        "64 B",
        col,
    )
    s.chip(
        (centre(b)[0] + centre(d)[0]) / 2 - 4,
        (centre(b)[1] + centre(d)[1]) / 2 + 6,
        "64 B, with a handler",
        col,
    )
    frag.draw({(1, 6): col, (3, 4): col, (6, 6): col})
    s.text(centre(a)[0], frag.bottom + 16, "start", 12, c["mute"], anchor="middle")
    s.text(centre(d)[0], frag.bottom + 16, "consume", 12, c["mute"], anchor="middle")
    code_rows(s, BOX_X + 322, y + 36, col, rows)
    return height


def panel_reduce(s, y):
    """2. Values move to a fixed slot and are reduced."""
    c, col = s.c, "r"
    emit = lines_of("02_merge_tree/kernels/writer.cpp", [43])
    arrive = lines_of("02_merge_tree/kernels/reader.cpp", [50])
    reduce = lines_of("02_merge_tree/kernels/compute.cpp", [69])
    rows = [
        ("code", emit[0], True, "emit", "into a fixed slot", "writer.cpp 43"),
        ("code", arrive[0], True, "wait", "for the senders' parts", "reader.cpp 50"),
        (
            "code",
            reduce[0],
            True,
            "reduce",
            "adds them in rank order",
            "compute.cpp 69",
        ),
    ]
    x = BOX_X + 262
    frag = Frag(s, BOX_X + 44, y + 52, [3, 4, 5, 6], [4, 5], pitch=44, tile=40)
    height = max(code_height(rows, x), frag.bottom + 30 - y)
    panel_frame(
        s, y, height, "values go to a fixed slot and are reduced", "02_merge_tree", col
    )
    owner = frag.box(3, 5)
    for rank, rise in ((1, 17), (2, 28), (3, 39)):
        sender = frag.box(3 + rank, 5)
        x1, y1 = sender[0] + sender[2] / 2, sender[1] - 1
        x2, y2 = owner[0] + 8.5 + (rank - 1) * 12, owner[1] - 2
        s.curve(x1, y1, (x1 + x2) / 2, y1 - 2 * rise, x2, y2, col, 2.2, dash="5 3")
    s.chip(frag.cx(5) + 4, frag.cy(4) - 6, "part", col)
    frag.draw({(3 + k, 5): col for k in range(4)})
    for k in range(3):  # the owner's slots, one for each sender
        s.rect(
            owner[0] + 4 + k * 12, owner[1] + 4, 9, 9, c["bg"], c[col][0], rx=1, sw=1.2
        )
    s.text(frag.x - 8, owner[1] + owner[3] + 16, "rank", 12, c["mute"], anchor="end")
    for rank in range(4):
        b = frag.box(3 + rank, 5)
        s.text(
            b[0] + b[2] / 2, b[1] + b[3] + 16, str(rank), 12, c["mute"], anchor="middle"
        )
    code_rows(s, x, y + 36, col, rows)
    return height


def panel_cast(s, y):
    """3. One core to a row or a column: a block, and a word."""
    col = "b"
    send = lines_of("03_broadcast/kernels/broadcast.h", [78])
    word = lines_of("15_leader/kernels/reader.cpp", [79, 85], drop_comment=(85,))
    rows = [
        ("head", "03_broadcast · broadcast.h 78", "cast", "a block to the row"),
        ("code", send[0], True),
        (
            "head",
            "15_leader · reader.cpp 79, 85",
            "cast",
            "a word, the same on every core",
        ),
        ("code", word[0], True),
        ("code", word[1], False),
    ]
    frag = Frag(s, BOX_X + 34, y + 52, [1, 2, 3, 4, 5, 6], [4, 5, 6])
    height = max(code_height(rows), frag.bottom + 30 - y)
    panel_frame(
        s, y, height, "one core to a row or a column", "03_broadcast, 15_leader", col
    )
    first, last, top = frag.box(1, 6), frag.box(6, 6), frag.box(3, 4)
    yb = first[1] + first[3] - 6
    s.line(first[0] + first[2] - 8, yb, last[0] + 10, yb, col, 2.4)
    xc = top[0] + top[2] - 6
    s.line(xc, top[1] + top[3] - 8, xc, frag.box(3, 6)[1] + 6, col, 2.0, dash="5 3")
    s.chip(frag.cx(4), frag.cy(5) + 2, "word, 4 B", col)
    s.chip(frag.cx(4), yb - 15, "block", col)
    frag.draw(
        {(1, 6): col, (3, 4): col, **{(k, 6): col for k in range(2, 7)}, (3, 5): col}
    )
    code_rows(s, BOX_X + 322, y + 36, col, rows)
    return height


def panel_x280(s, y):
    """4. A core and its X280 core: tell, ask, a pull."""
    c, col = s.c, "w"
    tell = lines_of("10_x280_ports/kernels/reader.cpp", [96])
    ask = lines_of("10_x280_ports/kernels/reader.cpp", [88])
    pull = lines_of("10_x280_ports/kernels/reader.cpp", [64])
    both = lines_of("18_thread_split/kernels/reader.cpp", [47], drop_comment=(47,))
    rows = [
        ("head", "10_x280_ports · reader.cpp 96, 88, 64"),
        ("code", tell[0], True, "tell", "latest wins"),
        ("code", ask[0], True, "ask", "and wait for the answer"),
        ("code", pull[0], True, "pull", "the next range"),
        (
            "head",
            "18_thread_split · reader.cpp 47",
            "pull",
            "a range, and a word with it",
        ),
        ("code", both[0], True),
    ]
    frag = Frag(s, BOX_X + 34, y + 52, [5, 6, 7, 8], [4, 5], pitch=56, tile=44)
    height = max(code_height(rows), frag.bottom + 30 - y)
    panel_frame(
        s, y, height, "a core and its X280 core", "10_x280_ports, 18_thread_split", col
    )
    core, x280 = frag.box(6, 5), frag.box(8, 5)
    xa, xb = core[0] + core[2] + 3, x280[0] - 3
    ya, yb = core[1] + 14, core[1] + core[3] - 14
    s.line(xa, ya, xb, ya, col)
    s.line(xb, yb, xa, yb, col)
    frag.draw({(6, 5): col, (8, 5): col})
    mid = (xa + xb) / 2
    s.text(mid, ya - 7, "tell, ask", 12, c[col][0], True, anchor="middle")
    s.text(mid, yb + 16, "answer", 12, c[col][0], True, anchor="middle")
    s.text(frag.x, frag.bottom + 18, "16 B each way", 12, c["mute"])
    code_rows(s, BOX_X + 322, y + 36, col, rows)
    return height


def panel_commands(s, y):
    """5. Commands: from the host, from an X280 thread, to a thread."""
    c, col = s.c, "a"
    host_grid = lines_of("13_thread_commands/thread_commands.py", [82])
    host_thread = lines_of("13_thread_commands/thread_commands.py", [88])
    submit = lines_of("19_thread_programs/thread/rounds.c", [17])
    answer = lines_of("13_thread_commands/thread/fold.c", [38])
    rows = [
        (
            "code",
            host_grid[0],
            True,
            "submit",
            "to a core group",
            "13_thread_commands · thread_commands.py 82",
        ),
        (
            "code",
            host_thread[0],
            True,
            "submit",
            "to an X280 thread",
            "13_thread_commands · thread_commands.py 88",
        ),
        (
            "code",
            submit[0],
            True,
            "kmx_submit",
            "from an X280 thread",
            "19_thread_programs · rounds.c 17",
        ),
        (
            "code",
            answer[0],
            True,
            "kmx_program",
            "answers with a word",
            "13_thread_commands · fold.c 38",
        ),
    ]
    x = BOX_X + 308
    height = code_height(rows, x)
    panel_frame(s, y, height, "commands", "13_thread_commands, 19_thread_programs", col)
    frag = Frag(s, BOX_X + 52, y + 76, [5, 6, 7, 8], [3, ("gap", 36, None), 9])
    d0, d1, core = frag.box(8, 3), frag.box(8, 9), frag.box(6, 3)
    host_w, host_h = 58, 22
    x0 = d0[0] + d0[2] / 2
    hy = y + 38
    s.rect(x0 - host_w / 2 + 8, hy, host_w, host_h, c["bg"], c["rule"], rx=4, sw=1.5)
    s.text(x0 + 8, hy + 15, "host", 12, c["fg"], True, anchor="middle")
    s.line(x0 + 10, hy + host_h + 2, x0 + 10, d0[1] - 3, col)
    s.line(d0[0] - 3, d0[1] + d0[3] / 2, core[0] + core[2] + 3, d0[1] + d0[3] / 2, col)
    xa, xb = x0 - 7, x0 + 7
    s.line(xa, d0[1] + d0[3] + 3, xa, d1[1] - 3, col)
    s.line(xb, d1[1] - 3, xb, d0[1] + d0[3] + 3, "w")
    frag.draw({(5, 3): col, (6, 3): col, (8, 3): col, (8, 9): "w"})
    s.text(x0 + 20, hy + host_h + 17, "64 B", 12, c[col][0], True)
    ty = d0[1] + d0[3] + 20
    s.text(frag.x - 8, ty + 8, "4 to 8", 12, c["mute"], anchor="end")
    s.text(x0 + 26, d0[1] + d0[3] / 2 + 4, "dispatcher", 12, c["mute"])
    s.text(x0 + 26, d1[1] + d1[3] / 2 + 4, "a thread", 12, c["mute"])
    s.text(xa - 8, ty + 8, "to the thread", 12, c[col][0], True, anchor="end")
    s.text(xb + 8, ty + 8, "kmx_submit", 12, c["w"][0], True)
    code_rows(s, x, y + 36, col, rows)
    return height


def panel_both(s, y):
    """6. Both data-movement CPUs of a core on one stream between DRAM and L1."""
    c, col = s.c, "p"
    code = lines_of("11_movement/kernels/movement.cpp", [14, 23, 36])
    rows = [
        ("head", "movement.cpp 14, 23, 36"),
        ("code", code[0], True, "movers", "one source runs on both"),
        ("code", code[1], False, "pattern"),
        ("code", code[2], True, "move", "this mover's blocks"),
    ]
    frag = Frag(s, BOX_X + 34, y + 62, [0, 1], [2], pitch=72, tile=48, heads="rows")
    height = max(code_height(rows), frag.bottom + 74 - y)
    panel_frame(
        s,
        y,
        height,
        "both data-movement CPUs of a core on one stream",
        "11_movement",
        col,
    )
    dram, core = frag.box(0, 2), frag.box(1, 2)
    x1, x2 = dram[0] + dram[2] / 2, core[0] + core[2] / 2
    top, bottom = dram[1] - 2, dram[1] + dram[3] + 2
    s.curve(x1, top, (x1 + x2) / 2, top - 38, x2, top, col, 2.2)
    s.curve(x1, bottom, (x1 + x2) / 2, bottom + 38, x2, bottom, col, 2.2, dash="5 3")
    s.chip((x1 + x2) / 2, top - 19, "A, NoC 0", col)
    s.chip((x1 + x2) / 2, bottom + 19, "B, NoC 1", col)
    frag.draw({(1, 2): col})
    by = frag.bottom + 36
    for k in range(8):
        owner = "AAAB"[k % 4]
        bx = frag.x + k * 22
        if owner == "A":
            s.rect(bx, by, 18, 18, c[col][1], c[col][0], rx=3)
        else:
            s.rect(bx, by, 18, 18, c[col][2], c[col][0], rx=3, sw=1.5)
        s.text(bx + 9, by + 13, owner, 12, c["fg"], anchor="middle")
    s.text(frag.x + 8 * 22 + 6, by + 13, "pattern AAAB", 12, c["mute"])
    code_rows(s, BOX_X + 322, y + 36, col, rows)
    return height


def legend(s, y, items, note):
    c = s.c
    x = BOX_X + 4
    for kind, text in items:
        if kind == "tile":
            s.rect(x, y, 22, 16, c["cell"], rx=3)
        elif kind == "dram":
            s.rect(x, y, 22, 16, c["dram"], rx=3)
        elif kind == "x280":
            s.rect(x, y, 22, 16, c["w"][2], c["w"][0], rx=3, sw=1.5)
        elif kind == "part":
            s.rect(x, y, 22, 16, c["y"][1], c["y"][0], rx=3)
        elif kind == "loading":
            s.rect(x, y, 22, 16, c["y"][2], c["y"][0], rx=3, sw=1.5, dash="4 3")
        else:  # a fused-off tile, an empty slot
            s.rect(x, y, 22, 16, "none", c["line"], rx=3, dash="3 3")
        s.text(x + 30, y + 13, text, 12)
        x += 30 + width(text) + 22
    for i, line in enumerate(note):
        s.text(BOX_X + 4, y + 38 + 18 * i, line, 12, c["mute"])
    return y + 38 + 18 * len(note) - 8


def stack(s, panels, top=10, gap=10):
    y = top
    for panel in panels:
        s.geom = []
        y += panel(s, y) + gap
    return y


def figure_patterns(theme):
    panels = [
        panel_move,
        panel_reduce,
        panel_cast,
        panel_x280,
        panel_commands,
        panel_both,
    ]
    s = Svg(theme, "A thread has a program counter and a place")
    bottom = stack(s, panels)
    items = [
        ("tile", "Tensix core, with its core index"),
        ("dram", "DRAM bank"),
        ("x280", "X280 core"),
        ("off", "fused off"),
    ]
    note = [
        "Column and row numbers are NoC0 coordinates. A line of dots is a line left "
        "out; code is copied from the",
        "named file and line, its indentation removed.",
    ]
    end = legend(s, bottom + 2, items, note)
    return s.finish(int(end + 10))


# ---------------------------------------------------------------------------
# Figure 2: code goes where the work is.
# ---------------------------------------------------------------------------


def l1_box(s, x, y, w, slots):
    """A core's L1 as its slots: (text, state), state held, loading or empty."""
    c = s.c
    n = len(slots)
    sw = (w - 12 - (n - 1) * 6) / n
    s.rect(x, y, w, 46, c["bg"], c["rule"], rx=4)
    boxes = []
    for k, (text, state) in enumerate(slots):
        sx, sy, sh = x + 6 + k * (sw + 6), y + 6, 34
        if state == "empty":
            s.rect(sx, sy, sw, sh, "none", c["line"], rx=3, dash="3 3")
        elif state == "loading":
            s.rect(sx, sy, sw, sh, c["y"][2], c["y"][0], rx=3, sw=1.5, dash="4 3")
        else:
            s.rect(sx, sy, sw, sh, c["y"][1], c["y"][0], rx=3)
        s.text(sx + sw / 2, sy + sh / 2 + 4, text, 12, c["fg"], anchor="middle")
        boxes.append((sx, sy, sw, sh))
    return boxes


def overlay_step(s, x, top, heading, held, numbers):
    """One command of an overlay on core 0: the tiles, its L1, the load, the call."""
    c = s.c
    s.text(x, top + 14, heading, 12, c["fg"], True)
    s.text(x + 372, top + 14, numbers, 12, c["y"][0], True, anchor="end")
    frag = Frag(s, x + 24, top + 52, [0, 1], [2], pitch=48, tile=44)
    box = l1_box(s, x + 150, top + 52, 210, [("scale", "held"), ("", "empty")])
    slot = box[0]
    d = frag.box(0, 2)
    if not held:
        x1, y1 = d[0] + d[2] / 2, d[1] + d[3] + 2
        x2, y2 = slot[0] + 22, slot[1] + slot[3] + 2
        s.curve(x1, y1, (x1 + x2) / 2, y1 + 56, x2, y2, "y")
        s.chip((x1 + x2) / 2 + 2, y1 + 28 + 3, "load", "y")
    cx = slot[0] + slot[2] - 24
    s.line(cx, top + 24, cx, slot[1] - 2, "y")
    s.chip(cx, top + 24, "call", "y")
    frag.draw({(1, 2): "y"})
    s.text(x + 360, top + 118, "L1 of core 0", 12, c["mute"], anchor="end")


def panel_load_call(s, y):
    """1. A core loads an overlay from DRAM into a slot of its L1 and calls it."""
    col = "y"
    code = lines_of(
        "12_overlays/kernels/reader.cpp", [14, 17, 25], drop_comment=(14, 17)
    )
    rows = [
        ("head", "12_overlays · reader.cpp 14, 17, 25"),
        ("code", code[0], False, "names it", "nothing moves"),
        (
            "code",
            code[1],
            True,
            "load",
            "into a slot of this core; nothing moves when it is held",
        ),
        ("code", code[2], True, "call", "the part's code, in the slot"),
    ]
    x = BOX_X + 22
    band = 140
    height = 34 + band + len(plan_rows(rows, x, LIMIT)) * ROW + 12
    panel_frame(
        s,
        y,
        height,
        "a core loads an overlay into a slot and calls it",
        "12_overlays",
        col,
    )
    top = y + 34
    overlay_step(
        s,
        BOX_X + 14,
        top,
        "first command: load, then call",
        False,
        "load(): 998 cycles",
    )
    overlay_step(
        s,
        BOX_X + 14 + 392,
        top,
        "next command: the slot holds it",
        True,
        "load(): 72 cycles",
    )
    code_rows(s, x, top + band, col, rows)
    return height


def panel_handler(s, y):
    """2. A message names a handler; its part is loaded where the message arrives."""
    c, col = s.c, "y"
    program = lines_of("20_message_overlays/message_overlays.py", [115])
    part = lines_of("20_message_overlays/kernels/lookup.cpp", [8, 46])
    rows = [
        (
            "code",
            program[0],
            False,
            "parts",
            "each handler's body is a part",
            "message_overlays.py 115",
        ),
        ("code", part[0], False, "the part", None, "lookup.cpp 8"),
        (
            "code",
            part[1],
            True,
            "forward",
            "the next handler is named in the message",
            "lookup.cpp 46",
        ),
    ]
    x = BOX_X + 22
    band = 150
    height = 34 + band + len(plan_rows(rows, x, LIMIT)) * ROW + 12
    panel_frame(
        s,
        y,
        height,
        "a message names its handler, which is loaded where it arrives",
        "20_message_overlays",
        "m",
    )
    top = y + 34
    ox = BOX_X + 14
    s.text(
        ox,
        top + 14,
        "the message reaches core 36 and its handler is not in a slot",
        12,
        c["fg"],
        True,
    )
    frag = Frag(
        s, ox + 24, top + 56, [0, 1, 2, 3], [5], pitch=48, tile=44, heads="rows"
    )
    sender, receiver, d = frag.box(3, 5), frag.box(1, 5), frag.box(0, 5)
    x1, x2 = sender[0] + sender[2] / 2, receiver[0] + receiver[2] / 2
    ytop = sender[1] - 2
    s.curve(x1, ytop, (x1 + x2) / 2, ytop - 52, x2, ytop, "m", dash="5 3")
    s.chip((x1 + x2) / 2, ytop - 26, "64 B, handler value", "m")
    box = l1_box(
        s,
        ox + 268,
        top + 62,
        330,
        [("lookup", "held"), ("value", "loading"), ("", "empty")],
    )
    slot = box[1]
    xa, ya = d[0] + d[2] / 2, d[1] + d[3] + 2
    xb, yb = slot[0] + slot[2] / 2, slot[1] + slot[3] + 2
    s.curve(xa, ya, (xa + xb) / 2, ya + 70, xb, yb, "y")
    s.chip((xa + xb) / 2, ya + 35 + 3, "load", "y")
    frag.draw({(1, 5): "y", (3, 5): "m"})
    s.text(ox + 268, top + 124, "L1 of core 36", 12, c["mute"])
    code_rows(s, x, top + band, col, rows)
    return height


def panel_decided(s, y):
    """3. Another party decides the code: a leader, an X280 core, run.load."""
    c, col = s.c, "y"
    cast = lines_of(
        "17_leader_overlay/kernels/reader.cpp", [155, 165, 76, 77], drop_comment=(165,)
    )
    pull = lines_of("18_thread_split/kernels/reader.cpp", [47, 64], drop_comment=(47,))
    load = lines_of("16_two_programs/two_programs.py", [354], drop_comment=(354,))
    x = BOX_X + 322
    rows_a = [
        ("head", "17_leader_overlay · reader.cpp 155, 165, 76, 77"),
        ("code", cast[0], True, "cast"),
        ("code", cast[1], False),
        ("code", cast[2], False),
        ("code", cast[3], True, "load", "and call it"),
    ]
    rows_b = [
        ("head", "18_thread_split · reader.cpp 47, 64"),
        ("code", pull[0], True, "pull"),
        ("code", pull[1], True, "names it"),
    ]
    rows_c = [
        ("head", "16_two_programs · two_programs.py 354"),
        ("code", load[0], True, "run.load", "another program"),
    ]
    heights = [
        max(len(plan_rows(rows_a, x, LIMIT)) * ROW + 16, 102),
        112,
        102,
    ]
    height = 34 + sum(heights) + 4
    panel_frame(
        s,
        y,
        height,
        "another party decides the code",
        "17_leader_overlay, 18_thread_split, 16_two_programs",
        col,
    )
    sub = y + 34
    # a leader casts the overlay's number to its row
    s.text(
        BOX_X + 14,
        sub + 14,
        "a leader casts the overlay's number to its group",
        12,
        c["fg"],
        True,
    )
    frag = Frag(s, BOX_X + 34, sub + 38, [1, 2, 3, 4, 5, 6], [4])
    leader, last = frag.box(1, 4), frag.box(6, 4)
    yb = leader[1] + leader[3] - 6
    s.line(leader[0] + leader[2] - 8, yb, last[0] + 10, yb, "b", 2.4)
    s.chip(frag.cx(4), frag.cy(4) + 1, "word", "b")
    frag.draw({(1, 4): "b", **{(k, 4): "y" for k in range(2, 7)}})
    s.text(frag.x, frag.bottom + 18, "the word is the overlay's number", 12, c["mute"])
    code_rows(s, x, sub + 8, col, rows_a)
    # an X280 core's answer names it
    sub += heights[0]
    s.line(BOX_X + 12, sub - 2, BOX_X + BOX_W - 12, sub - 2, c["line"], 1, head=False)
    s.text(BOX_X + 14, sub + 14, "an X280 core's answer names it", 12, c["fg"], True)
    frag = Frag(s, BOX_X + 34, sub + 38, [5, 6, 7, 8], [4], pitch=48, tile=44)
    core, x280 = frag.box(6, 4), frag.box(8, 4)
    xa, xb = core[0] + core[2] + 3, x280[0] - 3
    ya, yc = core[1] + 14, core[1] + core[3] - 14
    s.line(xa, ya, xb, ya, "w")
    s.line(xb, yc, xa, yc, "w")
    frag.draw({(6, 4): "y", (8, 4): "w"})
    s.text((xa + xb) / 2, ya - 5, "pull", 12, c["w"][0], True, anchor="middle")
    s.text((xa + xb) / 2, yc + 16, "answer", 12, c["w"][0], True, anchor="middle")
    s.text(
        frag.x,
        frag.bottom + 18,
        "a range, and a word: the overlay's number",
        12,
        c["mute"],
    )
    code_rows(s, x, sub + 8, col, rows_b)
    # run.load makes a group hold another program
    sub += heights[1]
    s.line(BOX_X + 12, sub - 2, BOX_X + BOX_W - 12, sub - 2, c["line"], 1, head=False)
    s.text(
        BOX_X + 14,
        sub + 14,
        "run.load: a group holds another program",
        12,
        c["fg"],
        True,
    )
    frag = Frag(s, BOX_X + 34, sub + 38, [1, 2, 3, 4, 5, 6], [4])
    frag.draw(
        {**{(k, 4): "ys" for k in range(1, 4)}, **{(k, 4): "y" for k in range(4, 7)}}
    )
    left, right = frag.cx(2), frag.cx(5)
    s.text(left, frag.bottom + 18, "sum", 12, c["mute"], anchor="middle")
    s.text(right, frag.bottom + 18, "widen", 12, c["y"][0], True, anchor="middle")
    s.text(
        frag.x + 6 * frag.pitch + 8,
        frag.cy(4) + 4,
        "2.3 to 2.6 µs",
        12,
        c["y"][0],
        True,
    )
    code_rows(s, x, sub + 8, col, rows_c)
    return height


def figure_placement(theme):
    panels = [panel_load_call, panel_handler, panel_decided]
    s = Svg(theme, "Code goes where the work is", markers="bmwy")
    bottom = stack(s, panels)
    items = [
        ("tile", "Tensix core"),
        ("dram", "DRAM bank"),
        ("x280", "X280 core"),
        ("part", "a part in a slot"),
        ("loading", "a part being loaded"),
        ("empty", "an empty slot"),
    ]
    note = [
        "Column and row numbers are NoC0 coordinates. Cycles: medians for a 128-byte "
        "reader part on one core,",
        "AICLK 1,350 MHz. Code is copied from the named file and line, its indentation "
        "removed.",
    ]
    end = legend(s, bottom + 2, items, note)
    return s.finish(int(end + 10))


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "docs" / "images"
    out.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        (out / f"breakpoints-{theme}.svg").write_text(figure_patterns(theme) + "\n")
        (out / f"placement-{theme}.svg").write_text(figure_placement(theme) + "\n")


if __name__ == "__main__":
    main()
