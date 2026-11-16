"""Draws two figures with the helpers of patterns.py, as
docs/images/<name>-{light,dark}.svg: reach ("A thread reaches any DRAM": one kmx_fetch
on the tile grid, the call in time, what it costs on one card and from another island)
and islands ("More than one card": two cards as two islands with flat ranks and the
link cores, islands as a graph, what the link cores carry drawn between two islands
and over a line of three with what starts work on a far core, and the ways a kernel
acts by island beside one written by rank).

usage: python docs/images/islands.py [output folder]    (needs Pillow, as patterns.py)

Every code line is read from the file it names when the figure is drawn. The numbers
are the pages': experiments/x280/bank_fetch/README.md, docs/reference/islands.md,
docs/concepts/islands.md, experiments/islands/link_ops/README.md and the READMEs of
examples 25 and 26.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from patterns import (
    BOX_W,
    BOX_X,
    REPO,
    RIGHT,
    THEMES,
    Frag,
    Lanes,
    Svg,
    card_tiles,
    centre,
    code_rows,
    core_at,
    edge_point,
    lines_of,
    panel_frame,
    stack,
    width,
)

FETCH = "../experiments/x280/bank_fetch/thread/islands.c"


def path(s, points, stroke, sw=2.0, dash=None):
    """Straight pieces through the points with no arrowhead, over the tiles."""
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in points)
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    s.add(
        f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}" '
        f'stroke-linejoin="round"{extra}/>',
        2,
    )


def outline(s, box, colour, sw=2.2, dash=None):
    px, py, w, h = box
    s.rect(px, py, w, h, "none", s.c[colour][0], rx=3, sw=sw, dash=dash, layer=2)


def steps(s, x, y, rows, gap=38):
    """Numbered steps: a pill with the number in the step's colour, a bold line and
    a muted one under it; returns the y after the last."""
    c = s.c
    for number, colour, head, gloss in rows:
        s.chip(x + 10, y + 8, number, colour)
        s.text(x + 28, y + 12, head, 12, c["fg"], True)
        if gloss:
            s.text(x + 28, y + 28, gloss, 12, c["mute"])
        y += gap if gloss else 24
    return y


# ---------------------------------------------------------------------------
# Figure "reach": a thread reaches any DRAM.
# ---------------------------------------------------------------------------


def panel_fetch_card(s, y):
    """One kmx_fetch on the card: the thread, the dispatcher, the copying cores."""
    c = s.c
    frag = Frag(
        s,
        BOX_X + 30,
        y + 46,
        list(range(17)),
        list(range(12)),
        pitch=24,
        tile=21,
        heads=False,
    )
    rows = [
        (
            "1",
            "w",
            "the thread empties its cluster's cache",
            "kmx_evict, 21 µs: from 160 KiB; a sweep below",
        ),
        (
            "2",
            "a",
            "one command to the dispatcher",
            "for cores 0 to 3; sent again if it is busy",
        ),
        (
            "3",
            "p",
            "the cores copy, bank to bank",
            "16 KiB chunks; four cores reach the bank's rate",
        ),
        ("4", "w", "the thread reads its region", "plain loads: the new bytes"),
    ]
    height = frag.bottom + 40 - y
    panel_frame(
        s,
        y,
        height,
        "one call, and Tensix cores bring the range to the thread",
        "08_parallel_thread, experiments/x280/bank_fetch",
        "w",
    )

    def core(index):
        return frag.box(*core_at(index))

    # 3: the copy, from the range in bank D0 through cores 0 to 3 into the region
    src, first, last = frag.box(0, 1), core(0), core(3)
    region, thread = frag.box(9, 9), frag.box(8, 9)
    mid = (first[0] + last[0] + last[2]) / 2
    s.curve(
        src[0] + src[2] + 2, src[1] + 6, mid, src[1] - 22, mid, first[1] - 3, "p", 2.4
    )
    out = (mid, first[1] + first[3] + 3)
    into = (region[0] + 4, region[1] - 3)
    bend = (out[0] - 30, into[1] - 20)
    s.curve(*out, *bend, *into, "p", 2.4)
    s.chip(
        0.25 * out[0] + 0.5 * bend[0] + 0.25 * into[0],
        0.25 * out[1] + 0.5 * bend[1] + 0.25 * into[1],
        "3",
        "p",
    )
    # 2: the thread's command, by its ring to the dispatcher, then to the cores
    d0 = frag.box(8, 3)
    lane = frag.cx(7)
    s.poly(
        [
            (thread[0] - 3, frag.cy(9)),
            (lane, frag.cy(9)),
            (lane, frag.cy(3) + 5),
            (d0[0] - 3, frag.cy(3) + 5),
        ],
        "w",
    )
    s.line(d0[0] - 3, frag.cy(3) - 5, last[0] + last[2] + 4, frag.cy(2) + 3, "a", 2.4)
    s.chip(lane, frag.cy(6), "2", "a")
    # 1 and 4: the thread and its region
    s.chip(frag.cx(8), thread[1] + thread[3] + 13, "1", "w")
    s.chip(frag.cx(9), region[1] + region[3] + 13, "4", "w")

    hot = {core_at(k): "p" for k in range(4)}
    hot[(8, 3)] = "a"
    card_tiles(s, frag, hot, compact=True)
    for row in (0, 1):  # the range: any bank, here D0
        outline(s, frag.box(0, row), "p")
    outline(s, region, "w", 1.8, "4 3")
    s.text(
        frag.x,
        frag.bottom + 22,
        "D0 to D7 are the eight DRAM banks. The range (outlined) lies in D0; work "
        "cluster 1's region (dashed) in D6, its own bank.",
        12,
        c["mute"],
    )

    x = frag.right + 26
    s.text(x, y + 60, "kmx_fetch(region, offset, &from, bytes, ...)", 12, mono=True)
    end = steps(s, x, y + 78, rows)
    for i, line in enumerate(
        [
            "from: a range of any bank, or a tensor's pages.",
            "kmx_store is the same the other way: the dirty",
            "lines written back, then the cores copy them out.",
            "A cluster's own cache reaches one bank only;",
            "its loads through a NoC window move 0.024 GB/s.",
        ]
    ):
        s.text(x, end + 16 + 18 * i, line, 12, c["mute"])
    return height


def panel_fetch_time(s, y):
    """The call in time, the lines that make it, what it costs."""
    c = s.c
    code = lines_of(FETCH, [58, 59, 60, 70, 71, 72])
    names = ["X280 1, the thread", "X280 0, the dispatcher", "cores 0 to 3"]
    lanes = Lanes(s, y + 60, names, BOX_X + 168, pitch=34)
    lanes.bar(names[0], 1, 9, "1", "w")
    lanes.bar(names[0], 9, 13, "", "w", "plain")
    lanes.bar(names[0], 13, 87, "waits for the command's event", "w", "plain")
    lanes.bar(names[0], 89, 100, "4 reads", "w")
    lanes.bar(names[1], 13, 17, "", "a")
    lanes.bar(
        names[2], 19, 85, "3 copy: 64 MiB in 1.06 ms, 63.4 GB/s on four cores", "p"
    )
    lanes.hop(names[0], names[1], 13, "w")
    lanes.hop(names[1], names[2], 17.5, "a")
    lanes.hop(names[2], names[1], 85.5, "g")
    lanes.hop(names[1], names[0], 87.5, "g")
    lanes.note(names[0], 1, "evict, 21 µs", "w")
    lanes.note(names[1], 19, "2 the command; the event comes back the same way")
    axis = lanes.axis()

    top = axis + 16
    rows = [
        ("head", "experiments/x280/bank_fetch/thread/islands.c 58 to 60, 70 to 72"),
        ("code", code[0], False, "from", "the tensor and the range, never the island"),
        ("code", "    " + code[1], False),
        ("code", code[2], False),
        ("code", code[3], True, "fetch"),
        ("code", "    " + code[4], True),
        ("code", code[5], True),
    ]
    end = code_rows(s, BOX_X + 24, top, "w", rows)

    # what the whole call costs
    ty = end + 26
    cols = [BOX_X + 14, BOX_X + 520, BOX_X + 650, BOX_X + 782]
    s.text(cols[0], ty, "the whole call", 12, c["fg"], True)
    for x, text in zip(cols[1:], ("1 MiB", "16 MiB", "64 MiB")):
        s.text(x, ty, text, 12, c["fg"], True, anchor="end")
    table = [
        ("kmx_fetch from another bank of the card", "40.2 µs", "289 µs", "1.08 ms"),
        ("kmx_store within one bank", "64.5 to 69.4 µs", "", "2.30 to 2.32 ms"),
        ("kmx_fetch of a tensor another island holds", "115 µs", "", "1.13 ms"),
        ("kmx_store into a tensor another island holds", "112 µs", "", "1.14 ms"),
        ("the X280's own loads through a NoC window", "44.1 ms", "", ""),
    ]
    for k, (label, *values) in enumerate(table):
        yy = ty + 22 + 20 * k
        s.add(
            f'<line x1="{cols[0]:.1f}" y1="{yy - 14:.1f}" x2="{cols[3]:.1f}" '
            f'y2="{yy - 14:.1f}" stroke="{c["line"]}" stroke-width="1"/>'
        )
        s.text(cols[0], yy, label, 12, c["fg"])
        for x, text in zip(cols[1:], values):
            s.text(x, yy, text, 12, c["fg"], anchor="end")
    yy = ty + 22 + 20 * len(table)
    notes = [
        "One card: four Tensix cores, work cluster 1, every byte checked. Another "
        "island: two cards, 16 cores, each run of pages one line,",
        "straight between the two DRAMs over four links (59.5 GB/s), every word "
        "checked; on one island the same call is the local copy.",
    ]
    for i, line in enumerate(notes):
        s.text(cols[0], yy + 6 + 18 * i, line, 12, c["mute"])
    height = yy + 6 + 18 * len(notes) - y
    panel_frame(
        s, y, height, "the call in time, and what it costs", "kmx_fetch, kmx_store", "w"
    )
    return height


def figure_reach(theme):
    s = Svg(theme, "A thread reaches any DRAM", markers="abmwrpyg")
    bottom = stack(s, [panel_fetch_card, panel_fetch_time])
    return s.finish(int(bottom + 4))


# ---------------------------------------------------------------------------
# Figure "islands": more than one card.
# ---------------------------------------------------------------------------

GROUP = (96, 44)  # the drawn command's group, in flat ranks


def panel_two_cards(s, y):
    """Two cards as two islands: flat ranks, a group over both, the link cores."""
    c = s.c
    pitch, tile = 22, 19
    left = BOX_X + 15
    top = y + 112
    span = 17 * pitch - (pitch - tile)
    frags = [
        Frag(
            s,
            left + k * (span + 26),
            top,
            list(range(17)),
            list(range(12)),
            pitch=pitch,
            tile=tile,
            heads=False,
        )
        for k in range(2)
    ]
    bottom = frags[0].bottom
    height = bottom + 62 - y
    panel_frame(
        s,
        y,
        height,
        "two cards, one run: ranks go on across the cards",
        "01_mapcollect/islands.py, four links",
        "a",
    )
    # the host: one session, one group word
    s.rect(left, y + 36, 2 * span + 26, 24, c["bg"], c["rule"], rx=4, sw=1.5)
    s.text(left + 12, y + 52, "host", 12, c["fg"], True)
    s.text(
        left + 12 + width("host", bold=True) + 10,
        y + 52,
        'km.Session(cards=[0, 1]);  run.submit("map", (96, 44)): one command, '
        "its group in flat ranks",
        12,
        c["mute"],
    )
    ranks = ((0, 117), (118, 235))
    for k, frag in enumerate(frags):
        x = frag.cx(8)
        s.line(x, y + 62, x, frag.box(8, 3)[1] - 3, "a", 2.4)
        s.text(
            frag.x,
            y + 84,
            f"island {k}: ranks {ranks[k][0]} to {ranks[k][1]}",
            12,
            c["fg"],
            True,
        )
        s.text(x + 10, y + 84, "its part, to its own dispatcher", 12, c["mute"])
        here = ("ranks 96 to 117: cores 96 to 117", "ranks 118 to 139: cores 0 to 21")[
            k
        ]
        s.text(frag.x, y + 100, "the group here, " + here, 12, c["a"][0])

    # the group over both islands: island 0's last ranks, island 1's first
    parts = (range(96, 118), range(0, 22))
    for frag, part in zip(frags, parts):
        hot = {core_at(k): "a" for k in part}
        hot[(8, 3)] = "a"
        card_tiles(s, frag, hot, strong={core_at(part[0])}, compact=True)
        for k, index in enumerate((118, 119)):  # the link cores: not ranks
            px, py, w, h = frag.box(*core_at(index))
            s.rect(px, py, w, h, c["bg"], c["fg"], rx=3, sw=1.6)
            s.text(
                px + w / 2, py + h / 2 + 4, f"L{k}", 12, c["fg"], True, anchor="middle"
            )

    # the four links: each CPU of a link core holds one
    kinds = ("a", "p", "p", "p")
    for link in range(4):
        core_k, cpu = divmod(link, 2)
        ends = []
        for frag in frags:
            px, py, w, h = frag.box(*core_at(118 + core_k))
            ends.append((px + 5 + cpu * 9, py + h + 1))
        level = bottom + 8 + link * 5
        (x1, y1), (x2, y2) = ends
        path(
            s,
            [(x1, y1), (x1, level), (x2, level), (x2, y2)],
            c[kinds[link]][0],
            2.0 if link else 2.6,
        )
    tx = frags[0].x
    s.text(tx, bottom + 22, "link 0, lines:", 12, c["a"][0], True)
    s.text(
        tx + width("link 0, lines:", bold=True) + 8,
        bottom + 22,
        "a command's part, ask and put lines, in order",
        12,
        c["mute"],
    )
    s.text(tx, bottom + 40, "every link, pages:", 12, c["p"][0], True)
    s.text(
        tx + width("every link, pages:", bold=True) + 8,
        bottom + 40,
        "a page a packet, into DRAM or a core's L1",
        12,
        c["mute"],
    )
    s.text(
        frags[1].x,
        bottom + 40,
        "L0, L1: link cores, not ranks; each data-movement CPU holds a link",
        12,
        c["mute"],
    )
    return height


def node(s, cx, cy, label, kind=None, w=28, h=22):
    """An island as a box with its number; returns its box."""
    c = s.c
    fill, edge = (c[kind][1], c[kind][0]) if kind else (c["bg"], c["rule"])
    box = (cx - w / 2, cy - h / 2, w, h)
    s.rect(*box, fill, edge, rx=5, sw=1.5)
    if label:
        s.text(cx, cy + 4, label, 12, c["fg"], True, anchor="middle")
    return box


def wire(s, a, b, colour=None, shift=0.0, sw=1.8):
    """A link between two islands' boxes, `shift` pixels beside the straight line."""
    (ax, ay), (bx, by) = centre(a), centre(b)
    (x1, y1), (x2, y2) = edge_point(a, (bx, by), 1.0), edge_point(b, (ax, ay), 1.0)
    length = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
    nx, ny = -(by - ay) / length * shift, (bx - ax) / length * shift
    stroke = s.c[colour][0] if colour else s.c["rule"]
    path(s, [(x1 + nx, y1 + ny), (x2 + nx, y2 + ny)], stroke, sw)


def torus(s, x, y, rows, columns, pitch=13, route=()):
    """rows x columns islands as dots joined to their four neighbours, the edges
    joined around; the islands of `route` and the links between them in colour."""
    c = s.c
    on = set(route)
    steps = set(zip(route, route[1:])) | set(zip(route[1:], route))

    def spot(r, k):
        return x + k * pitch, y + r * pitch

    for r in range(rows):
        for k in range(columns):
            px, py = spot(r, k)
            for dr, dk in ((0, 1), (1, 0)):
                there = (r + dr, k + dk)
                hot = ((r, k), there) in steps
                stroke = c["a"][0] if hot else c["rule"]
                if there[0] < rows and there[1] < columns:
                    qx, qy = spot(*there)
                    path(s, [(px, py), (qx, qy)], stroke, 2.2 if hot else 1.0)
                else:  # around the edge: a stub at each end
                    path(s, [(px, py), (px + dk * 6, py + dr * 6)], stroke, 1.0)
                    qx, qy = spot(there[0] % rows, there[1] % columns)
                    path(s, [(qx, qy), (qx - dk * 6, qy - dr * 6)], stroke, 1.0)
    for r in range(rows):
        for k in range(columns):
            px, py = spot(r, k)
            kind = c["a"] if (r, k) in on else None
            fill, edge = (kind[0], kind[0]) if kind else (c["bg"], c["rule"])
            s.add(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.4" fill="{fill}" '
                f'stroke="{edge}" stroke-width="1.2"/>',
                2,
            )


def panel_graph(s, y):
    """Islands as a graph: the graphs that ran, one of 32 islands, the flat ranks."""
    c = s.c
    left = BOX_X + 14
    cell = (BOX_W - 28) / 6
    gy = y + 76

    def at(k):
        return left + cell * k + cell / 2

    # two cards, four links
    cx = at(0)
    a, b = node(s, cx - 34, gy, "0"), node(s, cx + 34, gy, "1")
    for shift in (-6, -2, 2, 6):
        wire(s, a, b, shift=shift, sw=1.5)
    # a line of four: island 0 reaches island 3 over three links
    cx = at(1)
    row = [
        node(s, cx - 45 + 30 * k, gy, str(k), "a" if k in (0, 3) else None, w=22)
        for k in range(4)
    ]
    for one, two in zip(row, row[1:]):
        wire(s, one, two, "a", sw=2.4)
    # a ring of four
    cx = at(2)
    spots = [(-28, -22), (28, -22), (28, 22), (-28, 22)]
    ring = [node(s, cx + dx, gy + dy, str(k)) for k, (dx, dy) in enumerate(spots)]
    for k in range(4):
        wire(s, ring[k], ring[(k + 1) % 4])
    # a ring of three
    cx = at(3)
    spots = [(0, -22), (30, 22), (-30, 22)]
    three = [node(s, cx + dx, gy + dy, str(k)) for k, (dx, dy) in enumerate(spots)]
    for k in range(3):
        wire(s, three[k], three[(k + 1) % 3])
    # two by two, the edges joined around: two links a pair
    cx = at(4)
    spots = [(-28, -22), (28, -22), (-28, 22), (28, 22)]
    grid = [node(s, cx + dx, gy + dy, str(k)) for k, (dx, dy) in enumerate(spots)]
    for one, two in ((0, 1), (2, 3), (0, 2), (1, 3)):
        for shift in (-3, 3):
            wire(s, grid[one], grid[two], shift=shift, sw=1.5)
    # four by eight, the edges joined around: 32 islands, one route in colour
    route = [(2, 1), (1, 1), (1, 2), (1, 3), (1, 4), (0, 4), (0, 5)]
    torus(s, at(5) - 45.5, gy - 19.5, 4, 8, route=route)
    captions = [
        ("two cards", "four links"),
        ("a line", "three links from 0 to 3"),
        ("a ring", ""),
        ("a ring of three", ""),
        ("2 x 2, joined around", "two links a pair"),
        ("32 islands", "sized for; not run"),
    ]
    for k, (name, gloss) in enumerate(captions):
        s.text(at(k), gy + 52, name, 12, c["fg"], True, anchor="middle")
        if gloss:
            s.text(at(k), gy + 68, gloss, 12, c["mute"], anchor="middle")

    # the flat ranks over a line of four, and one group over an edge
    by = gy + 88
    s.text(left, by + 12, "flat ranks", 12, c["fg"], True)
    x0 = left + 86
    span = RIGHT - 14 - x0
    shares = (35, 34, 22, 23)
    kinds = ("b", "y", "m", "r")
    x, edges = x0, []
    for k, share in enumerate(shares):
        w = span * share / sum(shares)
        edge, fill, _ = c[kinds[k]]
        s.rect(x, by, w - 3, 16, fill, edge, rx=3)
        s.text(x + w / 2, by + 12, f"island {k}", 12, c["fg"], anchor="middle")
        edges.append((x, w))
        x += w
    g1 = edges[1][0] + edges[1][1] * 0.55
    g2 = edges[2][0] + edges[2][1] * 0.6
    path(s, [(g1, by + 20), (g1, by + 26), (g2, by + 26), (g2, by + 20)], c["a"][0])
    s.text(
        (g1 + g2) / 2,
        by + 42,
        "a group is (first, count), anywhere in the flat ranks",
        12,
        c["a"][0],
        True,
        anchor="middle",
    )
    s.text(
        left,
        by + 62,
        "Ran: two P150A joined by four links, and one card split into two to four "
        "islands joined in each of the first five ways.",
        12,
        c["mute"],
    )
    height = by + 62 + 14 - y
    panel_frame(
        s,
        y,
        height,
        "islands are a graph; a rank may be any number of links away",
        "docs/reference/islands.md",
        "a",
    )
    return height


def link_core(s, x, y, h=16):
    """A link core, drawn as a box with an L; returns its box."""
    c = s.c
    s.rect(x, y, 14, h, c["bg"], c["fg"], rx=3, sw=1.4)
    s.text(x + 7, y + h / 2 + 4, "L", 12, c["fg"], True, anchor="middle")
    return (x, y, 14, h)


def core(s, x, y, kind, size=16):
    """A rank's core in the colour `kind` ("g": not part of what is shown)."""
    edge, fill, _ = s.c[kind]
    s.rect(x, y, size, size, fill, edge, rx=3, sw=1.4)
    return (x, y, size, size)


def pages(s, x, y, kinds, w=10, h=12, gap=2):
    """A row of DRAM pages, each in its colour; returns the row's box."""
    for k, kind in enumerate(kinds):
        edge, fill, _ = s.c[kind]
        s.rect(x + k * (w + gap), y, w, h, fill, edge, rx=2)
    return (x, y, len(kinds) * (w + gap) - gap, h)


def two_islands(s, x, y, w, h, gap=24):
    """Two island boxes side by side with their link cores at the facing edges;
    returns the boxes' x and the two link cores."""
    c = s.c
    ix = [x, x + w + gap]
    for i in range(2):
        s.rect(ix[i], y, w, h, c["bg"], c["line"], rx=6)
        s.text(ix[i] + w - 7, y + h - 6, f"island {i}", 12, c["mute"], anchor="end")
    near = link_core(s, ix[0] + w - 18, y + 10, 26)
    far = link_core(s, ix[1] + 4, y + 10, 26)
    return ix, near, far


def caption(s, cx, y, name, gloss, measured, ink, mono=True):
    """Three centred lines under a drawing: its name, what it does, what it cost."""
    c = s.c
    s.text(cx, y, name, 12, ink, True, mono, anchor="middle")
    s.text(cx, y + 16, gloss, 12, c["fg"], anchor="middle")
    s.text(cx, y + 32, measured, 12, c["mute"], anchor="middle")


def panel_service(s, y):
    """What the link cores carry: bytes, an add, pages out and pages back, drawn
    between two islands; one call to a rank here, a link away and two away; and the
    three things that start work on a far core."""
    c = s.c
    left = BOX_X + 14
    cell = (BOX_W - 28) / 4
    top = y + 38
    ops = [
        ("a", "km::write_l1", "bytes into a rank's L1", "11.8 µs with an add"),
        (
            "m",
            "km::add_word",
            "adds to a count a rank waits on",
            "2.16 µs, not waited for",
        ),
        ("p", "km::write_pages", "pages, to the island's DRAM", "72.5 GB/s each way"),
        ("w", "km::read_pages", "pages, from the island's DRAM", "40.6 to 48.3 GB/s"),
    ]
    for k, (kind, name, gloss, measured) in enumerate(ops):
        x0 = left + k * cell + 7
        ink = c[kind][0]
        ix, near, far = two_islands(s, x0, top, 82, 74, 14)
        mid = top + 23
        core(s, x0 + 10, top + 15, kind)
        path(s, [(near[0] + 14, mid), (far[0], mid)], ink, 2.2)
        if k < 2:  # into the far rank's L1: a slot of bytes, or a count
            core(s, ix[1] + 56, top + 15, "g")
            slot = (ix[1] + 26, top + 40, 46, 14)
            s.rect(*slot, c[kind][1], ink, rx=2)
            if k == 1:
                s.text(
                    ix[1] + 49, top + 51, "5 + 1", 12, c["fg"], True, anchor="middle"
                )
            path(s, [(x0 + 26, mid), (near[0], mid)], ink, 2.2)
            s.poly(
                [(far[0] + 14, mid), (ix[1] + 40, mid), (ix[1] + 40, top + 39)], kind
            )
        else:  # the far island's DRAM: pages written from L1, or read into it
            s.rect(x0 + 10, top + 44, 28, 14, c[kind][1], ink, rx=2)
            s.text(x0 + 44, top + 56, "L1", 12, c["mute"])
            pages(s, ix[1] + 26, top + 45, [kind, kind, kind, "g"])
            if k == 2:
                path(s, [(x0 + 26, mid), (near[0], mid)], ink, 2.2)
                s.poly(
                    [(far[0] + 14, mid), (ix[1] + 43, mid), (ix[1] + 43, top + 44)],
                    kind,
                )
            else:
                path(
                    s,
                    [(ix[1] + 43, top + 44), (ix[1] + 43, mid), (far[0] + 14, mid)],
                    ink,
                )
                s.line(near[0], mid, x0 + 27, mid, kind, 2.2)
        caption(s, x0 + 89, top + 94, name, gloss, measured, ink)

    # one call by flat rank: here, one link away, two links away
    sy = top + 148
    s.text(left, sy + 20, "one call, by flat rank", 12, c["fg"], True)
    s.text(left, sy + 36, "km::write_l1(.., rank, ..)", 12, c["mute"], mono=True)
    bx, bw, gap = left + 196, 150, 34
    boxes = []
    for i in range(3):
        x = bx + i * (bw + gap)
        s.rect(x, sy, bw, 50, c["bg"], c["line"], rx=6)
        s.text(x + bw - 8, sy + 14, f"island {i}", 12, c["mute"], anchor="end")
        boxes.append(x)
    for i in range(2):
        link_core(s, boxes[i] + bw - 18, sy + 18, 28)
        link_core(s, boxes[i + 1] + 4, sy + 18, 28)
    s.rect(boxes[0] + 8, sy + 16, 14, 30, c["a"][1], c["a"][0], rx=3, sw=1.4)
    lanes = [
        (0, sy + 22, 54, "1.09 µs"),
        (1, sy + 31, 44, "8.6 µs"),
        (2, sy + 40, 44, "18.8 µs"),
    ]
    for island, ly, offset, measured in lanes:
        tx = boxes[island] + offset
        start = boxes[0] + 22
        for i in range(island):  # to the link core, across the link, on from the next
            near, far = boxes[i] + bw - 18, boxes[i + 1] + 4
            path(s, [(start, ly), (near, ly)], c["a"][0], 2.2)
            path(s, [(near + 14, ly), (far, ly)], c["a"][0], 2.2)
            start = far + 14
        s.line(start, ly, tx - 2, ly, "a", 2.2)
        core(s, tx, ly - 5, "a", 10)
        s.text(tx + 16, ly + 4, measured, 12, c["fg"], True)

    # what starts work on a far core
    wy = sy + 74
    s.text(left, wy + 4, "what starts work on a far core", 12, c["fg"], True)
    third = (BOX_W - 28) / 3
    starts = [
        (
            "a",
            "a command's part",
            "to that island's dispatcher",
            "4.4 to 5.2 µs a NOP across",
        ),
        ("m", "a message", "names the handler the rank runs", "4.7 µs a crossing"),
        (
            "r",
            "a port's send",
            "bytes, then 1 at the port it waits on",
            "km::emit, km::write_l1(.., port)",
        ),
    ]
    for k, (kind, name, gloss, measured) in enumerate(starts):
        x0 = left + k * third + 10
        ink = c[kind][0]
        ix, near, far = two_islands(s, x0, wy + 14, 106, 62, 22)
        mid = wy + 37
        path(s, [(near[0] + 14, mid), (far[0], mid)], ink, 2.2)
        if k == 0:  # dispatcher to dispatcher, then its cores
            for i in range(2):
                s.rect(ix[i] + 26 + 4 * i, wy + 27, 22, 18, c["w"][1], c["w"][0], rx=3)
                s.text(
                    ix[i] + 37 + 4 * i,
                    wy + 41,
                    "X0",
                    12,
                    c["fg"],
                    True,
                    anchor="middle",
                )
            path(s, [(ix[0] + 48, mid), (near[0], mid)], ink, 2.2)
            s.line(far[0] + 14, mid, ix[1] + 29, mid, kind, 2.2)
            for n in range(3):
                target = core(s, ix[1] + 66, wy + 20 + n * 15, kind, 11)
                s.line(ix[1] + 52, mid, target[0] - 1, target[1] + 5.5, kind, 1.4)
        elif k == 1:  # into the rank's queue; the rank takes it and runs the handler
            core(s, ix[0] + 30, wy + 30, kind, 14)
            path(s, [(ix[0] + 44, mid), (near[0], mid)], ink, 2.2)
            for n in range(3):
                fill = c[kind][1] if n == 0 else c["cell"]
                s.rect(ix[1] + 32 + n * 11, wy + 31, 10, 12, fill, ink, rx=2)
            s.line(far[0] + 14, mid, ix[1] + 31, mid, kind, 2.2)
            target = core(s, ix[1] + 80, wy + 30, kind, 14)
            s.line(ix[1] + 65, mid, target[0] - 1, mid, kind, 1.6)
        else:  # bytes into the slot, then the port's count: the waiting kernel goes on
            core(s, ix[0] + 30, wy + 30, kind, 14)
            path(s, [(ix[0] + 44, mid), (near[0], mid)], ink, 2.2)
            s.rect(ix[1] + 32, wy + 31, 26, 12, c[kind][1], ink, rx=2)
            s.line(far[0] + 14, mid, ix[1] + 31, mid, kind, 2.2)
            s.rect(ix[1] + 62, wy + 31, 20, 12, c["bg"], ink, rx=2)
            s.text(ix[1] + 72, wy + 41, "+1", 12, c["fg"], True, anchor="middle")
            core(s, ix[1] + 86, wy + 30, kind, 14)
        caption(s, x0 + 117, wy + 94, name, gloss, measured, ink, mono=False)
    s.text(
        left,
        wy + 148,
        "Measured across two cards; the three times of one call on one card split as "
        "a line of three, for 16 bytes and an add.",
        12,
        c["mute"],
    )
    height = wy + 148 + 14 - y
    panel_frame(
        s,
        y,
        height,
        "what the link cores carry, by flat rank or by tensor page",
        "experiments/islands/link_ops",
        "m",
    )
    return height


def comb(s, xs, y0, y1, to, ink, sw=1.4):
    """Ticks from the cores at `xs` down to a line at y1 that runs to x `to`."""
    for x in xs:
        path(s, [(x, y0), (x, y1)], ink, sw)
    path(s, [(max(*xs, to), y1), (min(*xs, to), y1)], ink, sw)


def panel_aware(s, y):
    """A kernel written by rank runs on islands; one that knows its islands crosses
    less: the fold to a leader, the broadcast, the leaders' exchange, the placement,
    then the words a kernel says them with."""
    c = s.c
    left = BOX_X + 14
    half = (BOX_W - 28 - 24) / 2
    top = y + 40
    ink = c["r"][0]
    bw = (half - 42) / 2
    sides = [
        (
            "by rank:",
            "every rank sends its own part to rank 0",
            "118 parts cross",
            "118",
        ),
        (
            "by island:",
            "each island folds to its leader; the leaders' parts cross",
            "1 part crosses: 238 ranks summed in 167.5 µs (26_islands_sum)",
            "1",
        ),
    ]
    for k, (head, gloss, measured, count) in enumerate(sides):
        x0 = left + k * (half + 24)
        ix = [x0, x0 + bw + 42]
        spots = [[], []]
        for i in range(2):
            s.rect(ix[i], top, bw, 134, c["bg"], c["line"], rx=6)
            s.text(ix[i] + 2, top + 148, f"island {i}", 12, c["mute"])
            for n in range(12):
                cx = ix[i] + 10 + 20 * i + n % 6 * 19
                cy = top + 14 + n // 6 * 20
                if i == 1 and k == 1 and n == 6:
                    s.rect(cx, cy, 14, 14, ink, ink, rx=3)
                else:
                    core(s, cx, cy, "r", 14)
                spots[i].append((cx, cy))
        links = []
        for x in (ix[0] + bw - 18, ix[1] + 4):  # tall, so every part passes through
            s.rect(x, top + 8, 14, 118, c["bg"], c["fg"], rx=3, sw=1.4)
            s.text(x + 7, top + 24, "L", 12, c["fg"], True, anchor="middle")
            links.append(x)
        # rank 0, drawn tall so each part that reaches it has its own arrowhead
        bar = (ix[0] + 96, top + 72, 16, 54)
        s.rect(*bar, ink, ink, rx=3)
        s.text(bar[0] - 8, top + 103, "rank 0", 12, c["fg"], True, anchor="end")
        low = top + 48  # the bottom of the cores
        # inside island 0: its ranks to rank 0 (a local send each)
        comb(s, [cx + 7 for cx, _ in spots[0][6:]], low, top + 58, bar[0] + 8, ink, 1.2)
        s.line(bar[0] + 8, top + 58, bar[0] + 8, top + 71, "r", 1.4)
        if k == 0:  # every rank of island 1 sends its own part across
            for col in range(6):
                for row in range(2):
                    cx, cy = spots[1][row * 6 + col]
                    ly = top + 76 + 4.2 * (2 * col + 1 - row)
                    if row:  # the lower core: straight down
                        start = [(cx + 7, low)]
                    else:  # the upper core: out to the right, down between columns
                        start = [(cx + 14, cy + 7), (cx + 16.5, cy + 7)]
                    path(s, [*start, (start[-1][0], ly), (bar[0] + 26, ly)], ink, 0.9)
                    s.line(bar[0] + 26, ly, bar[0] + 17, ly, "r", 0.9)
        else:  # island 1 folds to its leader, whose one part crosses
            lead = spots[1][6]
            comb(
                s,
                [cx + 7 for cx, _ in spots[1][7:]],
                low,
                top + 58,
                lead[0] + 7,
                ink,
                1.2,
            )
            s.line(lead[0] + 7, top + 58, lead[0] + 7, low + 1, "r", 1.4)
            ly = top + 99
            path(
                s,
                [(lead[0], lead[1] + 7), (lead[0] - 7, lead[1] + 7), (lead[0] - 7, ly)],
                ink,
                2.4,
            )
            path(s, [(lead[0] - 7, ly), (bar[0] + 40, ly)], ink, 2.4)
            s.line(bar[0] + 40, ly, bar[0] + 17, ly, "r", 2.4)
        s.chip((links[0] + 14 + links[1]) / 2, top + 44, count, "r")
        s.text(x0, top + 170, head, 12, c["fg"], True)
        s.text(x0 + width(head, bold=True) + 8, top + 170, gloss, 12, c["fg"])
        s.text(x0, top + 186, measured, 12, c["mute"])

    # three more ways a kernel acts by island
    r2 = top + 208
    third = (BOX_W - 28) / 3
    ways = [
        (
            "b",
            "a broadcast by island",
            "km::Broadcast(.., km::local)",
            "one block an island crosses",
        ),
        (
            "w",
            "the leaders gather and reduce",
            "km::all_gather, km::all_reduce",
            "256 pages: 21.3 µs and 180.0 µs",
        ),
        (
            "p",
            "pages placed with their ranks",
            "session.tensor(.., group=)",
            "201.1 µs; asked across, 306.9 µs",
        ),
    ]
    for j, (kind, name, call, measured) in enumerate(ways):
        x0 = left + j * third + 6
        tone = c[kind][0]
        ix = [x0, x0 + 106 + 32]
        for i in range(2):
            s.rect(ix[i], r2, 106, 98, c["bg"], c["line"], rx=6)
            s.text(ix[i] + 2, r2 + 112, f"island {i}", 12, c["mute"])
        near = link_core(s, ix[0] + 106 - 18, r2 + 26, 30)
        far = link_core(s, ix[1] + 4, r2 + 26, 30)
        spots = [[], []]
        for i in range(2):
            for n in range(8):
                cx = ix[i] + 8 + 20 * i + n % 4 * 16
                cy = r2 + 28 + n // 4 * 16
                first = (j == 0 and n == 0) or (j == 1 and n == (3 if i == 0 else 0))
                if first:
                    s.rect(cx, cy, 12, 12, tone, tone, rx=3)
                else:
                    core(s, cx, cy, kind, 12)
                spots[i].append((cx, cy))
        mid = r2 + 34
        if j == 0:  # cast along the part; one block to the next part's first core
            for i in range(2):
                (fx, fy), (lx, _) = spots[i][0], spots[i][3]
                path(
                    s,
                    [(fx + 6, fy - 1), (fx + 6, r2 + 12), (lx + 6, r2 + 12)],
                    tone,
                    1.4,
                )
                for cx, _ in spots[i][1:4]:
                    s.line(cx + 6, r2 + 12, cx + 6, fy - 1, kind, 1.2)
            end = spots[0][3][0] + 6
            path(
                s,
                [(end, r2 + 12), (near[0] + 7, r2 + 12), (near[0] + 7, r2 + 26)],
                tone,
            )
            s.line(near[0] + 14, mid + 7, far[0] - 1, mid + 7, kind, 2.0)
            first = spots[1][0][0] + 6
            path(
                s,
                [(far[0] + 7, r2 + 26), (far[0] + 7, r2 + 12), (first, r2 + 12)],
                tone,
            )
        elif j == 1:  # the leaders exchange their pages, each island's copy whole
            (ax, ay), (bx, by) = spots[0][3], spots[1][0]
            path(s, [(ax + 12, ay + 6), (near[0], ay + 6)], tone, 2.0)
            s.line(near[0] + 15, ay + 6, far[0] - 1, ay + 6, kind, 2.0, tail=True)
            path(s, [(far[0] + 14, by + 6), (bx, by + 6)], tone, 2.0)
            for i in range(2):
                pages(s, ix[i] + 8 + 20 * i, r2 + 78, [kind] * 4, 10, 10)
        else:  # each island reads its own part; the link carries nothing
            for i in range(2):
                box = pages(s, ix[i] + 8 + 20 * i, r2 + 78, [kind] * 4, 10, 10)
                for n in (0, 2):
                    px = box[0] + 5 + n * 12
                    s.line(px, r2 + 77, px, r2 + 58, kind, 1.4)
            path(
                s,
                [(near[0] + 14, mid + 7), (far[0], mid + 7)],
                c["rule"],
                1.4,
                dash="3 3",
            )
        cx = x0 + 122
        s.text(cx, r2 + 134, name, 12, tone, True, anchor="middle")
        s.text(cx, r2 + 150, call, 12, c["fg"], mono=True, anchor="middle")
        s.text(cx, r2 + 166, measured, 12, c["mute"], anchor="middle")

    # the words, from the kernel of the fold drawn above
    acting = lines_of(
        "26_islands_sum/kernels/writer.cpp", [23, 24, 26, 27, 31, 34, 35], [24, 31]
    )
    rows = [
        ("head", "26_islands_sum/kernels/writer.cpp 23 to 35"),
        ("code", acting[0], False, "leaders", "each island's first rank"),
        ("code", acting[1], False, "inside my island"),
        ("code", "    " + acting[2], True),
        ("code", "        " + acting[3], True),
        ("ell",),
        ("code", acting[4], False, "across", "my island's part, once"),
        ("code", "    " + acting[5], True),
        ("code", "        " + acting[6], True),
    ]
    end = code_rows(s, BOX_X + 24, r2 + 184, "r", rows)
    s.text(
        left,
        end + 18,
        "On one island each of these is the group, its first rank and a local send: "
        "the same kernel runs on one card.",
        12,
        c["mute"],
    )
    height = end + 18 + 14 - y
    panel_frame(
        s,
        y,
        height,
        "a kernel written by rank runs; one that knows its islands crosses less",
        "26_islands_sum, 03, 27, 28, 01",
        "r",
    )
    return height


def figure_islands(theme):
    s = Svg(theme, "More than one card", markers="abmwrpyg")
    bottom = stack(
        s,
        [
            panel_two_cards,
            panel_graph,
            panel_service,
            panel_aware,
        ],
    )
    c = s.c
    notes = [
        "Ranks are flat and contiguous by island. Between two cards a link runs "
        "through the router on an Ethernet tile of each card (row 1 of the",
        "grid); on a split card it is a write over the card's own NoC. Every number is "
        "a run on two P150A joined by four links, or on a split card where said.",
    ]
    for i, line in enumerate(notes):
        s.text(BOX_X + 4, bottom + 12 + 18 * i, line, 12, c["mute"])
    return s.finish(int(bottom + 12 + 18 * len(notes)))


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "docs" / "images"
    out.mkdir(parents=True, exist_ok=True)
    figures = {"reach": figure_reach, "islands": figure_islands}
    for theme in THEMES:
        for name, draw in figures.items():
            (out / f"{name}-{theme}.svg").write_text(draw(theme) + "\n")


if __name__ == "__main__":
    main()
