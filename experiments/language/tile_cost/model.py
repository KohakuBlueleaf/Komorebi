"""A rule for a command's time from the compiler's cost lines and the cost table,
against the published programs (README.md). Cycles are at 1,350 MHz.
    python model.py
"""

CLOCK = 1350.0

# ---- the cost table's rows, measured (README.md has the probes)
SYNC = 20.6  # acquire, commit, wait, release with no work
THROUGH_SRCA = 34.0  # a loop's cycles over its body: one bf16 tile through SrcA
EXTRA_SRCA = 30.0  # each further tile copied into Dst through SrcA
TO_DST = (
    132.4  # a loop of one fp32 tile unpacked straight to Dst, copy and pack, no body
)
EXTRA_TO_DST = 130.0  # each further such tile
PRODUCT = {
    "LoFi": 32.4,
    "HiFi2": 40.8,
    "HiFi4": 72.8,
}  # a tile product in a 1 x 1 x 4 loop
# the command's fixed term and the loop's, event microseconds (stream and loop probes)
FIXED = {1: 3.7, 2: 3.8, 4: 3.8, 8: 3.9, 32: 4.9, 36: 5.0}
# commands of k product loops (taps.py, 36 cores); beyond 8 loops each adds 1.5 us
FILL = 1.06  # a resident fp32 field's fill, us (waits.py apply0, apply1, apply2)
FIELD = (152.0, 283.0)  # cycles a tile the first and the second field add to a loop
# a field held in Dst for the stream (a pushed call's `.col()` or `.row()`): us a field
# a command, its fill and two loads (fitted: per-channel rows 0.94, the apply 0.50)
HELD = 0.72
# blocks of 2 tiles at every other tile row: us, us a block
STRIDED = (3.3, 0.51)
# a core's own data movement, read and write together, bytes a nanosecond (scale.py on
# float32, add of two streams: 12,288 B a tile in 0.43 us; bf16 add 6,144 B in 0.20 us)
MOVE = 29.5
BYTES = {"bf16": 2048.0, "fp32": 4096.0}
LOOPS_36 = {1: 5.16, 2: 7.66, 4: 11.40, 8: 17.38}
LOOP_36 = 1.5


def us(cycles):
    return cycles / CLOCK


def stream(cores, tiles, body, copies=1, fp32=False, fields=0, inputs=1, held=0):
    """A contiguous stream: fixed term + tiles x the longer of the core's loop (body +
    overhead) and its data movement (`inputs` tiles in, one out, a tile 2,048 B bf16 or
    4,096 B fp32); `fields`: per-tile copies of a resident fp32 field tile into Dst;
    `held`: fields held in Dst for the stream."""
    if fp32:
        over = TO_DST + EXTRA_TO_DST * (copies - 1)
    else:
        over = THROUGH_SRCA + EXTRA_SRCA * (copies - 1)
    over += sum(FIELD[:fields])
    size = BYTES["fp32" if fp32 else "bf16"]
    move = (inputs + 1) * size / MOVE / 1000.0
    fixed = FIXED[cores] + fields * FILL + held * HELD
    return fixed + tiles * max(us(body + over), move)


def strided(blocks, body_per_block=0.0, fields=0, tiles=2, held=0):
    """Blocks of 2 tiles at every other tile row (a group norm's, a conv's reads)."""
    base = STRIDED[0] + blocks * max(STRIDED[1], body_per_block) + held * HELD
    return base + fields * FILL + blocks * tiles * us(sum(FIELD[:fields]))


def chunk_row(cores, body, passes, tiles, fields):
    """The chunk form: a row of `tiles` fp32 tiles read `passes` times into Dst
    (126 cycles a tile) and `fields` field tiles copied; the body is the compiler's."""
    copies = (passes * tiles + fields) * 126.0
    return FIXED[cores] + us(body + copies + passes * (tiles // 2) * SYNC)


def loops(count):
    """A command of `count` product loops on 36 cores (the shift's and conv's shape)."""
    if count in LOOPS_36:
        return LOOPS_36[count]
    return LOOPS_36[8] + (count - 8) * LOOP_36


# ---- the programs: (name, model microseconds, measured warm median, README's first)
def table():
    rows = []

    def add(name, model, warm, first):
        rows.append((name, model, warm, first))

    add("SiLU, 8 cores x 8 tiles", stream(8, 8, 1475.6), 13.0, 13.2)
    add("GELU sigmoid form", stream(8, 8, 1795.6), 14.9, 15.2)
    add("GELU erf form, 16 tiles", stream(8, 16, 2723.6), 36.9, 38.0)
    add("residual add then SiLU", stream(8, 8, 1571.6, copies=2, inputs=2), 14.1, 14.2)
    add("per-channel rows (g.col)", stream(8, 8, 150.4, held=2), 6.9, 7.9)
    add("per-channel columns (g.row)", stream(8, 8, 123.7, fields=2), 9.2, 9.4)
    add("shift, 36 cores, 4 loops", loops(4), 11.3, 11.6)
    add("conv, 32 cores, 18 loops", loops(18), 31.4, 32.1)
    add("squares, 36 cores", stream(36, 4, 118.3), 5.4, 6.6)
    add(
        "group norm column_sums, 2 cores",
        strided(16, us(2 * PRODUCT["HiFi4"] + THROUGH_SRCA)),
        13.8,
        14.2,
    )
    add(
        "group norm lanes, 2 cores",
        FIXED[2] + us(465.0 + 2 * 126.0 + 2 * SYNC),
        4.2,
        4.3,
    )
    add(
        "group norm apply, 2 cores",
        strided(16, us(150.4 + THROUGH_SRCA), held=2),
        12.45,
        14.0,
    )
    add(
        "scores, 8 cores, 8 products",
        FIXED[8]
        + max(8 * us(PRODUCT["HiFi4"]), (10 * 2048.0 + 4 * 4096.0) / MOVE / 1000.0),
        5.4,
        6.5,
    )
    add(
        "masked, 8 cores, 4 fp32 tiles",
        stream(8, 4, 91.6, copies=2, fp32=True, inputs=2),
        5.75,
        6.0,
    )
    add(
        "weigh, 8 cores, 8 products",
        FIXED[8]
        + max(8 * us(PRODUCT["HiFi4"]), (12 * 2048.0 + 2 * 4096.0) / MOVE / 1000.0),
        6.1,
        6.3,
    )
    add("softmax, 8 cores, 4-tile row", chunk_row(8, 12211.0, 3, 4, 0), 16.6, 16.6)
    add("RMS norm, 4 cores, 64-tile row", chunk_row(4, 20235.0, 2, 64, 32), 30.6, 32.2)
    add(
        "layer norm, 4 cores, 64-tile row", chunk_row(4, 36180.0, 3, 64, 64), 63.0, 64.7
    )
    return rows


def main():
    print(
        f"{'program':38s} {'rule us':>9s} {'measured':>9s} {'error':>7s} "
        f"{'README':>7s}"
    )
    for name, model, warm, first in table():
        print(
            f"{name:38s} {model:9.1f} {warm:9.1f} "
            f"{100 * (model / warm - 1):+6.1f}% {first:7.1f}"
        )


if __name__ == "__main__":
    main()
