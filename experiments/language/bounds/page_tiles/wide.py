"""decode.py's chain with wider groups, so fewer groups read X: qkv 32 tiles in groups
of 14, o and down 28 in groups of 20, gate and up 28 in groups of 4 (one card).
README.md. python experiments/language/bounds/page_tiles/wide.py --pages 2 --chunk 1"""

import sys

from experiments.language.bounds import decode_cost as dc
from experiments.language.bounds.page_tiles import decode

WIDE = {
    "qkv": (32, (14, 28)),
    "o": (28, (20, 38)),
    "gate": (28, (4, 8)),
    "up": (28, (4, 8)),
    "down": (28, (20, 38)),
}


def made(session, t, chunk):
    """PagedLayer at WIDE's widths and group sizes."""
    at = 1 if decode.bd.islands_of(session) is not None else 0
    examples = {}
    for name, handler, source, target, _ in dc.CALLS:
        x, w, y = dc.PARAMS[handler]
        examples |= {x: t[source], w: t["w_" + name], y: t[target]}
    return decode.PagedLayer(
        log=decode.log,
        session=session,
        chain=True,
        S_QKV=WIDE["qkv"][1][at],
        S_O=WIDE["o"][1][at],
        S_G=WIDE["gate"][1][at],
        W16=WIDE["qkv"][0],
        W14=WIDE["o"][0],
        CHUNK=chunk,
        **examples,
    )


if __name__ == "__main__":
    dc.PLAN.update(WIDE)
    decode.made = made
    sys.exit(decode.main())
