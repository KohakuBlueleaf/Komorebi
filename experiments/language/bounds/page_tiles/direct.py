"""decode.py's chain with gate's X (h) sent by o's roots into gate's cores' L1
(km.direct), W streamed as before: refused at run.chain, as up reads h too (README.md).
python experiments/language/bounds/page_tiles/direct.py --format bfp4_b --pages 2"""

import sys

import komorebi as km
from experiments.language.bounds import decode_cost as dc
from experiments.language.bounds.page_tiles import decode


class DirectLayer(decode.PagedLayer):
    """PagedLayer whose gate_up may take x_gu sent direct by the call before's roots."""

    @km.handler(direct=("x_gu",))
    def gate_up(
        self,
        x_gu: km.Tensor,
        w_gu: km.Tensor,
        y_gu: km.Tensor,
        k: int,
        col0: int,
        wp: km.Pattern,
        xp: km.Pattern,
    ):
        share = self.rank % self.S_G
        first = col0 + self.rank // self.S_G * self.W14
        k0 = share * (k // self.S_G) + min(share, k % self.S_G)
        k1 = (share + 1) * (k // self.S_G) + min(share + 1, k % self.S_G)
        for xb, wb in zip(
            km.Stream(x_gu.tiles[0:1, k0:k1], block=self.CHUNK, pattern=xp),
            km.Stream(
                w_gu.tiles[k0:k1, first : first + self.W14], block=self.B14, pattern=wp
            ),
        ):
            self.sums14 += self.project(xb, wb)
        whole = km.reduce(self.sums14, form="tree", group=self.S_G)
        if self.rank == self.leader:
            y_gu.tiles[0:1, first : first + self.W14].write(whole)


def made(session, t, chunk):
    """DirectLayer at decode_cost's group sizes, its examples the layer's tensors."""
    at = 1 if decode.bd.islands_of(session) is not None else 0
    examples = {}
    for name, handler, source, target, _ in dc.CALLS:
        x, w, y = dc.PARAMS[handler]
        examples |= {x: t[source], w: t["w_" + name], y: t[target]}
    return DirectLayer(
        log=decode.log,
        session=session,
        chain=True,
        S_QKV=dc.PLAN["qkv"][1][at],
        S_O=dc.PLAN["o"][1][at],
        S_G=dc.PLAN["gate"][1][at],
        CHUNK=chunk,
        **examples,
    )


def calls_of(session, t, spec):
    """The five calls, gate's X marked direct."""
    out = []
    wp, xp = decode.patterns_of(spec)
    for name, handler, source, target, k in dc.CALLS:
        _, group, col0 = dc.groups_of(session, name)[0]
        given = t[source]
        if name == "gate":
            given = km.direct(given)
        if name == "up":
            given = km.past(given)
        method = getattr(DirectLayer, handler)
        out.append(
            method(given, t["w_" + name], t[target], k, col0, wp, xp, cores=group)
        )
    return out


if __name__ == "__main__":
    decode.made = made
    decode.calls_of = calls_of
    sys.exit(decode.main())
