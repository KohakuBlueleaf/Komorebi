"""The transport apply as a class: one handler on every core, each core's work its
table of entries in DRAM (plan.tables), the weights streamed from DRAM, the core's own
receiver tiles of y summed in L1 and written at the end."""

import komorebi as km

START = """
cb_reserve_back(sums, kFieldSumsPages);  // pack writes both in place
cb_reserve_back(hidden, kFieldHiddenPages);
"""
# The products are transport.hpp's (the turned product, the sums at an entry's tile).
BEGIN = """#include "../../../../projects/transport/transport.hpp"
transport::begin();
"""
FIRST = """
transport::first<sums, hidden, ready>(
    wt_buffer, wt_index, xs_buffer, e_k, e_a, e_r, e_rank);
"""
SECOND = """
transport::second<sums, hidden, ready>(ut_buffer, ut_index, e_rank, e_b, e_r);
"""
FINISH = "transport::finish<ready>();"
WRITE = "transport::write<sums>(out_buffer, j);"


class Slice(km.Entry):
    """A block's slice (plan.tables): x's tiles src : src + k, the first stream's
    pages w : w + a, U^T's u : u + b, the first own receiver r, R tiles of rank."""

    src: int
    k: int
    w: int
    a: int
    u: int
    b: int
    r: int
    rank: int


class Head(km.Entry):
    """A table page's words 1 and 2: the core's first receiver tile and its count."""

    first: int
    own: int


class Transport(
    km.Program, compute={"math_fidelity": "HiFi2", "fp32_dest_acc_en": True}
):
    """apply: y = M x. Core c's entries are the pages c pages .. of tables; its own
    receiver tiles, y's pages first .. first + own, are summed in L1 and written."""

    OWN = km.constant(16)  # own receiver tiles, most
    KMAX = km.constant(77)  # x's tiles of an entry, most
    RANK = km.constant(5)  # rank tiles, most
    BLOCK = km.constant(8)  # weight tiles a block
    sums: km.L1[km.fp32, OWN, 32, 32] = km.own(push=None)
    hidden: km.L1[km.bf16, RANK, 32, 32] = km.own(push=None)  # t = x V, t^T
    ready: km.L1[km.bf16, 2, 32, 32] = km.own(push="pack")  # pack to unpack

    @km.handler
    def apply(
        self,
        x: km.Tensor,
        w: km.Tensor,
        y: km.Tensor,
        tables: km.Tensor,
        profile: km.Tensor,
        pages: int,
    ):
        km.cxx(START, takes=[self.sums, self.hidden], at="start", on="pack")
        km.cxx(BEGIN)
        base = self.rank * pages
        for p in range(pages):
            for e in tables.pages(base + p, 1).entries(Slice, first=8, size=8, count=0):
                src = e.src
                xs = km.Stream(
                    x.tiles[0:1, src : min(src + self.KMAX, src + e.k)], block=self.KMAX
                )
                for wt in km.Stream(w.pages(e.w, e.a), block=self.BLOCK):
                    km.cxx(
                        FIRST,
                        takes=[wt, xs, e, self.sums, self.hidden, self.ready],
                    )
                for ut in km.Stream(w.pages(e.u, e.b), block=self.BLOCK):
                    km.cxx(
                        SECOND,
                        takes=[ut, e, self.sums, self.hidden, self.ready],
                    )
        for head in tables.pages(base, 1).entries(Head, first=1, size=2, fixed=1):
            first = head.first
            out = km.Stream(
                y.tiles[0:1, first : min(first + self.OWN, first + head.own)],
                write=True,
                block=1,
            )
            km.cxx(FINISH, takes=[self.ready])
            for j in range(head.own):
                km.cxx(WRITE, takes=[out, j, self.sums])
        km.cxx("transport::hand();", on="unpack")
        report = km.Stream(profile.pages(self.rank, 1), write=True, pattern="B")
        km.cxx("transport::report(report);", takes=[report], on="pack")
