"""A body on pack or math that takes a stream's page and a field (no card): unpack
writes the words of both waits and both addresses into the body CPU's one mailbox,
and the body's CPU reads the waits' words, then the addresses. Every wait comes
first, then every hand, so the two sequences agree; a body of compute's three hands
nothing and keeps its waits in takes order."""

import re

from chain_fixture import quiet
from class_reduce_arrival_fixture import tiles

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.front.nodes import prune, render

COPY = "for (uint32_t i = 0; i < 1024; ++i) result[i] = in[i];"
READ = "uint32_t word = in[0] + tex;"
ADD = "add_residual(whole_buffer, whole_index, residual_buffer, out_buffer);"


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 64, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def rows():
    """Two uint32 tensors of 4,096 B rows: uv and o."""
    return {n: Tensor(Pages(4096), (64, 1024), "uint32", "row") for n in ("uv", "o")}


class OnPack(km.Program):
    tex: km.L1[km.u32, 4, 32, 32]

    @km.handler
    def load_tex(self, texture: km.Tensor):
        self.tex.fill(km.Stream(texture.pages(0, 4)))

    @km.handler
    def copy(self, uv: km.Tensor, o: km.Tensor):
        out = km.Stream(o.pages(0, 1), write=True)
        for addr in km.Stream(uv.pages(0, 1)):
            km.cxx(COPY, takes=[addr, out, self.tex], on="pack")


class OnMath(km.Program):
    tex: km.L1[km.u32, 4, 32, 32]

    @km.handler
    def load_tex(self, texture: km.Tensor):
        self.tex.fill(km.Stream(texture.pages(0, 4)))

    @km.handler
    def copy(self, uv: km.Tensor, o: km.Tensor):
        for addr in km.Stream(uv.pages(0, 1)):
            km.cxx(READ, takes=[addr, self.tex], on="math")


class PageOnly(km.Program):
    @km.handler
    def copy(self, uv: km.Tensor, o: km.Tensor):
        out = km.Stream(o.pages(0, 1), write=True)
        for addr in km.Stream(uv.pages(0, 1)):
            km.cxx(COPY, takes=[addr, out], on="pack")


def compute(cls):
    tensors = rows()
    tensors["texture"] = rows()["uv"]
    compiled = compile_class(cls, {}, tensors)
    return "\n".join(render(prune(compiled.compute["copy"]), 0))


def sequences(text, body, cpu):
    """Up to the body: the words unpack writes to `cpu` (a wait's, a hand's) in its
    order, and the words `cpu` reads in its order."""
    text = text[: text.index(body)]
    wait, hand = r"compute\.wait_front\(", rf"compute\.hand\(ckernel::ThreadId::{cpu}"
    handed = r"compute\.handed\(ckernel::ThreadId::UnpackThreadId\)"
    written = re.findall(f"({wait})|({hand})", text)
    read = re.findall(f"({wait})|({handed})", text)
    return [("wait" if w else "hand") for w, _ in written], [
        ("wait" if w else "hand") for w, _ in read
    ]


def test_pack_reads_the_waits_then_the_hands_as_unpack_writes_them():
    written, read = sequences(compute(OnPack), COPY, "Pack")
    assert written == ["wait", "wait", "hand", "hand"]
    assert read == written


def test_math_reads_the_waits_then_the_hands_as_unpack_writes_them():
    written, read = sequences(compute(OnMath), READ, "Math")
    assert written == ["wait", "wait", "hand", "hand"]
    assert read == written


def test_a_body_of_a_page_alone_is_its_loop_wait_and_one_hand():
    text = compute(PageOnly)
    written, read = sequences(text, COPY, "Pack")
    assert written == read == ["wait", "hand"]
    hand = text.index("compute.hand(ckernel::ThreadId::PackThreadId, ")
    assert "get_tile_l1_byte_address(km_half0, km_tile0)" in text[hand:]


class Summed(km.Program):
    """A body of compute's three at a group's root takes the reduce's sum and a
    field: two waits, no hand, in takes order."""

    sums: km.L1[km.fp32, 1, 8, 32, 32]
    bias: km.L1[km.bf16, 8, 32, 32]

    @km.tile_function
    def project(x, w):
        return x @ w

    @km.handler
    def add(self, x: km.Tensor, w: km.Tensor, r: km.Tensor, y: km.Tensor):
        k0 = self.rank % 2 * 2
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, k0 : k0 + 2], block=2),
            km.Stream(w.tiles[k0 : k0 + 2, 0:8], block=16),
        ):
            self.sums += self.project(xb, wb)
        whole = km.reduce(self.sums, form="tree", group=2)
        if self.rank == self.leader:
            self.bias.fill(km.Stream(r.tiles[0:1, 0:8]))
            residual = km.Stream(r.tiles[0:1, 0:8], block=8)
            out = km.Stream(y.tiles[0:1, 0:8], write=True, block=8)
            km.cxx(ADD, takes=[whole, self.bias, residual, out])


def test_a_body_of_computes_three_keeps_the_sums_and_fields_waits_in_order():
    program = Summed(
        l1=1 << 20,
        log=quiet,
        x=tiles(1, 4, 0x10000),
        w=tiles(4, 8, 0x20000),
        r=tiles(1, 8, 0x80000),
        y=tiles(1, 8, 0x90000),
    )
    text = (program.built.folder / "compute.cpp").read_text()
    root = text[text.index("if (km_rank == (km_rank - km_rank % 2)) {") :]
    root = root[: root.index(ADD)]
    sum_wait = root.index("compute.wait_front(kFieldSums, kFieldSumsPages)")
    bias_wait = root.index("compute.wait_front(kFieldBias, kFieldBiasPages)")
    assert sum_wait < bias_wait
    assert "compute.hand(" not in root
