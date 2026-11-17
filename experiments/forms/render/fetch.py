"""A texel fetch per pixel, the texel's address a pixel's data, against numpy
(texels[address], exact): the texture held in L1 and gathered by a body on pack (and
three probes of that body on one page), or in DRAM, its page read by the entry's
word in a body on mover B."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nn"))

import numpy as np
from forms import card_options, finish, group, log, open_session, timed, words

import komorebi as km

HELD = 4  # pages of 1,024 texels the held texture takes
TEXTURE = 64  # pages of the texture in DRAM
GATHER = """
volatile tt_l1_ptr uint32_t* in = (volatile tt_l1_ptr uint32_t*)addr;
volatile tt_l1_ptr uint32_t* texels = (volatile tt_l1_ptr uint32_t*)tex;
volatile tt_l1_ptr uint32_t* result = (volatile tt_l1_ptr uint32_t*)out;
for (uint32_t i = 0; i < 1024; ++i) {
    result[i] = texels[in[i] & 4095];
}
"""
PROBES = {  # the held form narrowed: what pack's body writes into the out page
    "constant": "result[i] = 0x12345678u;",
    "copy": "result[i] = in[i];",
    "field_word_0": "result[i] = texels[0];",
}
PROBE = """
volatile tt_l1_ptr uint32_t* in = (volatile tt_l1_ptr uint32_t*)addr;
volatile tt_l1_ptr uint32_t* texels = (volatile tt_l1_ptr uint32_t*)tex;
volatile tt_l1_ptr uint32_t* result = (volatile tt_l1_ptr uint32_t*)out;
for (uint32_t i = 0; i < 1024; ++i) {
    %s
}
"""
CONSTANT = PROBE % PROBES["constant"]
COPY = PROBE % PROBES["copy"]
FIELD_WORD_0 = PROBE % PROBES["field_word_0"]
START = "uint32_t pixel = 0;"
FETCH = """
volatile tt_l1_ptr uint32_t* out_words =
    (volatile tt_l1_ptr uint32_t*)get_write_ptr(result);
km::read_pages(movement, command, tex, e_address >> 10, 1, get_write_ptr(stage));
noc_async_read_barrier();
volatile tt_l1_ptr uint32_t* page = (volatile tt_l1_ptr uint32_t*)get_write_ptr(stage);
out_words[pixel++] = page[e_address & 1023];
"""
WRITE = """
km::write_pages(movement, command, o, page, 1, get_write_ptr(result));
pixel = 0;
noc_async_write_barrier();
"""


class Texel(km.Entry):
    address: int


class Held(km.Program):
    """The texture's 4 pages held in L1 on every core; core r gathers n pages."""

    tex: km.L1[km.u32, HELD, 32, 32]

    @km.handler
    def load_tex(self, texture: km.Tensor):
        self.tex.fill(km.Stream(texture.pages(0, HELD)))

    @km.handler
    def fetch(self, uv: km.Tensor, o: km.Tensor, n: int):
        first = self.rank * n
        out = km.Stream(o.pages(first, n), write=True)
        for addr in km.Stream(uv.pages(first, n)):
            km.cxx(GATHER, takes=[addr, out, self.tex], on="pack")

    @km.handler
    def constant(self, uv: km.Tensor, o: km.Tensor, n: int):
        out = km.Stream(o.pages(0, n), write=True)
        for addr in km.Stream(uv.pages(0, n)):
            km.cxx(CONSTANT, takes=[addr, out, self.tex], on="pack")

    @km.handler
    def copy(self, uv: km.Tensor, o: km.Tensor, n: int):
        out = km.Stream(o.pages(0, n), write=True)
        for addr in km.Stream(uv.pages(0, n)):
            km.cxx(COPY, takes=[addr, out, self.tex], on="pack")

    @km.handler
    def field_word_0(self, uv: km.Tensor, o: km.Tensor, n: int):
        out = km.Stream(o.pages(0, n), write=True)
        for addr in km.Stream(uv.pages(0, n)):
            km.cxx(FIELD_WORD_0, takes=[addr, out, self.tex], on="pack")


class Dram(km.Program):
    """Rank r fetches the 1,024 texels of address page r, a texture page a texel."""

    stage: km.L1[km.u32, 1, 32, 32] = km.own(push=None)
    result: km.L1[km.u32, 1, 32, 32] = km.own(push=None)

    @km.handler
    def gather(self, uv: km.Tensor, tex: km.Tensor, o: km.Tensor):
        page = self.rank
        km.cxx(START, at="start", on="mover_b")
        for e in uv.pages(page, 1).entries(Texel, first=0, size=1, fixed=1024):
            km.cxx(FETCH, takes=[e, tex, self.stage, self.result], on="mover_b")
        km.cxx(WRITE, takes=[o, page, self.result], on="mover_b")


def held_form(session, texels, held_uv, report):
    """The held form and its three probes: the texels and the probes' pages."""
    cores = held_uv.shape[0] // 4
    texture = session.tensor(texels, "uint32")
    uv = session.tensor(held_uv, "uint32")
    o = session.tensor(np.zeros_like(held_uv), "uint32")
    page = np.zeros((1, 1024), np.uint32)
    ones = {name: session.tensor(page, "uint32") for name in PROBES}
    program = Held(texture=texture, uv=uv, o=o, session=session, log=log)
    run = session.load(program)
    try:
        timed(run, Held.load_tex, (0, cores), (texture,))
        report["held_us"] = timed(run, Held.fetch, (0, cores), (uv, o, 4))
        for name, one in ones.items():
            timed(run, getattr(Held, name), (0, 1), (uv, one, 1))
    finally:
        run.stop()
    return words(session, o), {
        name: words(session, one)[0] for name, one in ones.items()
    }


def held_checks(texels, held_uv, held, probes, report):
    flat = texels.reshape(-1)
    report["held_wrong"] = {
        "count": int(np.count_nonzero(held != flat[held_uv & 4095]))
    }
    report["held_zero"] = int(np.count_nonzero(held == 0))
    probe_want = {
        "constant": np.full(1024, 0x12345678, np.uint32),
        "copy": held_uv[0],
        "field_word_0": np.full(1024, flat[0], np.uint32),
    }
    for name, got in probes.items():
        report[f"probe_{name}_wrong"] = int(np.count_nonzero(got != probe_want[name]))
        report[f"probe_{name}_zero"] = int(np.count_nonzero(got == 0))
        log(f"probe {name}: {report[f'probe_{name}_wrong']} of 1024 wrong")
    log(f"held: {report['held_wrong']} wrong")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    card_options(parser)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    cores = args.cores
    texels = np.arange(TEXTURE * 1024, dtype=np.uint64) * 2654435761 + 0x9E3779B9
    texels %= 1 << 32  # no texel is 0
    texels = texels.astype(np.uint32).reshape(TEXTURE, 1024)
    held_uv = rng.integers(0, 1 << 32, (cores * 4, 1024), dtype=np.uint64)
    held_uv = held_uv.astype(np.uint32)
    dram_uv = rng.integers(0, TEXTURE * 1024, (cores, 1024)).astype(np.uint32)
    report = {"cores": cores, "cards": args.cards}
    one_card = "," not in args.cards  # two cards: the DRAM form alone
    with open_session(args) as session:
        if one_card:
            held, probes = held_form(session, texels, held_uv, report)
        texture = session.tensor(texels, "uint32")
        uv = session.tensor(dram_uv, "uint32")
        o = session.tensor(np.zeros_like(dram_uv), "uint32")
        program = Dram(uv=uv, tex=texture, o=o, session=session, log=log)
        run = session.load(program)
        ranks = group(session, args, cores)
        report["group"] = list(ranks)
        try:
            call = (uv, texture, o)
            report["dram_us"] = timed(run, Dram.gather, ranks, call)
        finally:
            run.stop()
        dram = words(session, o)
    checks = [("dram_wrong", "count", 0)]
    if one_card:
        held_checks(texels, held_uv, held, probes, report)
        checks.append(("held_wrong", "count", 0))
    wrong = int(np.count_nonzero(dram != texels.reshape(-1)[dram_uv]))
    report["dram_wrong"] = {"count": wrong}
    log(f"dram: {wrong} wrong")
    finish(report, checks)


if __name__ == "__main__":
    sys.exit(main())
