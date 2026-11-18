"""A class's thread that writes and runs the class's chain from its header
(km.x280_thread(chains={...})): the records' words as run.chain's, the tensors the
chain names beside the make's as load words, the per-submit word from the thread;
each refusal (no card)."""

import re
from pathlib import Path

import numpy as np
import pytest
from chain_fixture import CORES, TILES, WIDTH, Steps, quiet, run_of, tensors
from class_chain_direct_fixture import DirectSum

import komorebi as km
from komorebi import _env
from komorebi.chain import calls as kc
from komorebi.errors import ProgramError

pytestmark = pytest.mark.tree  # needs a tt-metal tree: the tree-tests lane

HERE = Path(__file__).resolve().parent / "class_thread"
TREE = _env.find_tree()[0]
COMPILER = pytest.mark.skipif(
    TREE is None or _env.x280_compiler(TREE) is None, reason="no X280 compiler"
)
GIVEN = tensors()
FLAGS = {"0u": 0, "KMX_CHAIN_MARKED": kc.words()["kChainMarked"],
         "KMX_CHAIN_SENDS": kc.words()["kChainSends"],
         "KMX_CHAIN_DIRECT": kc.words()["kChainDataMark"]}  # fmt: skip
SPEC = {"x": "a", "y": "out", "factor": km.at_submit("factor"), "width": WIDTH,
        "tiles": TILES, "cores": (0, CORES)}  # fmt: skip


class Threaded(Steps):
    records = km.region(4 * 64)
    thread = km.x280_thread(str(HERE / "chained.c"), chains={"steps": SPEC})


def made(cls=Threaded):
    x = GIVEN["x"]
    return cls(l1=1 << 20, log=quiet, a=x, b=x, out=GIVEN["y"], s=GIVEN["s"],
               t=GIVEN["t"], thread_tensors={"z": GIVEN["z"]})  # fmt: skip


def header(program):
    thread = program.threads[min(program.threads)]
    name = f"{type(program).__name__.lower()}_thread.h"
    return (Path(thread.includes[0]) / name).read_text()


@COMPILER
def test_the_header_writes_the_records_run_chain_writes():
    program = made()
    run_of(program).chain(
        program.steps(GIVEN["x"], GIVEN["y"], GIVEN["z"], GIVEN["s"], GIVEN["t"],
                      km.at_submit("factor"), WIDTH, TILES),
        cores=(0, CORES),
    )  # fmt: skip
    words = program.threads[min(program.threads)].words
    assert words[-1] == GIVEN["z"].address  # the chain's own tensor, after the make's


def emulated(text, program, w, upper="THREADED"):
    """Each record as kmx_chain_record would write it from the header's lines (the
    load's words from the thread's), `next` 0: (index, uint32 words)."""
    thread = program.threads[min(program.threads)]
    places = {
        name: int(place)
        for name, place in re.findall(
            rf"#define {upper}_WORD_(\w+) \({upper}_WORDS_OWN \+ (\d+)u\)", text
        )
    }
    out = {}
    for index, body in re.findall(r"/\* call (\d+): .*?\*/\n(.*?)if \(kmx", text, re.S):
        record = np.zeros(w["kChainWords"], np.uint32)
        fields = {}
        for name, value in re.findall(
            r"(command\.a\d|call\.\w+(?:\[\d\])?) = (.*);", body
        ):
            word = re.fullmatch(rf"{upper.lower()}_word\({upper}_WORD_(\w+)\)", value)
            if word:
                fields[name] = thread.words[places[word.group(1)]]
            elif value.endswith("u") and value[:-1].isdigit():
                fields[name] = int(value[:-1])
            elif name == "call.flags":
                fields[name] = sum(FLAGS[flag] for flag in value.split(" | "))
        group = re.search(r"kmx_command\(&command, \w+, (\d+)u, (\d+)u, (\d+)u\)", body)
        for place in range(10):
            record[w["kChainArguments"] + place] = fields.get(f"command.a{place}", 0)
        record[w["kChainGroup"]] = int(group.group(2)) | int(group.group(3)) << 16
        record[w["kChainInfo"]] = (
            fields["call.flags"] | fields["call.word"] << w["kChainTokenShift"]
            | fields["call.by_ranges"] << w["kChainNeedsShift"]
            | fields["call.size"] << w["kChainSizeShift"]
            | fields["call.width"] << w["kChainWidthShift"]
            | fields["call.handler"] << w["kChainHandlerShift"]
        )  # fmt: skip
        for place in range(w["kChainRangeCount"]):
            record[w["kChainRanges"] + place] = (
                fields[f"call.ranges[{place}]"]
                if fields["call.by_ranges"]
                else (
                    kc.Range(fields["call.k_base"], 0, fields["call.k_tiles"]).word()
                    if place == 0
                    else 0
                )
            )
        record[w["kChainNext"]] = fields["call.row"]
        out[int(index)] = record
    return out


def word_equal(program, calls, text, upper, cores=(0, CORES)):
    """The header's records against run.chain's, word by word (`next`: its row), and
    the chain command's cores."""
    w = kc.words()
    run = run_of(program)
    chain = run.chain(calls, cores=cores)
    first = re.search(rf"#define {upper}_CHAIN_\w+_FIRST (\d+)u", text).group(1)
    count = re.search(rf"#define {upper}_CHAIN_\w+_CORES (\d+)u", text).group(1)
    assert (int(first), int(count)) == tuple(chain.group)
    handlers = [run._handler(call.handler)[0] for call in chain.calls]
    want = kc.records(chain.calls, 0, handlers)
    want[:, w["kChainNext"]] &= w["kChainRowMask"]  # the address: the region's
    got = emulated(text, program, w, upper)
    assert sorted(got) == list(range(len(chain.calls)))
    for index, record in got.items():
        differ = np.flatnonzero(record != want[index])
        assert differ.size == 0, (index, differ, record[differ], want[index][differ])


@COMPILER
def test_the_records_are_word_equal_to_run_chain_s():
    program = made()
    calls = program.steps(GIVEN["x"], GIVEN["y"], GIVEN["z"], GIVEN["s"], GIVEN["t"],
                          km.at_submit("factor"), WIDTH, TILES)  # fmt: skip
    word_equal(program, calls, header(program), "THREADED")


class DirectThreaded(DirectSum):
    """The sum takes s direct (km.direct): the record's direct flag, nothing more."""

    @km.chain
    def summed(self, x, s, t, tiles: int):
        self.halves(x, s)
        self.sum(km.direct(s), t, tiles)

    records = km.region(2 * 64)
    thread = km.x280_thread(
        str(HERE / "direct.c"),
        chains={"summed": {"x": "a", "tiles": TILES, "cores": (0, CORES)}},
    )


@COMPILER
def test_a_direct_input_is_its_record_s_flag():
    x = GIVEN["x"]
    program = DirectThreaded(l1=1 << 20, log=quiet, a=x, b=x, out=GIVEN["y"],
                             s=GIVEN["s"], t=GIVEN["t"])  # fmt: skip
    text = header(program)
    calls = program.summed(GIVEN["x"], GIVEN["s"], GIVEN["t"], TILES)
    word_equal(program, calls, text, "DIRECTTHREADED")
    assert GIVEN["s"].sent == "the thread's chain summed"  # no DRAM copy: no host read
    GIVEN["s"].sent = None


class Drains(km.Program):
    @km.handler
    def drain(self, a: km.Tensor):
        km.Stream(a.pages(0, 1)).drain()


class MixedSteps(Steps):
    records = km.region(4 * 64)
    thread = km.x280_thread(str(HERE / "mixed.c"), chains={"steps": SPEC})


class Mixed(MixedSteps, Drains):
    pass


@COMPILER
def test_a_merged_program_s_thread_writes_its_records():
    x = GIVEN["x"]
    program = Mixed(l1=1 << 20, log=quiet, a=x, b=x, out=GIVEN["y"], s=GIVEN["s"],
                    t=GIVEN["t"], thread_tensors={"z": GIVEN["z"]})  # fmt: skip
    calls = program.steps(GIVEN["x"], GIVEN["y"], GIVEN["z"], GIVEN["s"], GIVEN["t"],
                          km.at_submit("factor"), WIDTH, TILES)  # fmt: skip
    word_equal(program, calls, header(program), "MIXED")


def test_a_chain_it_cannot_write_is_refused():
    class NoCores(Steps):
        records = km.region(4 * 64)
        thread = km.x280_thread(
            str(HERE / "chained.c"),
            chains={"steps": {k: v for k, v in SPEC.items() if k != "cores"}},
        )

    class Unknown(Steps):
        records = km.region(4 * 64)
        thread = km.x280_thread(str(HERE / "chained.c"), chains={"nothing": {}})

    with pytest.raises(ProgramError, match=r"chains='steps': give cores="):
        made(NoCores)
    with pytest.raises(ProgramError, match=r"chains='nothing': the class has no"):
        made(Unknown)
