"""The weights' placement "home" (projects/galaxy/weights/home.py) and the loader's
use of it, with no card: every weight placed once, the islands, the arenas' pages,
the budgets, the fold onto fewer islands, the relay first in its arena."""

import json
import os
import re

import numpy as np

from projects.galaxy.weights import home, loader

TILE = {"bfloat16": 2048, "float32": 4096, "bfp8_b": 1088, "bfp4_b": 576}
CARDS = [c for c in range(32) if c != 16]
LANDING = 31 * 31 * (512 + 8) * 2048  # the put batches' landing at 31 islands


def fake_default(layers=2, experts=40):
    """A placement.json in the default's form: per layer FP8 weights in two formats, a
    BF16 and an F32 one, experts 0 .. experts-1 on e mod 31, the head's slices."""
    entries, arenas = [], {}

    def add(name, island, dtype, pages, source, conv="T"):
        at = arenas.get((island, dtype), 0)
        entries.append(
            {
                "name": name,
                "island": island,
                "card": CARDS[island],
                "dtype": dtype,
                "shape": [32 * pages, 32],
                "first": at,
                "pages": pages,
                "source": source,
                "conv": conv,
            }
        )
        arenas[(island, dtype)] = at + pages

    for L in range(layers):
        h = 12 + L
        add(f"layers.{L}.hc_attn_fn@float32", h, "float32", 3, f"L{L}.hc")
        for w, n in (("attn.wq_a", 5), ("ffn.shared_experts.w1", 7)):
            for dtype in ("bfloat16", "bfp8_b"):
                add(f"layers.{L}.{w}@{dtype}", h, dtype, n, f"L{L}.{w}")
        add(f"layers.{L}.attn_norm@bfloat16", h, "bfloat16", 1, f"L{L}.n", "row")
        for e in range(experts):
            for m in ("w1", "w3", "w2"):
                add(
                    f"layers.{L}.ffn.experts.{e}.{m}@bfp4_b",
                    e % 31,
                    "bfp4_b",
                    4,
                    f"L{L}.e{e}.{m}",
                )
    for dtype in ("bfloat16", "bfp8_b"):
        add(f"layers.1.engram.wkv@{dtype}", 6, dtype, 2, "engram")
    for j in range(4):
        add(f"head.s{j}@bfloat16", j, "bfloat16", 9, "head", f"T rows {j}")
    add("norm@bfloat16", 0, "bfloat16", 1, "norm", "row")
    return {
        "version": 1,
        "cards": CARDS,
        "tile_bytes": TILE,
        "arenas": [
            {
                "island": i,
                "card": CARDS[i],
                "dtype": d,
                "pages": n,
                "bytes": n * TILE[d],
            }
            for (i, d), n in sorted(arenas.items())
        ],
        "entries": entries,
    }


def weight(name):
    return name.split("@")[0]


def check_table(default, p):
    """What every home placement holds to: each weight once, in its kept format, on its
    island; each arena's runs back to back from page 0 (the relay first on island 0);
    the cache positions the default's."""
    by_name = {e["name"]: e for e in default["entries"]}
    formats = {}
    for e in default["entries"]:
        formats.setdefault(weight(e["name"]), set()).add(e["dtype"])
    names = [weight(e["name"]) for e in p["entries"]]
    assert len(names) == len(set(names))
    assert set(names) == {weight(e["name"]) for e in default["entries"]}
    for e in p["entries"]:
        d = by_name[e["name"]]
        assert e["cache"] == [d["island"], d["first"]]
        assert (e["shape"], e["pages"], e["source"], e["conv"]) == (
            d["shape"],
            d["pages"],
            d["source"],
            d["conv"],
        )
        m = re.match(r"layers\.\d+\.ffn\.experts\.(\d+)\.", e["name"])
        assert e["island"] == (1 + int(m.group(1)) % 30 if m else 0)
        assert e["card"] == CARDS[e["island"]]
        if len(formats[weight(e["name"])]) > 1:
            exact = ".engram.wkv@" in e["name"] or ".attn.indexer.wq_b@" in e["name"]
            assert e["dtype"] == ("bfloat16" if exact else "bfp8_b")
    (relay,) = p["relay"]
    assert (relay["island"], relay["dtype"], relay["first"]) == (0, "bfp4_b", 0)
    run = [by_name[n] for n in home.RELAY_SOURCE]
    assert relay["cache"] == [run[0]["island"], run[0]["first"]]
    assert relay["pages"] == sum(e["pages"] for e in run)
    runs = {}
    for e in p["relay"] + p["entries"]:
        runs.setdefault((e["island"], e["dtype"]), []).append((e["first"], e["pages"]))
    arenas = {(a["island"], a["dtype"]): a for a in p["arenas"]}
    assert set(runs) == set(arenas)
    for key, got in runs.items():
        at = 0
        for first, pages in sorted(got):
            assert first == at
            at += pages
        assert arenas[key]["pages"] == at
        assert arenas[key]["bytes"] == at * TILE[key[1]]


def test_fake_table():
    default = fake_default()
    p = home.plan(default)
    check_table(default, p)
    assert p["name"] == p["fold"] == "home"
    assert all(e["island"] for e in p["entries"] if ".experts." in e["name"])


def test_fold():
    p = {"fold": "home"}
    assert [loader._fold(p, i, 8) for i in range(9)] == [0, 1, 2, 3, 4, 5, 6, 7, 1]
    assert {loader._fold(p, i, 8) for i in range(1, 31)} == set(range(1, 8))
    assert [loader._fold(p, i, 31) for i in range(31)] == list(range(31))
    assert loader._fold(p, 30, 1) == 0
    assert [loader._fold({}, i, 8) for i in (0, 8, 30)] == [0, 0, 6]


def test_placement_path(monkeypatch):
    monkeypatch.setattr(loader, "LLM", "/llm")
    assert loader.placement_path("") == "/llm/placement.json"
    assert loader.placement_path("home") == "/llm/placement-home.json"
    monkeypatch.setenv("GALAXY_PLACEMENT", "home")
    assert loader.placement_path() == "/llm/placement-home.json"


REAL = os.path.join(loader.LLM, "placement.json")


def real_budget():
    """The released model's placement: island 0 and every expert island fit the chip
    beside the landing; 13 or 12 experts a layer on islands 1-30."""
    with open(REAL) as f:
        default = json.load(f)
    p = home.plan(default)
    check_table(default, p)
    per = home.budget(p)
    assert set(per) == set(range(31))
    assert all(b + LANDING < home.DRAM_BYTES for b in per.values())
    assert per[0] == 9_384_378_368
    assert max(per[i] for i in range(1, 31)) == 13 * 40 * 34_560 * 576
    held = {}
    for e in p["entries"]:
        m = re.match(r"layers\.0\.ffn\.experts\.(\d+)\.w1@", e["name"])
        if m:
            held[e["island"]] = held.get(e["island"], 0) + 1
    assert sorted(held.values()) == [12] * 6 + [13] * 24


# The converted placement lives beside the model's weights: the test exists only on
# that machine (a skip elsewhere would count as a test that stopped running).
if os.path.exists(REAL):
    test_real_budget = real_budget


class _Part:
    def __init__(self, pages, tile):
        self.bytes = np.zeros((pages, tile), np.uint8)


class _Arena:
    def __init__(self, island, pages, tile):
        self.parts = {island: _Part(pages, tile)}
        self.address = island
        self.written = self.sent = None


class _One:
    def write(self, part, rows, first=0):
        part.bytes[first : first + len(rows)] = rows


class _Graph:
    def first(self, island):
        return 75 * island

    def count(self, island):
        return 75


class _Session:
    """What the loader asks of a session of `count` islands."""

    def __init__(self, count):
        self.sessions = [_One() for _ in range(count)]
        self.graph = _Graph()

    def empty(self, shape, dtype, form, split, group):
        pages = shape[0] // 32
        return _Arena(group[0] // 75, pages, TILE[dtype])


def test_load_fold_and_relay(tmp_path):
    """A subset load onto 8 islands: experts on islands 1-7, the rest on 0, the relay
    first in island 0's bfp4_b arena; each page the bytes of its cache position."""
    default = fake_default()
    p = home.plan(default)
    rng = np.random.default_rng(0)
    for a in default["arenas"]:
        raw = rng.integers(0, 256, (a["pages"], TILE[a["dtype"]]), np.uint8)
        raw.tofile(tmp_path / f"i{a['island']:02d}-{a['dtype']}.bin")
    path = tmp_path / "placement-home.json"
    path.write_text(json.dumps(p))
    names = [e["name"] for e in p["entries"] if e["name"].startswith("layers.1.")]
    W = loader.load(
        _Session(8), names, placement=str(path), cache=str(tmp_path), verify=None,
        log=lambda *a: None,
    )  # fmt: skip
    assert W["relay@bfp4_b"].first == 0 and W["relay@bfp4_b"].island == 0
    for name in names:
        h = W[name]
        assert h.island == (loader._fold(p, by_island(p, name), 8))
        assert (h.island == 0) == (".experts." not in name)
    for e in p["relay"] + [x for x in p["entries"] if x["name"] in names]:
        h = W[e["name"]]
        src, first = e["cache"]
        tile = TILE[e["dtype"]]
        want = np.fromfile(tmp_path / f"i{src:02d}-{e['dtype']}.bin", np.uint8)
        want = want.reshape(-1, tile)[first : first + e["pages"]]
        got = h.arena.parts[h.island].bytes[h.first : h.first + h.pages]
        assert np.array_equal(got, want), e["name"]


def by_island(p, name):
    return next(e["island"] for e in p["entries"] if e["name"] == name)
