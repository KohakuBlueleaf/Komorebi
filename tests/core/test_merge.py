"""Programs merged into one (km.merge): the numbering, a shared and a clashing buffer,
the pick of a source to become an overlay and the markers, and each refusal."""

import pytest

import komorebi as km
from komorebi.errors import L1LayoutError, ProgramError
from komorebi.run import Run

PROGRAM = "void km_program(km::{kind}& kernel, const km_command_t& command) {{}}\n"
IN = km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16)
OUT = km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2)


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("KMR_CACHE", str(tmp_path / "cache"))


def made(tmp_path, name, handlers, buffers=(IN, OUT), form="program", **options):
    folder = tmp_path / name
    folder.mkdir()
    for kernel, kind in (("reader", "Reader"), ("writer", "Writer"), ("compute", "")):
        text = PROGRAM.format(kind=kind or "Compute")
        (folder / f"{kernel}.cpp").write_text(
            text if form == "program" else "void kernel_main() {}\n"
        )
    return km.Program(
        name, kernels=folder, buffers=list(buffers), handlers=handlers, **options
    )


def test_numbering_names_and_a_source_handle(tmp_path):
    sums = made(tmp_path, "sums", {"sum": 1, "empty": 2})
    scales = made(tmp_path, "scales", {"scale": 1, "empty": 2})
    merged = km.merge(sums, scales, log=lambda line: None)
    assert merged.numbers == {1: (0, 1), 2: (0, 2), 3: (1, 1), 4: (1, 2)}
    assert merged.handler("scale") == 3 and merged.handler(sums, "empty") == 2
    assert merged.handler(scales.handler("empty")) == 4
    with pytest.raises(ValueError, match="'empty' is in sums and scales"):
        merged.handler("empty")
    run = Run.__new__(Run)
    run.programs = [merged]
    assert run._handler(scales.handler("scale")) == (3, 0)
    text = merged.kernels["reader"].read_text()
    assert "case 3u:  // scales's 1" in text and "km_merged_own(command, 1u)" in text
    assert merged.buffers == [IN, OUT]  # one shared buffer of each index


def test_a_clashing_buffer_and_the_count(tmp_path):
    other = km.Buffer(index=16, format="float32", page_bytes=4096, pages=2)
    sums = made(tmp_path, "sums", {"sum": 1})
    wide = made(tmp_path, "wide", {"widen": 1}, buffers=(IN, other))
    with pytest.raises(L1LayoutError, match="buffer 16 is declared by sums as"):
        km.merge(sums, wide, log=lambda line: None)
    many = made(tmp_path, "many", {f"h{n}": n for n in range(1, 239)})
    with pytest.raises(ProgramError, match="numbered up to 240; a program numbers"):
        km.merge(many, sums, made(tmp_path, "two", {"a": 1}), log=lambda line: None)


def test_the_pick_and_the_markers(tmp_path):
    small = made(tmp_path, "small", {"a": 1})
    large = made(tmp_path, "large", {"b": 1})
    third = made(tmp_path, "third", {"c": 1})
    merged = km.merge(small, large, third, resident=[third], log=lambda line: None)
    merged.part_bytes = {0: {"reader": 64}, 1: {"reader": 4096}, 2: {"reader": 8192}}
    assert merged.spill(lambda line: None) and merged.code_overlays == {1}
    assert "km_merged_overlay(kernel, 1u" in merged.kernels["writer"].read_text()
    assert merged.spill(lambda line: None) and merged.code_overlays == {0, 1}
    assert not merged.spill(lambda line: None)  # third is kept resident
    marked = km.merge(small, large, overlay=[small], log=lambda line: None)
    assert marked.code_overlays == {0} and marked.overlay_slots.count["reader"] == 3
    with pytest.raises(ProgramError, match="loaded alone"):
        marked.part_overlays(switch=False, alone=False)
    with pytest.raises(ProgramError, match="one switch the build cannot cut"):
        km.merge(small, large, overlay=["b"], log=lambda line: None)
    with pytest.raises(ProgramError, match="marked both overlay and resident"):
        km.merge(small, large, overlay=[small], resident=[small], log=print)


def test_what_is_refused(tmp_path):
    sums = made(tmp_path, "sums", {"sum": 1})
    hand = made(tmp_path, "hand", {"map": 1}, form="kernel")
    with pytest.raises(ProgramError, match="merges in the runtime's form"):
        km.merge(sums, hand)
    with pytest.raises(ValueError, match="two or more"):
        km.merge(sums)
    threaded = made(tmp_path, "threaded", {"go": 1})
    threaded.thread(sources=[tmp_path / "t.c"], regions={"state": 64})
    with pytest.raises(ProgramError, match="merge it first"):
        km.merge(sums, threaded)
    extra = made(tmp_path, "extra", {"go": 1})
    extra.overlay("more", {"reader": tmp_path / "extra" / "reader.cpp"}, number=1)
    with pytest.raises(ProgramError, match="overlays of its own"):
        km.merge(sums, extra)
    one = made(tmp_path, "one", {"a": 1}, args=lambda core: ([], [], []))
    two = made(tmp_path, "two", {"b": 1}, args=lambda core: ([], [], []))
    with pytest.raises(ProgramError, match="one source of a merged program"):
        km.merge(one, two)
