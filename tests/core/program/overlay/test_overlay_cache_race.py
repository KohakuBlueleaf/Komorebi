"""Two builds of one overlay started together share its cache: each file a build
writes there is whole when the other reads it (a stand-in toolchain writes slowly)."""

import json
import multiprocessing
import sys
import time

from komorebi.program import overlay

TOOLS = r"""
import os, re, sys, time
from pathlib import Path

tool, args = sys.argv[1], sys.argv[2:]


def slowly(path, data):  # in pieces, as a compiler writes: a reader sees it half done
    with open(path, "wb") as out:
        for at in range(0, len(data), 64):
            out.write(data[at : at + 64])
            out.flush()
            time.sleep(0.002)


def base_of(elf):
    return int(Path(elf).read_bytes().split(b"\n")[0].split(b"=")[1], 16)


if tool == "g++":
    source, out = args[-3], args[-1]
    text = Path(source).read_bytes()
    if "-E" in args:
        slowly(out, b'# 1 "%s"\n' % source.encode() + text * 40)
    elif "-c" in args:
        slowly(out, b"OBJ\n" + text * 40)
    else:
        script = next(word[2:] for word in args if word.startswith("-T"))
        base = re.search(r"ORIGIN = (0x[0-9a-f]+)", Path(script).read_text())[1]
        obj = next(word for word in args if word.endswith(".o"))
        first = Path(obj).read_bytes()
        mark = os.environ.get("RACE_MARK")
        if mark:  # read again after a build beside it replaced part.o, as lto-wrapper
            Path(mark).write_text("linking")
            while not Path(mark + ".replaced").exists():
                time.sleep(0.01)
            if Path(obj).read_bytes() != first:
                sys.exit("lto1: resolution sub id not in object file")
        slowly(out, b"BASE=%s\n" % base.encode() + first)
elif tool == "readelf":
    flags, elf = args
    if flags == "-SW" and elf.endswith("kernel.elf"):
        print("  [ 1] .text PROGBITS 00000000 001000 000100 00 AX 0 0 4")
    elif flags == "-sW" and not elf.endswith("kernel.elf"):
        print(f"  1: {base_of(elf):08x} 4 FUNC GLOBAL DEFAULT 1 km_overlay_entry")
    elif flags == "-hlSW":
        at = base_of(elf)
        print(f"  LOAD 0x001000 {at:#010x} {at:#010x} 0x000040 0x000040 RWE 0x1000")
elif tool == "ar":
    slowly(args[1], b"!<arch>\n")
elif tool == "objcopy":
    slowly(args[-1], Path(args[-2]).read_bytes().split(b"\n", 1)[1][:64])
"""


def toolchain(folder):
    """A Recipe whose g++ and binutils are the stand-ins above (objdump: no calls)."""
    script = folder / "tools.py"
    script.write_text(TOOLS)
    for name in ("g++", "readelf", "gcc-nm", "objcopy", "objdump", "ar"):
        path = folder / f"fake-{name}"
        path.write_text(f'#!/bin/sh\nexec {sys.executable} {script} {name} "$@"\n')
        path.chmod(0o755)
    elf = folder / "kernel.elf"
    elf.write_bytes(b"\x7fELF persistent")
    compiler = (str(folder / "fake-g++"),)
    return overlay.Recipe("brisc", elf, compiler, "Os", (), (), (), "Os", ())


def build(recipe, folder, start, out, second):
    made = overlay.declare("race", {"reader": folder / "reader.cpp"}, 2)
    start.wait()
    sources, waited = folder / "cache" / "sources", time.monotonic()
    while second and time.monotonic() - waited < 30:  # the first's -E far on
        if any(path.stat().st_size > 8192 for path in sources.glob("*.ii")):
            break
        time.sleep(0.001)
    linked, elf, _ = overlay.build_part_cached(
        recipe, made, "reader", "reader", folder / "cache"
    )
    out.put((linked.image, str(elf), elf.read_bytes()))


def test_two_builds_of_one_overlay_started_together_end_with_one_part(tmp_path):
    (tmp_path / "reader.cpp").write_text("int f() { return 1; }\n" * 50)
    recipe = toolchain(tmp_path)
    context = multiprocessing.get_context("fork")
    start, out = context.Barrier(2), context.Queue()
    builds = [
        context.Process(target=build, args=(recipe, tmp_path, start, out, second))
        for second in (False, True)
    ]
    for each in builds:
        each.start()
    made = [out.get(timeout=60) for _ in builds]
    for each in builds:
        each.join(60)
        assert each.exitcode == 0
    assert made[0] == made[1]  # one ELF, at one path, its image as linked
    (index,) = (tmp_path / "cache" / "built").iterdir()
    assert json.loads(index.read_text())["elf"] == made[0][1]
    assert not [path.name for path in tmp_path.rglob(".*-*")]  # no temporary left


def link_alone(recipe, folder):
    made = overlay.declare("race", {"reader": folder / "reader.cpp"}, 2)
    overlay.build_part(recipe, made, "reader", "reader", folder / "cache")


def test_a_part_links_from_its_own_name_while_a_build_beside_replaces_it(
    tmp_path, monkeypatch
):
    """The link reads the object twice (ld, then lto-wrapper): a build beside this
    one that replaces part.o in between must not change what the second read sees."""
    (tmp_path / "reader.cpp").write_text("int f() { return 1; }\n" * 50)
    recipe = toolchain(tmp_path)
    mark = tmp_path / "mark"
    monkeypatch.setenv("RACE_MARK", str(mark))
    context = multiprocessing.get_context("fork")
    builder = context.Process(target=link_alone, args=(recipe, tmp_path))
    builder.start()
    waited = time.monotonic()
    while not mark.exists() and time.monotonic() - waited < 60:
        time.sleep(0.01)
    (obj,) = (tmp_path / "cache").glob("*/part.o")
    other = obj.with_name(".part-other.o")
    other.write_bytes(b"OBJ\nanother build's copy\n")
    other.replace(obj)
    (tmp_path / "mark.replaced").write_text("")
    builder.join(60)
    assert builder.exitcode == 0
