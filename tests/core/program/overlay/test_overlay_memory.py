"""A part that copies a struct, built with the cores' toolchain: the compiler calls
memcpy for the copy at Os, which the plain link sends to the persistent kernel's copy
(refused by name) and the part's own archive keeps in the part."""

import subprocess

import pytest

from komorebi import _env
from komorebi.device import native
from komorebi.errors import ProgramError
from komorebi.program import overlay

FLAGS = ("-mcpu=tt-wh", "-mabi=ilp32")
KERNEL = """
void* memcpy(void* d, const void* s, unsigned n) {
    char* a = d; const char* b = s; while (n--) *a++ = *b++; return d;
}
void* memset(void* d, int c, unsigned n) {
    char* a = d; while (n--) *a++ = c; return d;
}
volatile int sink;
int main(void) {
    char x[100], y[100];
    memcpy(x, y, sink); memset(x, 0, sink); sink = x[3]; return 0;
}
"""
PART = """
struct Big { unsigned a[16]; unsigned b; };
__attribute__((noinline)) Big take(const Big& x) { Big y = x; y.b++; return y; }
extern "C" __attribute__((section(".km_overlay_entry"), used))
unsigned km_overlay_entry(void* p) {
    Big v = take(*static_cast<const Big*>(p));
    *static_cast<Big*>(p) = v;
    return v.b;
}
"""


def recipe(tmp_path):
    """A writer's Recipe at Os for a stand-in kernel whose memcpy and memset are in
    its text at Wormhole's NCRISC code address."""
    try:
        tree = native.metal_tree()
    except Exception:  # noqa: BLE001 (no tree: the test is skipped)
        pytest.skip("needs TT_METAL_HOME")
    gpp = _env.sfpi_bin(tree) / "riscv-tt-elf-g++"
    if not gpp.is_file():
        pytest.skip(f"needs {gpp.name}")
    (tmp_path / "kernel.c").write_text(KERNEL)
    elf = tmp_path / "kernel.elf"
    subprocess.run(
        [str(gpp.with_name("riscv-tt-elf-gcc")), *FLAGS, "-Os", "-nostdlib"]
        + ["-Wl,-Ttext=0xffc00000", "-Wl,-e,main", str(tmp_path / "kernel.c")]
        + ["-o", str(elf)],
        check=True,
    )
    cflags = (*FLAGS, "-flto", "-fno-exceptions", "-fno-rtti", "-std=c++17")
    lflags = (*FLAGS, "-flto", "-nostdlib")
    return overlay.Recipe(
        "ncrisc", elf, (str(gpp),), "Os", cflags, (), lflags, "Os", ()
    )


def links(made, obj, folder, memory):
    """The part linked at its two bases, with part_memory's archive `memory` or none."""
    out = []
    for base in (overlay.LINK_BASE, overlay.LINK_BASE + overlay.ALIGN):
        script, elf = folder / f"part-{base:x}.ld", folder / f"part-{base:x}.elf"
        script.write_text(overlay.SCRIPT.format(base=base, length=overlay.LINK_LENGTH))
        overlay.run(made.link_command(obj, script, elf, memory))
        out.append(overlay.linked_facts(made, elf, base, ()))
    return out


@pytest.mark.tree
def test_a_part_that_copies_a_struct_takes_its_own_memcpy_and_no_memset(tmp_path):
    made = recipe(tmp_path)
    (tmp_path / "part.cpp").write_text(PART)
    obj = tmp_path / "part.o"
    overlay.run(made.compile_command(tmp_path / "part.cpp", obj, {}))
    persistent = overlay.persistent_facts(made)
    plain_folder, own_folder = tmp_path / "plain", tmp_path / "own"
    plain_folder.mkdir()
    own_folder.mkdir()
    plain = links(made, obj, plain_folder, None)
    kernel_memcpy = persistent.symbols["memcpy"][0]
    assert ("memcpy", kernel_memcpy) in plain[0].calls
    with pytest.raises(ProgramError, match=rf"calls \['memcpy at {kernel_memcpy:#x}'"):
        overlay.check("copy", "writer", *plain, persistent)
    own = links(made, obj, own_folder, overlay.part_memory(made, tmp_path / "cache"))
    overlay.check("copy", "writer", *own, persistent)
    assert own[0].calls == ()
    assert "__wrap_memcpy" in own[0].symbols and "__wrap_memset" not in own[0].symbols
