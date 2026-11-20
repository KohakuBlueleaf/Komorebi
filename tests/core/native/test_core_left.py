"""A give-up's rule for a core that left its command (kmd_core_left): only a core whose
control area carries the magic of this run's kernel (its tag) may have left; a core read
as zeros or with another run's tag is in no known state, so the recovery resets it."""

import itertools
import shutil
import subprocess
from pathlib import Path

import pytest

from komorebi import abi

ROOT = Path(__file__).resolve().parents[3]
NATIVE = ROOT / "src" / "komorebi_native"
H = abi.headers()
SEQUENCE, TAG = 7, 0x6543  # the command given up, the run's tag

LEFT_CHECK = r"""
#include <stdio.h>
#include "km_dispatch.h"

/* Per input line: the magic word, then KM_LIVE_ROLES live sequence words. */
int main(void) {
    unsigned magic, words[KM_LIVE_ROLES];
    while (scanf("%u", &magic) == 1) {
        uint32_t sequences[KM_LIVE_ROLES];
        for (unsigned role = 0; role < KM_LIVE_ROLES; role++) {
            if (scanf("%u", &words[role]) != 1) {
                return 1;
            }
            sequences[role] = words[role];
        }
        printf("%d\n", kmd_core_left(magic, 0x6543u, sequences, 7u));
    }
    return 0;
}
"""


def model(magic, sequences):
    """Left: this run's magic (top byte and tag) and no kernel still in the command."""
    there = magic >> 24 == H["kStateMagic"] >> 24 and (magic >> 8) & 0xFFFF == TAG
    return int(there and SEQUENCE not in sequences)


@pytest.mark.skipif(shutil.which("cc") is None, reason="needs a C compiler")
def test_a_core_without_its_magic_has_not_left_its_command(tmp_path):
    source_file = tmp_path / "left.c"
    source_file.write_text(LEFT_CHECK)
    exe = tmp_path / "left"
    include = ["-I", str(NATIVE / "include"), "-I", str(NATIVE / "x280" / "kernel")]
    command = ["cc", "-std=c11", "-Wall", "-Werror", *include]
    subprocess.run([*command, str(source_file), "-o", str(exe)], check=True)
    magics = (
        0,
        H["kStateMagic"] | TAG << 8 | 36,
        H["kStateMagic"] | (TAG + 2) << 8 | 36,  # another run's kernel
        0x12000000 | TAG << 8 | 36,
    )
    roles = H["KM_LIVE_ROLES"]
    cases = [
        (magic, sequences)
        for magic in magics
        for sequences in itertools.product((0, SEQUENCE - 1, SEQUENCE), repeat=roles)
    ]
    lines = [" ".join(map(str, (magic, *sequences))) for magic, sequences in cases]
    out = subprocess.run(
        [str(exe)], input="\n".join(lines), capture_output=True, text=True, check=True
    )
    got = [int(line) for line in out.stdout.splitlines()]
    assert got == [model(magic, sequences) for magic, sequences in cases]
    assert model(0, (0,) * roles) == 0  # all zeros: the area not there, not "left"
