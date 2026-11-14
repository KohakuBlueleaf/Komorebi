"""The pull's word and example 18 on the host: KMX_PULL_WORD and kmx_pull as C sees
them, a unit's kind the same in C and numpy, and the stripes and split the host gives
the thread (no card)."""

import importlib.util
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from komorebi import thread as thread_module

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "18_thread_split"
SPEC = importlib.util.spec_from_file_location(
    "thread_split", EXAMPLE / "thread_split.py"
)
split = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(split)
KMX_DIRECTORY = thread_module.native.kmx_directory()
CHECK = r"""
#include <stddef.h>
#include <stdio.h>
#include "kmx.h"
#include "split.h"
int main(void) {
    const uint64_t answer = KMX_PULL_WORD(7u, TS_PULL_WORD);
    printf("%u %u %u\n", (unsigned)offsetof(kmx_pull, pick_word),
           (uint32_t)answer, (uint32_t)(answer >> 32));
    for (uint32_t unit = 0; unit < 300; unit++) {
        printf("%u", TS_KIND(unit, 16u));
    }
    printf("\n");
    return 0;
}
"""


@pytest.mark.skipif(shutil.which("cc") is None, reason="needs a C compiler")
def test_a_pick_word_answer_is_the_range_low_and_the_word_high(tmp_path):
    (tmp_path / "check.c").write_text(CHECK)
    includes = [KMX_DIRECTORY, KMX_DIRECTORY.parents[1] / "include", EXAMPLE]
    exe = tmp_path / "check"
    subprocess.run(
        ["cc", "-std=c11", "-Wall", "-Werror", *[f"-I{path}" for path in includes]]
        + [str(tmp_path / "check.c"), "-o", str(exe)],
        check=True,
    )
    words, kinds = subprocess.run(
        [exe], capture_output=True, text=True, check=True
    ).stdout.split("\n")[:2]
    assert words.split() == ["32", "7", str(split.C["TS_PULL_WORD"])]  # after `arg`
    assert kinds == "".join(map(str, split.kinds(300, 16)))


def test_the_stripes_cover_every_unit_once_in_whole_ranges():
    bounds = split.stripes(1000, 4, 120)
    assert bounds[0][0] == 0 and bounds[-1][1] == 1000
    assert all(end == first for (_, end), (first, _) in zip(bounds, bounds[1:]))
    assert all(first % 4 == 0 for first, _ in bounds)


def test_the_host_split_is_each_stripe_s_commonest_kind_packed_two_bits_a_core():
    kinds = np.array([0, 0, 1, 2, 2, 2, 1, 1, 1, 1])
    choice = split.host_split(kinds, [(0, 3), (3, 6), (6, 10)])
    assert choice == [1, 3, 2]
    words = split.split_words(choice + [1] * 17)
    assert len(words) == 8 and words[0] == 1 | 3 << 2 | 2 << 4 | 0x55555540
    assert words[1] == 0b01010101  # cores 16 to 19
