"""A line's last word as an X280 rank seals it (kmx_line.h, C) against the one the link
cores check it with (km_pages.hpp line_end, C++): the two folds agree on every line
(no card)."""

import random
import re
import shutil
import subprocess

import pytest

from komorebi.program import thread as thread_module

NATIVE = thread_module.native.kmx_directory().parents[1]
PAGES = NATIVE / "include" / "komorebi" / "device" / "links" / "km_pages.hpp"

MAIN = r"""
#include <cstdint>
#include <cstdio>
extern "C" {
#include "kmx_line.h"
}
namespace km {
%s
}
int main() {
    uint32_t state = %du, line[16];
    for (int n = 0; n < 20000; n++) {
        for (int w = 0; w < 16; w++) {
            state = state * 1664525u + 1013904223u;
            line[w] = n %% 3 == 0 ? (state >> 28) : state;
        }
        const uint32_t number = n %% 5 == 0 ? (uint32_t)n : line[0];
        if (km::line_end(line, number) != kmx_line_end(line, number)) {
            std::printf("differ at %%d\n", n);
            return 1;
        }
    }
    std::printf("ok\n");
    return 0;
}
"""


def device_fold():
    """km_pages.hpp's kLineFirst / kLineLast and line_end, as the file has them."""
    text = PAGES.read_text()
    start = text.index("    constexpr uint32_t kLineFirst")
    tail = "        return check;\n    }\n"
    end = text.index(tail, start) + len(tail)
    fold = text[start:end]
    assert "inline uint32_t line_end" in fold
    return re.sub(r"#pragma GCC unroll \d+\n", "", fold)


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a C++ compiler")
def test_the_x280_fold_is_the_link_cores_fold(tmp_path):
    source = tmp_path / "fold.cpp"
    source.write_text(MAIN % (device_fold(), random.getrandbits(31)))
    exe = tmp_path / "fold"
    includes = [
        f"-I{NATIVE / 'include'}",
        f"-I{thread_module.native.kmx_directory()}",
    ]
    subprocess.run(
        ["c++", "-std=c++17", "-Wall", "-Werror", *includes, str(source), "-o", exe],
        check=True,
    )
    assert subprocess.run([exe], capture_output=True, text=True).stdout == "ok\n"
