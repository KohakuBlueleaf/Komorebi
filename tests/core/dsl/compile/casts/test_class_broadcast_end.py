"""Where compute's padding pop of a broadcast (`km::broadcast_end`) stands in the
generated text (no card): after the block's last use, in the kernel form and in the
runtime's, for a cast outside every loop and for one in a loop; a broadcast casts
outside a loop once."""

import pytest
from class_lines_fixture import tensor

import komorebi as km
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class


def twice():
    class Twice(km.Program):
        @km.handler
        def cast(self, x: km.Tensor, out: km.Tensor):
            row = km.Broadcast(self, along="row", window=2)
            if row.sender():
                first = row.cast(x.pages(self.y, 1).read())
            else:
                first = row.arrive()
            for _tile in first:
                pass
            if row.sender():
                second = row.cast(x.pages(self.y + 1, 1).read())
            else:
                second = row.arrive()
            for _tile in second:
                pass

    return Twice


def test_a_broadcast_casts_outside_a_loop_once():
    with pytest.raises(Refused, match=r"casts outside a loop already"):
        compile_class(twice(), {}, {"x": tensor("bfloat16"), "out": tensor("bfloat16")})
