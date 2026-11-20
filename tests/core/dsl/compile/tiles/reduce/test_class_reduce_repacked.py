"""A reduce's sum that compute frees at the leader (packed again in another format, or
held across a loop): mover B moves its own read pointer of the sum's buffer past those
pages (km::passed_front), else its next read there, a later handler's write or a
sender's emit, finds the page a slot off (no card)."""

from class_lines_fixture import Pages

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.program import compute_source, movement_source


class Repacked(km.Program):
    @km.handler
    def h(self, x: km.Tensor, out: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(x.pages(self.rank, 1)):
            total += tile
        whole = km.reduce(total, form="tree", group=4)
        if self.rank == self.leader:
            out.pages(self.rank // 4, 1).write(whole)


class Held(km.Program):
    @km.handler
    def h(self, x: km.Tensor, out: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(x.pages(0, 2)):
            total += tile
        whole = km.reduce(total, form="tree")
        for i in range(2):
            if self.rank == self.leader:
                out.pages(i, 1).write(whole)


def built(cls, dtype):
    page = 4096 if dtype == "float32" else 2048
    tensors = {
        "x": Tensor(Pages(4096), (256, 128), "float32", "tile"),
        "out": Tensor(Pages(page), (256, 128), dtype, "tile"),
    }
    compiled = compile_class(cls, {}, tensors)
    return movement_source(compiled), compute_source(compiled)


def test_a_sum_packed_again_moves_mover_b_s_pointer_past_it():
    movement, compute = built(Repacked, "bfloat16")
    assert "cb_pop_front(kOut0, 1);" in compute  # compute frees the float32 sum
    write = movement[movement.index("km::write_pages(") :]
    freed = write.index("cb_pop_front(kOut1, 1);")
    passed = write.index("km::passed_front(kOut0, 1);")
    assert freed < passed < write.index("#endif")  # after the write, mover B's
    before = movement[: movement.index("km::write_pages(")]
    assert "COMPILE_FOR_BRISC)  // mover B" in before


def test_a_sum_written_as_it_is_needs_no_follow():
    movement, compute = built(Repacked, "float32")
    assert "km::passed_front(" not in movement
    assert "cb_pop_front(kOut0, 1);" not in compute  # mover B frees it


def test_a_held_sum_freed_after_its_loop_moves_mover_b_s_pointer_after_it():
    movement, compute = built(Held, "float32")
    loop = movement[movement.index("for (uint32_t i = 0;") :]
    passed = loop.index("km::passed_front(kOut0, 1);")
    assert loop.index("km::write_pages(") < passed
    assert movement.count("km::passed_front(") == 1
    after = compute[compute.index("held across the loop, freed") :]
    assert after.count("cb_pop_front(kOut0, 1);") == 1
