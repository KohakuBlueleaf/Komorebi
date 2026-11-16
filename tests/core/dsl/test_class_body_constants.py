"""A class's constant taken by an inline body: `takes=[self.K]` gives the body a
`constexpr uint32_t K` of this program's value, on compute and on a mover, a constant
made of others too (no card)."""

from chain_fixture import quiet, tensor

import komorebi as km

BODY = "use(K, BLOCK);"


class Sized(km.Program):
    """One body on compute and one on mover B taking the class's constants."""

    K = km.constant(4)
    BLOCK = K * 2

    @km.handler
    def run(self, a: km.Tensor):
        km.cxx(BODY, takes=[self.K, self.BLOCK])
        km.cxx(BODY, takes=[self.K, self.BLOCK, a], on="mover_b")


def test_a_constant_is_a_constexpr_of_its_name_in_the_body():
    program = Sized(l1=1 << 20, log=quiet, a=tensor(8, 0x10000))
    folder = program.built.folder
    for kernel in ("compute.cpp", "movement.cpp"):
        text = (folder / kernel).read_text()
        body = text[: text.index(BODY)]
        assert "constexpr uint32_t K = 4;" in body
        assert "constexpr uint32_t BLOCK = 8;" in body


def test_each_program_has_its_own_value():
    program = Sized(l1=1 << 20, log=quiet, a=tensor(8, 0x10000), K=3)
    text = (program.built.folder / "compute.cpp").read_text()
    assert "constexpr uint32_t K = 3;" in text
    assert "constexpr uint32_t BLOCK = 6;" in text
