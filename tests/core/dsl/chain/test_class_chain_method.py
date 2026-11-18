"""@km.chain: a method of calls of the class's handlers in order, read when the class
is made (a loop of a constant count unrolled), called on a program to give the same
calls as the host's list, so the same records; each line it cannot hold refused at
its line (no card)."""

import importlib.util

import numpy as np
import pytest
from chain_fixture import (
    CORES,
    TILES,
    WIDTH,
    Steps,
    bits,
    host_list,
    made,
    run_of,
    tensors,
)

from komorebi.errors import ProgramError

GIVEN = tensors()


def records(run):
    return np.frombuffer(run.backend.blocks[-1].data, np.uint32).reshape(-1, 16)


def test_the_method_gives_the_host_lists_records():
    program = made(Steps, GIVEN)
    run = run_of(program)
    listed = run.chain(host_list(GIVEN, 5), cores=(0, CORES))
    from_list = records(run).copy()
    x, y, z, s, t = (GIVEN[name] for name in "xyzst")
    method = run.chain(
        program.steps(x, y, z, s, t, bits(5), WIDTH, TILES), cores=(0, CORES)
    )
    numbered = [[c._replace(handler=int(c.handler)) for c in chain.calls]
                for chain in (listed, method)]  # fmt: skip
    assert numbered[0] == numbered[1]
    from_method = records(run)
    next_word = 15  # each record's next address: the same chain at the same block
    assert (
        np.delete(from_list, next_word, 1) == np.delete(from_method, next_word, 1)
    ).all()


def test_a_loop_of_a_constant_count_is_unrolled():
    program = made(Steps, GIVEN)
    calls = program.twice(GIVEN["x"], GIVEN["y"], bits(2), WIDTH)
    assert [call.method.name for call in calls] == ["scale", "scale"]


def chain_class(folder, body):
    """A subclass of the handlers whose km.chain method is `body`, in a file of its
    own (the build reads a class's source)."""
    source = (
        "import komorebi as km\n"
        "from chain_fixture import Handlers\n\n"
        "LIMIT = 300\n\n\n"
        "class Bad(Handlers):\n"
        "    @km.chain\n"
        "    def steps(self, x, y, n: int):\n"
        + "".join(f"        {line}\n" for line in body)
    )
    path = folder / "bad_chain.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(
        f"bad_chain_{abs(hash(source))}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Bad


@pytest.mark.parametrize(
    ("body", "why"),
    [
        (["if n:", "    self.scale(x, y, 1, 2)"], "a branch or a loop on data"),
        (["while n:", "    self.scale(x, y, 1, 2)"], "a branch or a loop on data"),
        (["print(x)"], "a call of something other than a handler of the class"),
        (
            ["for i in range(n):", "    self.scale(x, y, 1, 2)"],
            "a loop the build cannot count",
        ),
    ],
)
def test_a_line_the_method_cannot_hold_is_refused_at_the_make(tmp_path, body, why):
    with pytest.raises(ProgramError, match=r"Bad.steps, line \d+: .*" + why):
        made(chain_class(tmp_path, body), GIVEN)


def test_more_calls_than_a_sequence_numbers_records_make_one_chain(tmp_path):
    body = ["for i in range(LIMIT):", "    self.scale(x, y, 1, 2)"]
    program = made(chain_class(tmp_path, body), GIVEN)
    assert len(program.steps(GIVEN["x"], GIVEN["y"], 1)) == 300  # LIMIT, 2 numbers
