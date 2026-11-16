"""A load of programs placed by groups (a dict, a list, a class) gives its link cores'
kernels the put queue's define and every program's link defines, and a load that would
not is refused by name (no card: a recording launch)."""

from types import SimpleNamespace

import pytest

from komorebi import session as session_module
from komorebi.errors import ProgramError


class Launch:
    def __init__(self):
        self.kernels = []

    def cb(self, *args):
        pass

    def kernel(self, path, cores, processor, compile_args, **options):
        self.kernels.append((path, dict(options.get("defines", []))))


def program(name, **defines):
    return SimpleNamespace(name=name, defines={"KM_SPLIT": 1, **defines})


def link_defines(queue, placed=None):
    backend = SimpleNamespace(trace_cap=0, cb_indices=set())
    backend._runtime_cbs = lambda: session_module._Backend._runtime_cbs(backend)
    placed = placed or [program("sum", KM_PAGE_LINKS=1)]
    launch = Launch()
    args = [[[]] * 120] * 3
    session_module._Backend._link_cores(
        backend, launch, placed, args, 118, 2, (10, 12), queue
    )
    return [defines for path, defines in launch.kernels if "link_" in path]


def test_the_link_kernels_take_the_put_queue_s_define():
    every = link_defines([("KM_LINK_QUEUE", "1")])
    assert len(every) == 3
    assert all(d.get("KM_LINK_QUEUE") == "1" for d in every)
    assert all(d["KM_LINK_SERVICES"] == "1" and "KM_SPLIT" not in d for d in every)


def test_a_load_whose_ranks_put_across_without_the_queue_is_refused_by_name():
    with pytest.raises(ProgramError, match="sum: .* without KM_LINK_QUEUE"):
        link_defines(())


def test_the_link_kernels_take_every_placed_program_s_link_defines():
    placed = [program("sum", KM_PAGE_LINKS=1), program("scale", KM_TENSOR_TABLE=1)]
    every = link_defines([("KM_LINK_QUEUE", "1")], placed)
    assert all(d.get("KM_TENSOR_TABLE") == "1" for d in every)


def test_the_check_names_a_link_define_the_link_kernels_lack():
    placed = [program("scale", KM_PAGE_LINKS=1, KM_ASKS_BY_HOST=1)]
    defines = [("KM_PAGE_LINKS", "1"), ("KM_LINK_QUEUE", "1")]
    with pytest.raises(ProgramError, match="scale: .* without KM_ASKS_BY_HOST"):
        session_module.check_link_defines("scale", defines, placed)
