"""A load of programs placed by groups (a dict, a list, a class) gives its link cores'
kernels the put queue's define with the ranks', so the queue the ranks put into is
served (no card: a recording launch)."""

from types import SimpleNamespace

from komorebi import session as session_module


class Launch:
    def __init__(self):
        self.kernels = []

    def cb(self, *args):
        pass

    def kernel(self, path, cores, processor, compile_args, **options):
        self.kernels.append((path, dict(options.get("defines", []))))


def link_defines(queue):
    backend = SimpleNamespace(trace_cap=0, cb_indices=set())
    backend._runtime_cbs = lambda: session_module._Backend._runtime_cbs(backend)
    program = SimpleNamespace(defines={"KM_PAGE_LINKS": 1, "KM_SPLIT": 1})
    launch = Launch()
    args = [[[]] * 120] * 3
    session_module._Backend._link_cores(
        backend, launch, program, args, 118, 2, (10, 12), queue
    )
    return [defines for path, defines in launch.kernels if "link_" in path]


def test_the_link_kernels_take_the_put_queue_s_define():
    every = link_defines([("KM_LINK_QUEUE", "1")])
    assert len(every) == 3
    assert all(d.get("KM_LINK_QUEUE") == "1" for d in every)
    assert all(d["KM_LINK_SERVICES"] == "1" and "KM_SPLIT" not in d for d in every)


def test_without_a_put_queue_the_link_kernels_name_none():
    assert all("KM_LINK_QUEUE" not in d for d in link_defines(()))
