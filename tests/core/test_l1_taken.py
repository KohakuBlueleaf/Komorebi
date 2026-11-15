"""The L1 a session already holds in a load's L1 budget table: the L1 top down to the
lowest L1 buffer, less what the owner keeps; a finished run's L1 given back."""

from komorebi import session
from komorebi.errors import L1BudgetError


class Module:
    """The native module's two answers: the lowest L1 buffer, the L1 layout."""

    def __init__(self, lowest):
        self.lowest = lowest

    def lowest_l1(self, device):
        return self.lowest

    def kernel_buffer(self, device):
        return (70656, 1461120, 1572864)


def opened(lowest, held=()):
    made = session.Session()
    made.device, made._module = object(), Module(lowest)
    made.l1_held = lambda: list(held)
    return made


def test_nothing_held_has_no_row():
    assert opened(0).l1_taken() == []


def test_the_top_down_to_the_lowest_buffer_less_the_owners():
    made = opened(1423872)
    assert made.l1_taken() == [(session.TAKEN_ROW, 1572864 - 1423872)]
    made = opened(1423872, [("link area", 100000)])
    assert made.l1_taken() == [(session.TAKEN_ROW, 1572864 - 1423872 - 100000)]
    assert opened(1572864 - 1000, [("link area", 1000)]).l1_taken() == []


def test_a_load_counts_what_was_held_before_its_own():
    made = opened(1500000)
    made._taken = [(session.TAKEN_ROW, 5)]  # as _try_load found it
    made._module.lowest = 1000000  # the load's own transfer buffer since
    assert made.l1_taken() == [(session.TAKEN_ROW, 5)]


def test_the_refusal_says_what_to_free():
    error = L1BudgetError("Linear", "  table", held=82000, taken=82000)
    assert "82,000 B of it the session held before this load" in str(error)
    assert "run.stop()" in str(error) and error.taken == 82000


class Tensor:
    def __init__(self, in_dram):
        self.in_dram = in_dram


def test_a_finished_run_gives_its_l1_back():
    backend = session._Backend.__new__(session._Backend)
    dram, l1 = Tensor(True), Tensor(False)
    backend.io, backend.control, backend.trace_tensor = [dram, l1], l1, l1
    backend.release_l1()
    assert backend.io == [dram] and backend.control is backend.trace_tensor is None
