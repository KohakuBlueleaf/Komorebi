"""The map/collect example's tail (examples/01_mapcollect/mapcollect.py): when the ring
gives a page twice, the last jobs that fit one more tensor write into it alone and are
the jobs checked; the feeder harness's copy (mapcollect_fed.py) does the same."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mapcollect = load("mapcollect_example", "examples/01_mapcollect/mapcollect.py")
fed = load("mapcollect_fed", "experiments/checks/feeder_runs/mapcollect_fed.py")
FORMS = [mapcollect, fed]


def writers(groups, first_pages, pages):
    """The number of jobs that write each page."""
    count = np.zeros(pages, np.int64)
    for (_, size), first in zip(groups, first_pages):
        count[first : first + size] += 1
    return count


def tail_of(form, counts, pages, tensors=1):
    groups = [(0, count) for count in counts]
    ring = form.place_outputs(groups, pages, tensors)
    return groups, *form.tail_outputs(groups, ring, pages, tensors)


@pytest.mark.parametrize("form", FORMS)
def test_the_tail_of_small_patterns(form):
    groups, first_pages, tail = tail_of(form, [2, 2, 2], 8)
    assert tail is None and first_pages == [0, 2, 4]  # no page twice: as before
    assert form.checked_jobs(groups, tail) == [0, 1, 2]
    groups, first_pages, tail = tail_of(form, [2, 2, 2, 2, 2], 4)
    assert tail == 3 and first_pages == [0, 2, 0, 4, 6]  # jobs 3, 4: tensor 1
    assert form.checked_jobs(groups, tail) == [3, 4]
    groups, first_pages, tail = tail_of(form, [3, 2, 1, 2], 4)
    assert tail == 2 and first_pages == [0, 0, 4, 5]  # 1 + 2 fit, 2 + 1 + 2 do not
    groups, first_pages, tail = tail_of(form, [4, 4, 4], 4, tensors=2)
    assert tail == 2 and first_pages == [0, 4, 8]  # the tail's tensor is the third


@pytest.mark.parametrize("form", FORMS)
def test_the_card_lists_line_has_no_tail(form):
    groups, first_pages, tail = tail_of(form, [8] * 20, 2048)  # --jobs 20 --count 8
    assert tail is None and form.checked_jobs(groups, tail) == list(range(20))


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize(
    "seed,pages,tensors", [(1, 2048, 1), (2, 2048, 3), (3, 512, 1)]
)
def test_a_page_of_the_tails_tensor_has_one_writer(form, seed, pages, tensors):
    rng = np.random.default_rng(seed)
    counts = [int(count) for count in rng.integers(1, 121, 1000)]
    groups, first_pages, tail = tail_of(form, counts, pages, tensors)
    assert tail is not None
    ring = pages * tensors
    assert sum(counts[tail:]) <= pages < sum(counts[tail - 1 :])  # the longest suffix
    assert first_pages[:tail] == form.place_outputs(groups, pages, tensors)[:tail]
    count = writers(groups, first_pages, ring + pages)
    assert (count[ring:] <= 1).all() and count[ring:].sum() == sum(counts[tail:])
    assert all(first_pages[job] >= ring for job in form.checked_jobs(groups, tail))
    assert all(first_pages[job] + counts[job] <= ring for job in range(tail))
