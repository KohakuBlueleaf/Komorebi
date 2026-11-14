"""A thread reads another DRAM bank through Tensix cores (KMR_CARD=1): kmx_fetch of
64 KiB of bank 0 into work cluster 1's region on 1 and 4 cores, then kmx_store of it
back to bank 0, every byte checked."""

from pathlib import Path

import pytest
from shared_card import run_json

pytestmark = pytest.mark.card
SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "bank_fetch" / "fetch.py"


def test_a_thread_fetches_another_banks_range_and_stores_it_back():
    out = run_json(SCRIPT, "--bytes", 65536, "--cores", 1, 4, "--store", 65536)
    assert out["ok"] and out["region_exact"] and out["store"]["exact"], out
    assert all(row["exact"] for row in out["cores"].values())
