"""What a backend reads of its session: the values a load fixes before the backend is
made, and the session's calls it makes. A backend holds this, not the session."""

from collections.abc import Callable
from typing import Any, NamedTuple


class Card(NamedTuple):
    """The session's side of a backend (Session._card), made as the backend is."""

    log: Callable  # the session's log
    module: Any  # the native module
    tree: Any  # the runtime tree the kernels build against
    device: Any  # the open device
    dram_banks: int  # its DRAM banks (Session.dram_banks)
    l1_bytes: int  # a Tensix core's L1 of the open device (Session.l1_bytes)
    island: Any  # (graph, i) of the load: Session.load's `island`, or None
    link_area: int  # its link area's L1 address, or 0
    link_plan: Any  # Session.load's `link`
    link_plans: Any  # per island LinkPlan of a load over a split card
    packet: int  # an island session's link packet bytes (0: none)
    tensor_table: int  # its tensor table's L1 address (load's `tensors`)
    page_bounds: int  # the most a load's list of page bounds holds (0: none)
    slow_code: bool  # slots may end above L1_FAST_CODE
    grid_timeout: float  # the wait for the grid after a stop
    transfer_chunk: Any  # a larger transfer chunk asked for, where it fits
    channel: bool  # the host channel is mapped (a prebuild's: the card would map it)
    tensor: Callable  # Session.tensor
    empty: Callable  # Session.empty
    write: Callable  # Session.write
    l1: Callable  # Session.l1: (taken, free)
    l1_rows: Callable  # Session.l1_rows
    l1_held: Callable  # Session.l1_held
    l1_taken: Callable  # Session.l1_taken
    nocs: Callable  # Session.nocs
    page_bounds_list: Callable  # Session.page_bounds_list
    largest_page: Callable  # bytes of the largest DRAM page made so far
    hand_nocs: Callable  # Session._hand_nocs
    resize: Callable  # Session._resize: raises kernel_buffer.Resize
    mark_stuck: Callable  # the grid did not return: the runtime closes it at exit
