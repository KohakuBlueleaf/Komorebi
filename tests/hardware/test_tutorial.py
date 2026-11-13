"""The tutorial's programs on a card (KMR_CARD=1): each folder once, at its smallest
size. The scripts check their own results; the line each page shows is looked for in
what they print."""

from pathlib import Path

import pytest
from child import run_text

pytestmark = pytest.mark.card
TUTORIAL = Path(__file__).resolve().parents[2] / "tutorial"


@pytest.mark.parametrize(
    ("script", "line"),
    [
        ("01_first_look/add.py", "1000 + 2345 = 3345"),
        ("03_the_card/card_facts.py", "120 Tensix cores: 12 in a row, 10 rows"),
        ("04_hello_core/hello.py", "the command is done"),
        ("05_commands/commands.py", "command 3: done after"),
        ("06_a_result/a_result.py", "the core answered 42"),
        ("07_arguments/arguments.py", "1000 + 2345 = 3345"),
        ("08_handlers/handlers.py", "6 multiply 7 = 42"),
        ("09_many_commands/many_commands.py", "squares: [1, 4, 9, 16, 25, 36, 49, 64]"),
        ("10_when_it_goes_wrong/when_it_goes_wrong.py", "the card stopped the command"),
        ("11_core_group/core_group.py", "group of 12: the command took"),
        ("12_partitions/partitions.py", "together they cover items 0 to 9"),
        ("13_the_leader/the_leader.py", "only rank 3 answers; Python receives 0"),
        ("14_placing_groups/placing_groups.py", "right panel of rows 2 to 4"),
        ("15_tiles_and_tensors/tiles_and_tensors.py", "6 tiles of 4096 bytes"),
        ("16_reading_from_dram/reading_from_dram.py", "tile 3, number 0: 3000"),
        (
            "17_writing_to_dram/writing_to_dram.py",
            "are tiles 0 and 1 of the destination",
        ),
        ("18_three_kernels/three_kernels.py", "all 8 tiles of the destination"),
        ("19_circular_buffers/circular_buffers.py", "buffer pages 8:"),
        ("20_computing_on_a_tile/computing_on_a_tile.py", "is the sum of tile 1"),
        ("21_the_matrix_unit/the_matrix_unit.py", "is the matrix product of tile 1"),
        ("22_a_whole_map/a_whole_map.py", "all three results are the sum"),
        ("23_both_movers/both_movers.py", "both copies hold every tile of the source"),
        ("24_why_cores_talk/why_cores_talk.py", "[0, 1, 4, 9, 16, 25, 0, 0]"),
        ("25_port/port.py", "core 0 holds number 5 of it: 4077"),
        ("26_reduce/reduce.py", "is the sum of the 8 tiles"),
        ("27_broadcast/broadcast.py", "each of the 6 cores holds the sum"),
        ("28_routed_message/routed_message.py", "rank 4: received 4 messages"),
        ("29_forward_and_consume/forward_and_consume.py", "as the replay says"),
        ("30_full_buffers/full_buffers.py", "rank 0 asked 20 times and 10 sends"),
        ("31_streams_and_graphs/streams_and_graphs.py", "the same rounds in Python"),
        ("32_hello_thread/hello_thread.py", "it returned 42 = 20 + 22"),
        ("33_thread_submits/thread_submits.py", "core 0 answered the thread with 42"),
        ("34_four_cores/four_cores.py", "four cores took"),
        ("35_sharing_memory/sharing_memory.py", "the thread went through 1024 numbers"),
        ("36_twelve_cores/twelve_cores.py", "their sum, modulo 2**32"),
        ("37_core_asks_x280/core_asks_x280.py", "it told the X280 145"),
        ("38_measuring/measuring.py", "host clock: first"),
        ("39_finding_a_bug/finding_a_bug.py", "with bug 0: the command is done"),
        ("40_a_whole_program/a_whole_program.py", "both sums are the numpy sum"),
        ("41_your_own_kernel/your_own_kernel.py", "the squares of the 48 tiles"),
    ],
)
def test_a_tutorial_program_runs_and_prints_its_line(script, line):
    out = run_text(TUTORIAL / script, timeout=180.0)
    assert line in out
