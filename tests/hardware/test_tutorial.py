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
    ],
)
def test_a_tutorial_program_runs_and_prints_its_line(script, line):
    out = run_text(TUTORIAL / script, timeout=180.0)
    assert line in out
