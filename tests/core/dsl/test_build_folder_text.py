"""A class's build folder is named by its generated text, not by the make's inputs:
equal text is one folder (and one path in each part's cache key), text that differs
never is, and a folder is not trusted by its name; a file a build writes is whole or
absent for a build beside it (no card, no compiler)."""

import contextlib
import threading
from types import SimpleNamespace

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl import first_use
from komorebi.dsl.files import TEXT_KEY_CHARS, put, text_key
from komorebi.dsl.program import sources_key
from komorebi.dsl.thread import class_thread
from komorebi.program import overlay


class Pages:
    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 2048, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tensor():
    return Tensor(Pages(2048), (2048, 32), "bfloat16", "tile")


def class_one():
    """A class named Same whose handler drains one page."""

    class Same(km.Program):
        @km.handler(overlay=True)
        def nothing(self, source: km.Tensor):
            km.Stream(source.pages(0, 1)).drain()

    return Same


def class_two():
    """Another class of the same name, its handler draining two pages."""

    class Same(km.Program):
        @km.handler(overlay=True)
        def nothing(self, source: km.Tensor):
            km.Stream(source.pages(0, 2)).drain()

    return Same


def part_key(tmp_path, program):
    """built_key of the class's reader part against a recipe of one fixed ELF."""
    elf = tmp_path / "kernel.elf"
    elf.write_bytes(b"elf")
    recipe = overlay.Recipe("brisc", elf, ("g++",), "-Os", (), (), (), "-Os", ())
    return overlay.built_key(recipe, program.overlays["nothing"], "reader")


def made(cls, **given):
    return cls(log=lambda line: None, l1=1 << 20, source=tensor(), **given)


def test_equal_text_under_two_makes_is_one_folder(tmp_path):
    one = made(class_one())
    two = made(class_one(), depth=3)  # another make key, the same text
    assert one.built.make != two.built.make
    assert one.built.digest == two.built.digest
    assert one.built.folder == two.built.folder
    assert one.built.folder.name == f"same-{one.built.digest}"
    # what each part's cache key reads of the class: the same file paths
    assert one.overlays["nothing"].sources == two.overlays["nothing"].sources
    assert part_key(tmp_path, one) == part_key(tmp_path, two)


def test_two_classes_of_one_name_with_different_text_have_two_folders(tmp_path):
    one, two = made(class_one()), made(class_two())
    assert one.built.make == two.built.make  # sources_key does not see the text
    assert one.built.folder != two.built.folder
    assert part_key(tmp_path, one) != part_key(tmp_path, two)  # nor is a part's
    again = made(class_one())
    assert again.built.folder == one.built.folder
    for program in (one, two):  # each folder still holds its own class's text
        assert (
            text_key(
                {
                    name: (program.built.folder / name).read_text()
                    for name in program.built.files
                }
            )
            == program.built.digest
        )


def test_a_folder_is_not_trusted_by_its_name():
    program = made(class_one())
    path = program.built.folder / "movement.cpp"
    wanted = path.read_text()
    path.write_text("// stale\n")
    again = made(class_one())
    assert again.built.folder == program.built.folder
    assert path.read_text() == wanted  # written again: not equal to the text
    stamp = path.stat().st_mtime_ns
    made(class_one())
    assert path.stat().st_mtime_ns == stamp  # equal: left as it is


def test_the_make_key_is_the_inputs_and_the_folder_the_text():
    small, wide = tensor(), Tensor(Pages(2048), (4096, 32), "bfloat16", "tile")
    base = ({}, {"source": small}, None, 4, None, None)
    other = ({}, {"source": wide}, None, 4, None, None)
    assert sources_key(*base) != sources_key(*other)
    assert hasattr(first_use, "folder_key")


def test_a_file_a_build_writes_is_never_read_half_written(tmp_path):
    """Two texts of one name written in turn while a reader reads the file: every read
    is a whole text (a write that truncates first shows an empty or partial one)."""
    path = tmp_path / "threaded_thread.h"
    texts = ["// one\n" * 40000, "// two!\n" * 30000]
    seen, stop = [], threading.Event()

    def read():
        while not stop.is_set():
            with contextlib.suppress(FileNotFoundError):
                seen.append(path.read_text())

    reader = threading.Thread(target=read)
    reader.start()
    try:
        for turn in range(150):
            put(path, texts[turn % 2])
    finally:
        stop.set()
        reader.join()
    assert seen and all(text in texts for text in seen)
    assert [file.name for file in tmp_path.iterdir()] == [path.name]


def test_two_classes_of_one_name_leave_two_whole_headers(tmp_path):
    """The thread header's folder is named by the header's text: another text is
    another folder, an equal text the same one, and each holds its own whole text."""
    program = SimpleNamespace(built=SimpleNamespace(folder=tmp_path))
    parts = SimpleNamespace(compiled=SimpleNamespace(name="Threaded"))
    one, two = "// handlers add, sum\n" * 500, "// records, chain steps\n" * 500
    first = class_thread.header_written(program, parts, one)
    second = class_thread.header_written(program, parts, two)
    assert first != second and first.parent == second.parent == tmp_path
    assert len(first.name.removeprefix("thread-")) == TEXT_KEY_CHARS
    assert (first / "threaded_thread.h").read_text() == one
    assert (second / "threaded_thread.h").read_text() == two
    assert class_thread.header_written(program, parts, one) == first
    assert (first / "threaded_thread.h").read_text() == one
