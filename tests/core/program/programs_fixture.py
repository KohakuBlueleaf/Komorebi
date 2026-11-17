"""What test_programs.py and other tests share: BUFFERS, GRID, KERNEL, PROGRAM, made,
sources."""

import komorebi as km

GRID = (12, 10)


BUFFERS = [
    km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
    km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
]


PROGRAM = "void km_program(km::{kind}& kernel, const km_command_t& command) {{}}\n"
KERNEL = "void kernel_main() {}\n"


def sources(folder, form="program"):
    folder.mkdir(parents=True, exist_ok=True)
    for kernel, kind in (("reader", "Reader"), ("writer", "Writer")):
        text = PROGRAM.format(kind=kind) if form == "program" else KERNEL
        (folder / f"{kernel}.cpp").write_text(text)
    text = PROGRAM.format(kind="Compute") if form == "program" else KERNEL
    (folder / "compute.cpp").write_text(text)
    return folder


def made(tmp_path, name, buffers=BUFFERS, **options):
    return km.Program(
        name,
        kernels=sources(tmp_path / name),
        buffers=list(buffers),
        handlers={"one": 1, "two": 2},
        **options,
    )
