"""One exponent family a kernel: a program whose buffers mix the A and the B family
is refused when it is made, naming one buffer of each; float32 and the integers go
with either; an A buffer with an fp32 Dst is refused."""

import pytest

import komorebi as km
from komorebi.errors import ProgramError
from komorebi.program.description import Buffer


def sources(folder):
    folder.mkdir(parents=True)
    for kernel in ("reader", "compute", "writer"):
        (folder / f"{kernel}.cpp").write_text("void kernel_main() {}\n")
    return folder


def made(tmp_path, buffers, compute=None):
    return km.Program(
        "mixed",
        kernels=sources(tmp_path / "k"),
        buffers=buffers,
        compute=compute or {},
    )


def test_a_and_b_buffers_in_one_program_are_refused_naming_both(tmp_path):
    buffers = [
        Buffer(0, "bfloat16", 2048, 2),
        Buffer(1, "float32", 4096, 2),
        Buffer(2, "bfp8", 1088, 2),
    ]
    with pytest.raises(ProgramError) as refused:
        made(tmp_path, buffers)
    said = str(refused.value)
    assert "buffer 2 (bfp8) is of the A family" in said
    assert "buffer 0 (bfloat16) of the B family" in said


def test_float32_and_integers_go_with_either_family(tmp_path):
    made(tmp_path, [Buffer(0, "float16", 2048, 2), Buffer(1, "float32", 4096, 2)])
    made(tmp_path / "b", [Buffer(0, "bfp4_b", 576, 2), Buffer(1, "uint32", 4096, 1)])
    made(tmp_path / "c", [Buffer(0, "bfloat8_b", 1088, 2), Buffer(1, "bfp2_b", 320, 1)])


def test_an_a_buffer_with_fp32_in_dst_is_refused(tmp_path):
    with pytest.raises(ProgramError, match=r"buffer 3 \(bfp4\).*16-bit Dst"):
        made(
            tmp_path,
            [Buffer(3, "bfp4", 576, 2)],
            compute={"fp32_dest_acc_en": True},
        )
