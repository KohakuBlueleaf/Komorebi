"""build.py: reading build.ninja and assembling the compile, link and image commands."""

import pytest
from fake_tree import build_ninja, make_tree

from komorebi import _env, build, cli


def test_ninja_block_reads_the_variables_of_the_first_match_up_to_the_next_statement(
    tmp_path,
):
    (tmp_path / "build.ninja").write_text(
        "build one.o: cc a.c\n  FLAGS = -O1\n  EMPTY = \n\n"
        "build one.o2: cc b.c\n  FLAGS = -O9\n"
    )
    assert build.ninja_block(tmp_path, "build one.o:") == {"FLAGS": "-O1", "EMPTY": ""}
    assert build.ninja_block(tmp_path, "build one.o2") == {"FLAGS": "-O9"}
    (tmp_path / "build.ninja").write_text(
        "build a: cc\n  X = 1\nbuild b: cc\n  Y = 2\n"
    )
    assert build.ninja_block(tmp_path, "build a") == {"X": "1"}


def test_ninja_block_without_a_build_ninja_says_the_tree_is_not_built(tmp_path):
    with pytest.raises(build.BuildError, match=r"build\.ninja not found.*not built"):
        build.ninja_block(tmp_path, "build one")


def test_ninja_block_without_a_match_is_an_error(tmp_path):
    (tmp_path / "build.ninja").write_text(build_ninja(("host",)))
    with pytest.raises(build.BuildError, match="no build statement"):
        build.ninja_block(tmp_path, "build nothing")


def test_cxx_commands_follow_the_borrowed_flags(tmp_path):
    kind = "host"
    tree = make_tree(tmp_path / "tree")
    out = tmp_path / "out" / "prog"
    compile_cmd, link_cmd = build.cxx_commands(
        kind, tree, out, "main.cpp", includes=["/mine"], defines=["N=3"]
    )
    assert compile_cmd[0] == "clang++-20"
    assert compile_cmd[1:5] == ["-DFOO=1", "-DBAR=two words", "-DN=3", "-I/mine"]
    assert compile_cmd[5:9] == ["-I/a/include", "-isystem", "/b/include", "-O2"]
    assert compile_cmd[-4:] == ["-c", "main.cpp", "-o", str(out.with_suffix(".o"))]
    assert not any("KERNEL_DIR" in tok for tok in compile_cmd)

    flags = build.TEMPLATES[kind][2]
    build_directory = tree / "build"
    assert link_cmd == [
        "clang++-20",
        *flags,
        str(out.with_suffix(".o")),
        "-o",
        str(out),
        str(build_directory / "lib/libone.so"),
        "-lpthread",
        "/usr/lib/libz.so",
        str(build_directory / "lib/libtwo.a"),
    ]


def test_cxx_commands_kinds_differ_in_link_flags(tmp_path):
    tree = make_tree(tmp_path / "tree")
    host = build.cxx_commands("host", tree, tmp_path / "a", "m.cpp")[1]
    module = build.cxx_commands("pymodule", tree, tmp_path / "a", "m.cpp")[1]
    assert "-shared" in module and "-shared" not in host


def ninja_without(tmp_path, *names):
    """A tree whose host statements lack the variables `names`."""
    tree = make_tree(tmp_path / "tree")
    path = tree / "build" / "build.ninja"
    lines = path.read_text().splitlines()
    kept = [
        x
        for x in lines
        if not x.strip().startswith(tuple(f"{name} =" for name in names))
    ]
    path.write_text("\n".join(kept) + "\n")
    return tree


def test_cxx_commands_name_the_statement_that_lacks_flags_or_libraries(tmp_path):
    for name, prefix in [
        ("FLAGS", "build tt_metal"),
        ("LINK_LIBRARIES", "build test/tt_metal"),
    ]:
        tree = ninja_without(tmp_path / name, name)
        with pytest.raises(build.BuildError, match=f"{prefix}.*has no {name}"):
            build.cxx_commands("host", tree, tmp_path / "a", "m.cpp")


def test_image_commands_compile_link_and_convert(tmp_path):
    tree = make_tree(tmp_path / "tree")
    out = tmp_path / "out" / "image.bin"
    commands = build.image_commands(tree, out, ["x/a.c", "y/b.c"])
    gcc = str(_env.sfpi_bin(tree) / "riscv-tt-elf-gcc")
    assert len(commands) == 4
    assert commands[0][0] == commands[1][0] == commands[2][0] == gcc
    assert commands[0][-4:] == ["-c", "x/a.c", "-o", str(out.parent / "a.x280.o")]
    assert commands[1][-4:] == ["-c", "y/b.c", "-o", str(out.parent / "b.x280.o")]
    assert "-Werror" in commands[0] and "-mcpu=sifive-x280" in commands[0]
    assert f"-I{_env.x280_source_directory(tree) / 'include'}" in commands[0]

    link = commands[2]
    elf = str(out.with_suffix(".elf"))
    assert link[link.index("-o") + 1] == elf
    assert link[link.index("-T") + 1] == str(_env.x280_linker_script(tree))
    runtime = [str(path) for path in _env.x280_objects(tree)]
    objects = [str(out.parent / name) for name in ("a.x280.o", "b.x280.o")]
    assert link[-4:] == [*runtime, *objects]
    assert commands[3][1:] == ["-O", "binary", elf, str(out)]
    assert commands[3][0].endswith("riscv-tt-elf-objcopy")


def test_image_commands_pass_includes_and_defines_to_every_compile(tmp_path):
    tree = make_tree(tmp_path / "tree")
    out = tmp_path / "image.bin"
    commands = build.image_commands(
        tree, out, ["a.c", "b.c"], includes=["inc/one", "inc/two"], defines=["N=4", "V"]
    )
    runtime = f"-I{_env.x280_source_directory(tree) / 'include'}"
    for compile_cmd in commands[:2]:
        flags = compile_cmd[: compile_cmd.index("-c")]
        assert "-DN=4" in flags and "-DV" in flags
        mine = [flags.index("-Iinc/one"), flags.index("-Iinc/two")]
        assert mine == sorted(mine) and mine[-1] < flags.index(runtime)
    link, objcopy = commands[2], commands[3]
    assert not any(argument.startswith(("-I", "-D")) for argument in link + objcopy)
    assert commands[2:] == build.image_commands(tree, out, ["a.c", "b.c"])[2:]


def test_a_failing_step_is_a_build_error(tmp_path):
    with pytest.raises(build.BuildError):
        build.run([["false"]])
    with pytest.raises(build.BuildError):
        build.run([[str(tmp_path / "no-such-compiler")]])


def test_build_image_creates_the_output_directory_and_runs_in_order(
    tmp_path, monkeypatch
):
    tree = make_tree(tmp_path / "tree")
    ran = []
    monkeypatch.setattr(
        build.subprocess, "run", lambda command, check: ran.append(command)
    )
    out = tmp_path / "deep" / "dir" / "image.bin"
    build.build_image(tree, out, ["a.c"])
    assert out.parent.is_dir()
    assert ran == build.image_commands(tree, out, ["a.c"])


def test_kmr_build_without_a_tree_is_a_usage_error(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("TT_METAL_HOME", raising=False)
    assert cli.main(["build", "image", str(tmp_path / "i.bin"), "a.c"]) == 2
    assert "TT_METAL_HOME" in capsys.readouterr().err


def test_kmr_build_in_a_tree_that_is_not_built_prints_one_line(tmp_path, capsys):
    tree = make_tree(tmp_path / "tree", ninja=False)
    for kind in ["host", "pymodule"]:
        args = ["build", kind, str(tmp_path / "prog"), "m.cpp", "--tree", str(tree)]
        assert cli.main(args) == 2
        error_text = capsys.readouterr().err
        assert error_text.startswith("kmr build:") and "not built" in error_text
        assert error_text.count("\n") == 1
