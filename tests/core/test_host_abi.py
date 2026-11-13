"""komorebi.abi: the host's reading of the shared C headers, strict about what it
cannot read."""

import pytest

from komorebi import abi

H = abi.headers()


def test_record_and_event_fields_follow_the_abi_header():
    assert H.record_fields == [
        "seq", "hdr", "team", "a0", "land1", "a1", "a2", "a3",
        "land2", "a4", "a5", "a6", "land3", "a7", "a8", "a9",
    ]  # fmt: skip
    assert H.event_fields == ["index", "req", "status", "ticks"]


def test_start_up_arguments_are_in_header_order():
    assert H.arg_fields[:6] == ["ctl", "me", "cores", "disp_xy", "ack_base", "mailbox"]
    assert len(H.arg_fields) == H["kArgs"] == 19
    assert {"row", "workers"} <= set(H.arg_fields)
    values = {name: i for i, name in enumerate(H.arg_fields)}
    assert abi.start_args(values) == list(range(19))
    with pytest.raises(ValueError, match="missing"):
        abi.start_args({"ctl": 1})
    with pytest.raises(ValueError, match="unknown"):
        abi.start_args(values | {"extra": 1})


def test_constants_of_all_three_headers():
    assert H["KM_RING_SLOTS"] == 16 and H["KM_CTL_BYTES"] == 0x1580
    assert H["KM_READY"] == 0xFFFFFFFF and H["KM_SEG_EXIT"] == 255
    assert H["KMD_LIM_MAILBOX"] == 0x100 and H["KMD_HOST_EVENTS"] == 2
    assert H["KMD_R_WORDS"] == 32 and H["KMD_R_LAST_HI"] == 23
    assert (
        H["kOutReady"] == 1
        and H["kOutSlots"] == (H["KM_CTL_CORES"] - H["KM_CTL_OUT"]) // 16
    )
    assert "KM_TEAM_ALL" not in H.constants  # a function-like macro


def test_record_encoding_puts_fields_where_the_header_says():
    words = abi.record(7, 3, 9, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    named = dict(zip(H.record_fields, words))
    assert named["hdr"] == 7 and named["team"] == 3 | 9 << 16
    assert [named[f"a{i}"] for i in range(10)] == list(range(10, 20))
    assert named["seq"] == named["land1"] == named["land2"] == named["land3"] == 0
    assert abi.record(1, 0, 1, [0xFFFFFFFF])[3] == 0xFFFFFFFF
    with pytest.raises(ValueError, match="11 arguments"):
        abi.record(1, 0, 1, range(11))


def test_column_groups_cover_every_core_once():
    groups = abi.column_groups((12, 10), 120)
    assert sorted(groups) == list(range(120))
    leaders = {c for c, g in groups.items() if g[0] == c}
    assert leaders == set(range(12))
    for leader in leaders:
        members = [c for c, g in groups.items() if g[0] == leader and c != leader]
        assert groups[leader][2] == len(members) == 9
        assert sorted(groups[c][1] for c in members) == list(range(9))
        assert {groups[c][3] for c in [leader, *members]} == {leader}


def test_limit_units():
    assert abi.limit_units(1.0, 1024) == 1_000_000
    assert abi.limit_units(0.0, 1350) == 1
    assert abi.limit_units(1e9, 1350) == 0xFFFFFFFF


def write(tmp_path, text, name="h.h"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_expressions_suffixes_and_continued_lines(tmp_path):
    path = write(
        tmp_path,
        "#define KM_A 0x10U\n#define KM_B \\\n  (KM_A + 4ul) /* c */\n"
        "#define KM_F(x) ((x) + 1)\n"
        "constexpr uint32_t kC = KM_B << 2;  // c\n"
        "constexpr uint32_t kD = sizeof(S) / 4;\n"
        "enum { KMD_R_X = 3, KMD_R_Y, KMD_R_Z = KMD_R_X + 7, };\n",
    )
    found = abi.constants(path, sizes={"S": 12})
    assert found == {
        "KM_A": 16, "KM_B": 20, "kC": 80, "kD": 3, "KMD_R_X": 3, "KMD_R_Y": 4,
        "KMD_R_Z": 10,
    }  # fmt: skip


@pytest.mark.parametrize(
    "line",
    [
        "#define KM_A some_function()",
        "#define KM_A (1 ? 2 : 3)",
        "#define KM_A KM_UNKNOWN + 1",
        "#define KM_A 1.5f",
        "constexpr uint32_t kA = sizeof(Unknown);",
        "#define KM_A",
    ],
)
def test_a_definition_that_cannot_be_read_is_an_error(tmp_path, line):
    with pytest.raises(abi.HeaderError, match="cannot read"):
        abi.constants(write(tmp_path, line + "\n"))


def test_a_name_defined_twice_with_different_values_is_an_error(tmp_path):
    one = write(tmp_path, "#define KM_A 1u\n", "one.h")
    same = write(tmp_path, "#define KM_A 0x1\n", "same.h")
    other = write(tmp_path, "#define KM_A 2u\n", "other.h")
    assert abi.constants(one, same) == {"KM_A": 1}
    with pytest.raises(abi.HeaderError, match="disagrees"):
        abi.constants(one, other)


def test_struct_fields_are_anchored_to_the_named_struct(tmp_path):
    path = write(
        tmp_path,
        "typedef struct {\n uint32_t a, b; /* x */\n} first_t;\n"
        "typedef struct {\n uint32_t c;  // y\n uint32_t d;\n} second_t;\n"
        "struct Named {\n uint32_t e;\n};\n",
    )
    assert abi.struct_fields(path, "first_t") == ["a", "b"]
    assert abi.struct_fields(path, "second_t") == ["c", "d"]
    assert abi.struct_fields(path, "Named") == ["e"]
    with pytest.raises(abi.HeaderError, match="no struct"):
        abi.struct_fields(path, "third_t")


@pytest.mark.parametrize(
    "member", ["uint32_t x[2];", "uint64_t y;", "uint16_t z;", "const uint32_t* p;"]
)
def test_a_struct_member_that_cannot_be_read_is_an_error(tmp_path, member):
    path = write(tmp_path, f"struct Args {{\n uint32_t a;\n {member}\n}};\n")
    with pytest.raises(abi.HeaderError, match="cannot read member"):
        abi.struct_fields(path, "Args")


def read(tmp_path, text):
    path = tmp_path / "h.h"
    path.write_text(text)
    return abi.constants(path)


@pytest.mark.parametrize(
    "text",
    [
        "enum Mode { KM_A = 1 };",
        "constexpr int kA = 1;",
        "constexpr uint32_t kA{1};",
        "#define KM_A (0u - 1u)",
        "#define KM_A (1 / 0)",
        "#define KM_A 0b101",
        "#define KM_A 0x1FFFFFFFF",
        "#if defined(SOME_BUILD_FLAG)\n#define KM_A 1\n#endif",
        "#if 1\n#define KM_A 1\n#else\n#define KM_A 2\n",
    ],
)
def test_what_the_reader_cannot_evaluate_as_c_would_is_an_error(tmp_path, text):
    with pytest.raises(abi.HeaderError):
        read(tmp_path, text)


def test_comments_and_disabled_blocks_are_removed_before_reading(tmp_path):
    found = read(
        tmp_path,
        "#ifndef GUARD_H\n#define GUARD_H\n"
        'static const char* s = "/*";\n#define KM_A 1\n'
        "/* #define KM_B 2 */\n// #define KM_C 3\n"
        "#if 0\n#define KM_A 99\n#define KM_D 4\n#else\n#define KM_E 5\n#endif\n"
        "#define KM_F 6\n#undef KM_F\n#define KM_F 7\n"
        "#if !defined(SOME_BUILD_FLAG)\nvoid f();\n#endif\n#endif\n",
    )
    assert found == {"KM_A": 1, "KM_E": 5, "KM_F": 7}


def test_a_name_an_earlier_header_defined_decides_a_conditional(tmp_path):
    one = write(tmp_path, "#define KM_A 2u\n", "one.h")
    two = write(
        tmp_path,
        "#ifndef KM_A\n#define KM_X 9u\n#endif\n#ifdef KM_A\n#define KM_B 3u\n#endif\n"
        "#if defined(__cplusplus)\n#define KM_M(c) f(c)\n#else\n#define KM_M(c) g(c)\n"
        "#endif\n",
        "two.h",
    )
    assert abi.constants(one, two) == {"KM_A": 2, "KM_B": 3}
    with pytest.raises(abi.HeaderError, match="cannot decide"):
        abi.constants(two)


def test_a_programs_header_reads_the_package_constants_by_their_km_names(tmp_path):
    header = write(
        tmp_path,
        '#include "komorebi/device/km_lane.hpp"\n'
        "constexpr uint32_t kArgLane = 0;\n"
        "constexpr uint32_t kArgIndex = km::kLaneArgs + 1;\n"
        "#if !defined(COMPILE_FOR_TRISC)\n"
        "inline void f() { if constexpr (true) {} }\n"
        "#endif\n",
        "program.h",
    )
    assert H["kLaneArgs"] == 11 and H["kWherePort"] == 17  # device headers too
    assert abi.constants(header, known=H.constants) == {
        "kArgLane": 0,
        "kArgIndex": 12,
    }
    with pytest.raises(abi.HeaderError, match="km::kLaneArgs"):
        abi.constants(header)


def test_a_struct_must_be_defined_once_with_distinct_members(tmp_path):
    path = tmp_path / "s.h"
    path.write_text("typedef struct { uint32_t a, a; } two_t;")
    with pytest.raises(abi.HeaderError, match="twice"):
        abi.struct_fields(path, "two_t")
    path.write_text("struct S {\n#if X\n uint32_t a;\n#endif\n uint32_t b; };")
    with pytest.raises(abi.HeaderError, match="undecided"):
        abi.struct_fields(path, "S")
    path.write_text("struct S { uint32_t a; };\nstruct S { uint32_t b; };")
    with pytest.raises(abi.HeaderError, match="2 times"):
        abi.struct_fields(path, "S")
