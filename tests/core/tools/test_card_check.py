"""The card check's shared open on the mock: a child that dies before its BUILT lines
is named by its return code and stderr, and each attempt keeps its own log; the
tensix lines' card and dispatcher; the rule that judges a result's last JSON line."""

import json
import subprocess

from komorebi.tools import card_check


def ended(code, stderr=""):
    return subprocess.CompletedProcess([], code, stdout="", stderr=stderr)


def test_a_negative_code_is_named_by_its_signal_with_the_stderr_tail():
    said = card_check.child_ended(ended(-11, "a\n\nb\nc\nd\ne\nf\n"))
    assert said == "killed by SIGSEGV; its stderr ended: b | c | d | e | f"


def test_an_exit_code_is_given_as_it_is_and_no_stderr_is_said():
    assert card_check.child_ended(ended(1)) == "exit 1; no stderr"


def test_a_refusal_printed_to_stdout_is_shown_when_there_is_no_stderr():
    out = "logs: x\nREFUSED: not prebuilt (run --prebuild): 1 check(s), first:\n  a b\n"
    done = subprocess.CompletedProcess([], 1, stdout=out, stderr="")
    assert card_check.child_ended(done) == (
        "exit 1; no stderr; its stdout ended: "
        "REFUSED: not prebuilt (run --prebuild): 1 check(s), first: | a b"
    )


def shared(monkeypatch, tmp_path, results):
    calls = iter(results)
    monkeypatch.setattr(card_check.subprocess, "run", lambda *a, **k: next(calls))
    monkeypatch.setattr(card_check.native, "cache_directory", lambda: tmp_path)
    monkeypatch.setattr(card_check.mock, "descriptor", lambda cards: "mock.yaml")
    return card_check.shared_pass("1", tmp_path, tmp_path, tmp_path / "list")


def test_a_child_dead_before_its_built_lines_fails_saying_how(
    monkeypatch, tmp_path, capsys
):
    ok, _ = shared(monkeypatch, tmp_path, [ended(-11, "opening\n")])
    out = capsys.readouterr().out
    assert not ok
    assert "0 of 0 checks compiled; killed by SIGSEGV" in out
    assert "its stderr ended: opening" in out
    assert (tmp_path / "mock-pass-1-1.log").read_text() == "opening\n"


def test_each_attempt_keeps_its_own_log(monkeypatch, tmp_path, capsys):
    first = subprocess.CompletedProcess([], 0, stdout="BUILT 2 a\n", stderr="")
    second = subprocess.CompletedProcess([], 0, stdout="BUILT 0 a\n", stderr="")
    ok, first_count = shared(monkeypatch, tmp_path, [first, second])
    assert ok and first_count == 2
    assert (tmp_path / "mock-pass-1-1.log").read_text() == "BUILT 2 a\n"
    assert (tmp_path / "mock-pass-1-2.log").read_text() == "BUILT 0 a\n"
    assert "killed" not in capsys.readouterr().out


def test_the_tensix_lines_open_card_1_with_the_tensix_dispatcher(monkeypatch):
    monkeypatch.delenv("KMR_DISPATCHER", raising=False)
    tensix = card_check.read_list(card_check.LIST, "tensix")
    ones = {check.line[2:] for check in card_check.read_list(card_check.LIST, "1")}
    assert tensix and all(check.line[len("tensix ") :] in ones for check in tensix)
    assert card_check.SHAPES["tensix"] == (1,)
    assert tensix[0].name.startswith("tensix_")  # its logs beside card 0's own
    assert card_check.shape_environment("tensix")["KMR_DISPATCHER"] == "tensix"
    assert "KMR_DISPATCHER" not in card_check.shape_environment("1")


def judged(out, code=0):
    return card_check.verdict(code, None, "log line\n" + json.dumps(out) + "\n")


def test_booleans_under_ok_and_exact_keys_are_judged_at_any_depth():
    assert (
        judged({"ok": True, "runs": [{"exact": True}], "a_exact": {"b": True}}) is None
    )
    assert judged({"ok": True, "runs": [{"exact": False}]}) == "not true: runs.0.exact"
    assert judged({"ok": False, "x": 1}) == "not true: ok"


def test_a_number_a_string_or_null_under_such_a_key_is_not_a_verdict():
    assert judged({"summary": {"exact": 3}}) == "not a boolean: summary.exact"
    assert judged({"lanes_exact": 0}) == "not a boolean: lanes_exact"
    assert judged({"ok": "yes"}) == "not a boolean: ok"
    assert judged({"ok": None}) == "not a boolean: ok"
    assert judged({"exact": [True, 1]}) == "not a boolean: exact.1"


def test_keys_that_are_not_verdicts_and_a_result_without_one_are_not_read():
    assert judged({"count": 3, "rows": [{"wrong": 2}]}) is None
    assert judged({"ok": True}, code=1) == "exit 1"
