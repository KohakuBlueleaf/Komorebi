"""The open's output redirect keeps the interpreter's fault handler report: a crash
inside it leaves Python's stack on the real stderr, and the handler is as before after
it; a handler a program pointed at a file of its own is left alone."""

import os
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def child(body, handler="1"):
    """Runs `body` in a fresh interpreter with the open's module imported (as
    `open_part`), PYTHONFAULTHANDLER=`handler` (None: unset); its CompletedProcess."""
    code = "import ctypes, faulthandler, sys\n"
    code += "from komorebi.session.parts import open as open_part\n"
    code += textwrap.dedent(body)
    paths = [os.path.join(ROOT, "src"), os.environ.get("PYTHONPATH", "")]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in paths if p))
    env.pop("PYTHONFAULTHANDLER", None)
    if handler is not None:
        env["PYTHONFAULTHANDLER"] = handler
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_a_crash_inside_the_redirect_leaves_the_stack_on_stderr():
    done = child("""
        def crash_here():
            ctypes.string_at(0)

        with open_part._output_seen():
            crash_here()
        """)
    assert done.returncode < 0
    assert "Fatal Python error" in done.stderr
    assert "in crash_here" in done.stderr


def test_the_handler_is_on_stderr_again_after_the_block():
    done = child("""
        with open_part._output_seen():
            print("inside", flush=True)
        print("enabled", faulthandler.is_enabled(), flush=True)
        ctypes.string_at(0)
        """)
    assert "enabled True" in done.stdout
    assert done.returncode < 0
    assert "Fatal Python error" in done.stderr


def test_a_handler_on_a_file_of_its_own_is_left_there(tmp_path):
    report = tmp_path / "faults.txt"
    done = child(
        f"""
        kept = open({str(report)!r}, "w")
        faulthandler.enable(file=kept, all_threads=True)
        with open_part._output_seen():
            pass
        ctypes.string_at(0)
        """,
        handler=None,
    )
    assert done.returncode < 0
    assert "Fatal Python error" in report.read_text()
    assert "Fatal Python error" not in done.stderr


def test_a_handler_that_was_off_stays_off():
    done = child(
        """
        with open_part._output_seen():
            pass
        print("enabled", faulthandler.is_enabled(), flush=True)
        """,
        handler=None,
    )
    assert done.returncode == 0 and "enabled False" in done.stdout
