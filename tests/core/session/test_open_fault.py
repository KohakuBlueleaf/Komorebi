"""The open's output redirect keeps an enabled fault handler's report: a crash inside
it leaves Python's stack on the real stderr, and the handler is as before after it."""

import os
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def child(body, handler=True):
    """Runs `body` in a fresh interpreter with the open's module imported (as
    `open_part`) and the fault handler on or off; its CompletedProcess."""
    code = "import ctypes, faulthandler, sys\n"
    code += "from komorebi.session.parts import open as open_part\n"
    code += "faulthandler.enable(all_threads=True)\n" if handler else ""
    code += textwrap.dedent(body)
    paths = [os.path.join(ROOT, "src"), os.environ.get("PYTHONPATH", "")]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in paths if p))
    env.pop("PYTHONFAULTHANDLER", None)
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


def test_a_handler_that_was_off_stays_off():
    done = child(
        """
        with open_part._output_seen():
            pass
        print("enabled", faulthandler.is_enabled(), flush=True)
        """,
        handler=False,
    )
    assert done.returncode == 0 and "enabled False" in done.stdout
