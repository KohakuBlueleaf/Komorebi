"""O3, O4, O7, O8, O9 of the session's open and close, each forced on the mock."""

import inspect
import os
import signal
import sys
import threading
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.device import mock, native, remembered
from komorebi.session import kernels as kernel_module
from komorebi.session import process
from komorebi.session.backend.backend import _Backend
from komorebi.session.parts import open as open_part

sys.path.insert(0, "examples/29_chain/as_class")  # from the checkout's root
sys.path.insert(0, "examples/26_islands_sum")
import sum as islands_sum
from steps_class import WORDS, Steps

HERE = Path(__file__).parent
said = []


def log(message):
    said.append(message)


def say(name, text):
    print(f"CASE {name}: {text}", flush=True)


def steps(session):
    zeros = np.zeros((16, WORDS), np.float32)
    t = {
        n: session.tensor(zeros if n != "t" else zeros[:1], "float32") for n in "xyzst"
    }
    return Steps(a=t["x"], b=t["x"], out=t["y"], s=t["s"], t=t["t"], session=session)


def o8_thread():
    """O8b: the first open on another thread: said; then the main thread's installs."""
    box = []

    def opened():
        with km.Session(card=0, log=log) as session:
            box.append(session.device is not None)

    worker = threading.Thread(target=opened)
    worker.start()
    worker.join()
    line = [m for m in said if "SIGTERM hook" in m]
    say(
        "O8b thread",
        f"opened {box}; said {line}; hooked "
        f"{signal.getsignal(signal.SIGTERM) is process._on_sigterm}",
    )
    with km.Session(card=0, log=log):
        pass
    say(
        "O8b main",
        "hooked after a main-thread open: "
        f"{signal.getsignal(signal.SIGTERM) is process._on_sigterm}",
    )


def o8_cards():
    os.environ.pop("TT_VISIBLE_DEVICES", None)
    with km.Session(card=0, log=log):
        during = os.environ.get("TT_VISIBLE_DEVICES")
    after = os.environ.get("TT_VISIBLE_DEVICES")
    try:
        with km.Session(card=1, log=log):
            second = os.environ.get("TT_VISIBLE_DEVICES")
            second = f"card 1 opened (TT_VISIBLE_DEVICES {second})"
    except Exception as error:  # noqa: BLE001
        second = f"{type(error).__name__}: {error}"
    say(
        "O8a",
        f"during card 0 {during}; after its close {after}; then {second}; "
        f"after {os.environ.get('TT_VISIBLE_DEVICES')}",
    )
    os.environ["TT_VISIBLE_DEVICES"] = "1"  # the user's own: still refused for card 0
    try:
        with km.Session(card=0, log=log):
            pass
        say("O8a user", "card 0 opened under the user's TT_VISIBLE_DEVICES=1 (WRONG)")
    except ValueError as error:
        say("O8a user", f"refused: {error}")
    os.environ.pop("TT_VISIBLE_DEVICES", None)


def o4():
    session = km.Session(card=0, log=log)
    session.open()
    try:
        session.open()
        say("O4", "a second open was taken (WRONG)")
    except km.errors.KomorebiError as error:
        say("O4", f"refused: {error}")
    session.close()


def o3():
    folder = HERE / "readonly"
    folder.mkdir(exist_ok=True)
    os.chmod(folder, 0o555)
    real = remembered._path
    remembered._path = lambda: folder / remembered.FILE
    try:
        session = km.Session(card=0, log=log)
        session.open()
        program = steps(session)
        run = session.load(program)
        owned = session.run is run
        run.stop()
        session.close()
        say(
            "O3",
            f"load returned, run owned {owned}; close: device {session.device}, "
            f"live {session in process._LIVE}",
        )
    finally:
        remembered._path = real


def o9():
    with km.Session(card=0, log=log, kernel_bytes=16 << 10) as session:
        session.kernel_followed = True  # a session that follows its kernels
        session.stuck = True  # stand-in: a GridTimeout before this load
        try:
            session.load(steps(session))
            say("O9", "loaded (WRONG)")
        except km.errors.KernelSizeError as error:
            say("O9", f"refused: {str(error)[-260:]}")
        session.stuck = False


def o7():
    real_active, real_close, real_stop = (
        mock.active,
        kernel_module._Dispatches.close,
        _Backend.stop_grid,
    )
    stopped = []

    def active():
        frame = inspect.currentframe().f_back
        if (
            frame.f_code.co_name == "load_islands"
            and "# built: a mock run an island"
            in inspect.getsource(frame).splitlines()[
                frame.f_lineno - frame.f_code.co_firstlineno
            ]
        ):
            return False
        return real_active()

    def close(self):
        raise RuntimeError("forced: the runtime's close raised")

    def stop_grid(self):
        stopped.append(self.rows)

    with km.Session(card=0, islands=2, log=log) as session:
        x = session.tensor(np.ones((64, 1024), np.float32), "float32", copy=True)
        y = session.tensor(np.zeros((1, 1024), np.float32), "float32", copy=True)
        mock.active, kernel_module._Dispatches.close, _Backend.stop_grid = (
            active,
            close,
            stop_grid,
        )
        try:
            session.load(islands_sum.program(x, y))
            say("O7", "loaded (WRONG: the launch should fail on the mock)")
        except Exception as error:  # noqa: BLE001
            say(
                "O7", f"{type(error).__name__}: {error}; stop_grid on islands {stopped}"
            )
        finally:
            mock.active, kernel_module._Dispatches.close, _Backend.stop_grid = (
                real_active,
                real_close,
                real_stop,
            )


def r23():
    """Row 23: a non-channel error after the device is built closes it."""
    real = open_part.OpenMixin._open_channel

    def failing(self, said):
        raise OSError("forced: /proc/self/maps unreadable")

    open_part.OpenMixin._open_channel = failing
    session = km.Session(card=0, log=log)
    try:
        session.open()
        say("r23", "opened (WRONG)")
    except OSError as error:
        say("r23", f"{error}; device {session.device}; live {session in process._LIVE}")
    finally:
        open_part.OpenMixin._open_channel = real


def r26():
    with km.Session(card=0, log=log) as session:
        during = session.channel is not None or session._mock_channel is not None
    say("r26", f"a channel during {during}; after the close channel {session.channel}")


def r21():
    with km.Session(card=0, log=log) as opened:
        real_active = mock.active
        given = km.Session(device=opened.device, log=log)
        os.environ.pop("TT_VISIBLE_DEVICES", None)
        given.card = None
        try:
            mock.active = lambda: False
            given._clock_mhz()
            say("r21", "a clock read (WRONG)")
        except km.errors.KomorebiError as error:
            say("r21", f"refused: {error}")
        finally:
            mock.active = real_active
        os.environ["TT_VISIBLE_DEVICES"] = "1"
        try:
            say("r21 one card", f"card 1's clock {given._clock_mhz()} MHz")
        except km.errors.KomorebiError as error:
            say("r21 one card", f"named: {error}")
        os.environ.pop("TT_VISIBLE_DEVICES", None)


def r24():
    with km.Session(cards=[0, 1], links=1, log=log) as session:
        island = session.sessions[1]
        real = type(island).close

        def raising(self):
            if self is island:
                real(self)
                raise RuntimeError("forced: island 1's close raised")
            return real(self)

        type(island).close = raising
        said.clear()
        try:
            session.close()
            say("r24", "closed (WRONG)")
        except RuntimeError as error:
            lines = [m for m in said if "island 1" in m or "mesh" in m]
            say("r24", f"raised {error}; mesh {session.mesh}; said {lines}")
        finally:
            type(island).close = real


def r28():
    module = native.load_module()
    try:
        module.format_tables(
            ["bfloat16"] * 3, True, ["UnpackToDestFP32"], False, "blackhole"
        )
        say("r28", "a misspelt mode taken (WRONG)")
    except ValueError as error:
        say("r28", f"refused: {error}")
    tables = module.format_tables(
        ["bfloat16"] * 3, True, ["UnpackToDestFp32"], False, "blackhole"
    )
    say("r28 right", f"{len(tables)} tables for the right spelling")


ONLY = sys.argv[1:]
for name, case in (
    ("O8b", o8_thread),
    ("r28", r28),
    ("r23", r23),
    ("r26", r26),
    ("r21", r21),
    ("r24", r24),
    ("O8a", o8_cards),
    ("O4", o4),
    ("O3", o3),
    ("O9", o9),
    ("O7", o7),
):
    if (name == "r24") != (ONLY == ["r24"]):  # the mesh: a process of its own
        continue
    if ONLY and name not in ONLY:  # the cases named, else all but r24
        continue
    try:
        case()
    except Exception as error:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        say(name, f"RAISED {type(error).__name__}: {error}")
