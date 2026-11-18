"""The environment the checks start a child process with: Python's fault handler on and,
unless the user set one, a TRACY_PORT of its own, never handed out twice by this
process, so that each child's Tracy can listen however many start at once (tt-metal
issue 64717: a process's native load waits for that listener)."""

import socket
import threading

TRACY_PORT = "TRACY_PORT"
# Set beside a TRACY_PORT this module chose: a child of a child gets a port of its own
# (its parent is alive and listens on the inherited one); one without it is the user's.
PORT_CHOSEN = "KMR_TRACY_PORT_CHOSEN"

_handed_out = set()
_handed_out_lock = threading.Lock()


def free_port():
    """A TCP port free on this host now (the kernel's pick for a bind to port 0) that
    this process has not handed out before."""
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("", 0))
            port = probe.getsockname()[1]
        with _handed_out_lock:
            if port not in _handed_out:
                _handed_out.add(port)
                return port


def child_environment(environ, **extra):
    """`environ` with `extra`, PYTHONFAULTHANDLER=1 and, when it has no TRACY_PORT of
    the user's (none, empty, or one chosen here for a parent), a free TRACY_PORT with
    the PORT_CHOSEN mark; a new dict."""
    env = dict(environ, **extra)
    env["PYTHONFAULTHANDLER"] = "1"
    if env.get(TRACY_PORT) and not env.get(PORT_CHOSEN):
        env.pop(PORT_CHOSEN, None)
    else:
        env[TRACY_PORT] = str(free_port())
        env[PORT_CHOSEN] = "1"
    return env
