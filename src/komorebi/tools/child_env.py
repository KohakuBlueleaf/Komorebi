"""The environment the checks start a child process with: Python's fault handler on (a
crash names itself) and, unless set, a TRACY_PORT of its own, a free port found here,
so that each child's Tracy can listen however many start at once (tt-metal issue
64717: a process's native load waits for that listener)."""

import socket

TRACY_PORT = "TRACY_PORT"


def free_port():
    """A TCP port free on this host now (the kernel's pick for a bind to port 0)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("", 0))
        return probe.getsockname()[1]


def child_environment(environ, **extra):
    """`environ` with `extra`, PYTHONFAULTHANDLER=1 and, when it has none, a free
    TRACY_PORT; a new dict."""
    env = dict(environ, **extra)
    env["PYTHONFAULTHANDLER"] = "1"
    if not env.get(TRACY_PORT):
        env[TRACY_PORT] = str(free_port())
    return env
