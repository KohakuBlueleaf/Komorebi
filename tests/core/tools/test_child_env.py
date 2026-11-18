"""The checks' child environment: the fault handler on, and a Tracy port of its own
unless one is set."""

from komorebi.tools.child_env import child_environment


def port(value):
    return value.isdigit() and 0 < int(value) < 65536


def test_the_fault_handler_is_on_and_an_unset_port_is_found():
    env = child_environment({"PATH": "/bin"})
    assert env["PYTHONFAULTHANDLER"] == "1" and env["PATH"] == "/bin"
    assert port(env["TRACY_PORT"])


def test_an_empty_port_is_found():
    assert port(child_environment({"TRACY_PORT": ""})["TRACY_PORT"])


def test_a_port_already_set_is_kept():
    assert child_environment({"TRACY_PORT": "9123"})["TRACY_PORT"] == "9123"


def test_two_calls_give_ports_and_extra_names_go_in():
    first = child_environment({}, KMR_PREBUILD="1")
    second = child_environment({})
    assert port(first["TRACY_PORT"]) and port(second["TRACY_PORT"])
    assert first["KMR_PREBUILD"] == "1"
