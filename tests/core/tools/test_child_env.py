"""The checks' child environment: the fault handler on, and a Tracy port of its own,
never handed out twice by one process, unless the user set one."""

from komorebi.tools.child_env import PORT_CHOSEN, child_environment


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


def test_a_chosen_port_is_marked_and_a_kept_one_is_not():
    assert child_environment({})[PORT_CHOSEN] == "1"
    assert PORT_CHOSEN not in child_environment({"TRACY_PORT": "9123"})


def test_a_port_with_the_mark_is_replaced():
    parent = child_environment({})
    child = child_environment(parent)
    assert port(child["TRACY_PORT"]) and child["TRACY_PORT"] != parent["TRACY_PORT"]
    assert child[PORT_CHOSEN] == "1"


def test_two_thousand_calls_give_two_thousand_ports():
    ports = {child_environment({})["TRACY_PORT"] for _ in range(2000)}
    assert len(ports) == 2000


def test_two_calls_give_ports_and_extra_names_go_in():
    first = child_environment({}, KMR_PREBUILD="1")
    second = child_environment({})
    assert port(first["TRACY_PORT"]) and port(second["TRACY_PORT"])
    assert first["KMR_PREBUILD"] == "1"
