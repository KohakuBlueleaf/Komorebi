"""What test_class_first_use.py and other tests share: Added, FakeRun, fresh, made."""

import shutil

from class_handler_constants_fixture import TENSORS, Gate, quiet

from komorebi.dsl import first_use
from komorebi.session.run import Run


def made(**given):
    return Gate(l1=1 << 20, log=quiet, **TENSORS, **given)


class Added:
    def __init__(self, number):
        self.number = number


class FakeRun:
    """What a first use reads of a run (its programs, what was added) and run.addon,
    which records each add."""

    def __init__(self, programs):
        self.programs, self.added, self.adds = programs, {}, []

    def addon(self, name, parts, handler=False, defines=None):
        assert handler
        self.added[name] = Added(20 + len(self.added))
        self.adds.append((name, parts, defines))
        return self.added[name].number

    _built_for = Run._built_for
    _handler = Run._handler
    _class_form_refused = Run._class_form_refused
    _program_told = Run._program_told
    program_field = Run.program_field


def fresh(program, size):
    """The set's build folder, removed (an earlier test run wrote it)."""
    folder = first_use.folder_of(program, Gate.gate, {"size": size})
    shutil.rmtree(folder, ignore_errors=True)
    return folder
