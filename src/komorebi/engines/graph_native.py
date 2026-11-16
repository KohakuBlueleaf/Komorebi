"""The graph engine's C file built for the host and driven through ctypes: the same
`kmg_*` functions the dispatcher runs, behind the interface of `graph.Executor`.
"""

import ctypes
import os
import shutil
import subprocess

from komorebi import native
from komorebi.engines import graph as graph_module

LIBRARY = "libkm_graph.so"
CFLAGS = ["-std=c11", "-O2", "-ffreestanding", "-fPIC", "-shared", "-Wall", "-Werror"]
U32P = ctypes.POINTER(ctypes.c_uint32)
SUBMIT_FUNCTION = ctypes.CFUNCTYPE(
    ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, U32P
)
STATE_WORDS = 20  # registers, runs, phase, error reason, error node

_loaded = {}


def engine_source():
    return native.native_directory() / "x280" / "engines" / "km_graph_engine.c"


def library_path(compiler=None):
    """The engine as a host shared library, built once into the cache."""
    compiler = compiler or shutil.which("cc")
    if compiler is None:
        raise native.build.BuildError("building the graph engine needs a C compiler")
    source = engine_source()
    include = native.native_directory() / "include"
    key = native.source_key([source.parent, include / "komorebi" / "abi"], compiler)
    out = native.cache_directory() / f"graph-engine-{key}" / LIBRARY
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_suffix(f".{os.getpid()}.tmp")
        command_line = [
            compiler,
            *CFLAGS,
            f"-I{include}",
            str(source),
            "-o",
            str(temporary),
        ]
        done = subprocess.run(command_line, capture_output=True, text=True)
        if done.returncode:
            raise native.build.BuildError(f"{' '.join(command_line)}\n{done.stderr}")
        os.replace(temporary, out)
    return out


def load(compiler=None):
    """The engine library with its function types set."""
    path = library_path(compiler)
    if path not in _loaded:
        lib = ctypes.CDLL(str(path))
        u32, void_pointer = ctypes.c_uint32, ctypes.c_void_p
        lib.kmg_state_bytes.restype = u32
        lib.kmg_check.argtypes, lib.kmg_check.restype = [void_pointer, u32], u32
        lib.kmg_start.argtypes = [void_pointer, void_pointer, U32P, u32, u32]
        lib.kmg_start.restype = u32
        lib.kmg_island.argtypes = [void_pointer, u32, u32]
        lib.kmg_island.restype = None
        lib.kmg_step.argtypes, lib.kmg_step.restype = [
            void_pointer,
            SUBMIT_FUNCTION,
            void_pointer,
        ], u32
        lib.kmg_complete.argtypes, lib.kmg_complete.restype = [
            void_pointer,
            u32,
            u32,
        ], u32
        lib.kmg_retry.argtypes, lib.kmg_retry.restype = [void_pointer, u32], u32
        lib.kmg_stop.argtypes, lib.kmg_stop.restype = [void_pointer, u32, u32], None
        _loaded[path] = lib
    return _loaded[path]


def native_check(data, lib=None):
    """`kmg_check` on an upload: None or `GraphError(reason, node)`."""
    lib = lib or load()
    buffer = ctypes.create_string_buffer(bytes(data), max(len(data), 1))
    got = lib.kmg_check(buffer, len(data))
    return graph_module.GraphError(got & 0xFF, got >> 8) if got else None


class NativeExecutor:
    """`kmg_start`, `kmg_step`, `kmg_complete`, `kmg_retry` and `kmg_stop` on a state
    block."""

    def __init__(self, lib=None):
        self.lib = lib or load()
        self._state = ctypes.create_string_buffer(self.lib.kmg_state_bytes())
        self._upload = None

    def _words(self):
        return ctypes.cast(self._state, U32P)[:STATE_WORDS]

    @property
    def registers(self):
        return self._words()[: graph_module.REGISTERS]

    @property
    def runs(self):
        return self._words()[16]

    @property
    def phase(self):
        return self._words()[17]

    @property
    def error(self):
        words = self._words()
        return (
            graph_module.GraphError(words[18], words[19])
            if words[17] == graph_module.ERROR
            else None
        )

    def start(self, data, bindings=(), core_count=119):
        self._upload = ctypes.create_string_buffer(bytes(data), len(data))
        values = (ctypes.c_uint32 * max(len(bindings), 1))(*bindings)
        return self.lib.kmg_start(
            self._state, self._upload, values, len(bindings), core_count
        )

    def island(self, ranks):
        """`kmg_island`: a node's group may name any of `ranks` flat ranks."""
        self.lib.kmg_island(self._state, 0, ranks)

    def step(self, submit):
        """One `kmg_step`; `submit(node, words)` gets the command's 16 words. An error
        raised by `submit` counts as busy inside the step and is raised after it."""
        raised = []

        def call(_ctx, node, rec):
            if raised:
                return graph_module.BUSY
            try:
                return submit(node, list(rec[:16]))
            except Exception as error:  # noqa: BLE001 (raised again below)
                raised.append(error)
                return graph_module.BUSY

        code = self.lib.kmg_step(self._state, SUBMIT_FUNCTION(call), None)
        if raised:
            raise raised[0]
        return code

    def complete(self, node, result):
        return self.lib.kmg_complete(self._state, node, result & graph_module.MASK)

    def retry(self, node):
        return self.lib.kmg_retry(self._state, node)

    def stop(self, reason, node):
        self.lib.kmg_stop(self._state, reason, node)
