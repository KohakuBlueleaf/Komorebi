"""A class's program built through the Komorebi IR, beside today's make (unchanged):
its module compiled, emitted on Tensix and loaded with the make's tensors, its files
in today's folder, a `km_program.Program` of the output (docs/ir/frontends.md). It
imports the komorebi-ir wheel; `komorebi.dsl.ir` does not import it."""

from pathlib import Path

import komorebi_ir
from komorebi.dsl.compile.source import file_of
from komorebi.dsl.files import put, text_key
from komorebi.dsl.ir.walk import given_tensors, module
from komorebi.dsl.program import BUILD
from komorebi.program import description as km_program


def load_tensor(tensor):
    """A make's tensor as a load takes it: its pages, its layout words (what km::Tensor
    reads) and its page in DRAM, when it is in DRAM and the tensor knows it."""
    page = getattr(tensor, "aligned_page_bytes", None)
    if not getattr(tensor, "in_dram", True):
        page = None
    return komorebi_ir.LoadTensor(int(tensor.pages), list(tensor.tensor_args()), page)


def buffers_of(loaded):
    """The buffers as Program takes them: a pair once, as `km_program.BufferPair` of
    its two halves (the first listed is mover A's), a lone buffer as itself."""
    out, seen = [], set()
    by_index = {b.index: b for b in loaded.buffers}
    for b in loaded.buffers:
        if b.index in seen:
            continue
        if b.pair is None:
            out.append(km_program.Buffer(b.index, b.dtype, b.page_bytes, b.pages))
            continue
        other = by_index[b.pair]
        seen.update((b.index, other.index))
        out.append(
            km_program.BufferPair(
                b.index, other.index, b.dtype, b.page_bytes, (b.pages, other.pages)
            )
        )
    return out


def compile_args_of(loaded):
    """The kernels' compile arguments by kernel: one "movement" entry when both movers
    build one source with one list (today's kernel form), else each its own."""
    by = {k.processor: k for k in loaded.kernels}
    out = {}
    reader, writer = by.get("reader"), by.get("writer")
    if (
        reader
        and writer
        and (reader.source, reader.args) == (writer.source, writer.args)
    ):
        if reader.args:
            out["movement"] = list(reader.args)
    else:
        for kernel in (reader, writer):
            if kernel is not None and kernel.args:
                out[kernel.processor] = list(kernel.args)
    compute = by.get("compute")
    if compute is not None and compute.args:
        out["compute"] = list(compute.args)
    return out


def program(cls, **make):
    """The `km_program.Program` of class `cls` made with `make` (a `km.Program`'s
    keywords, as `module` takes them), built through the IR; its files written to
    build/<name>-<text_key> beside the class's source, as today's make writes them."""
    log = make.get("log", print)
    made = module(cls, **make)
    compiled = komorebi_ir.compile(made.text, komorebi_ir.Origin(*made.origin))
    out = komorebi_ir.emit(compiled, "tensix")
    tensors = {
        name: load_tensor(tensor) for name, tensor in given_tensors(cls, make).items()
    }
    loaded = komorebi_ir.load(compiled, tensors)
    files = {name: data.decode() for name, data in out.files}
    files[loaded.thread_name] = loaded.thread
    name = cls.__name__.lower()
    digest = text_key(files)
    folder = Path(file_of(cls)).parent / BUILD / f"{name}-{digest}"
    folder.mkdir(parents=True, exist_ok=True)
    for file, text in files.items():
        put(folder / file, text)
    log(f"{cls.__name__}: built through the IR, files {digest} (key {out.key[:16]})")
    compute = next((k for k in loaded.kernels if k.processor == "compute"), None)
    return km_program.Program(
        name=name,
        kernels={k.processor: folder / k.source for k in loaded.kernels},
        buffers=buffers_of(loaded),
        compile_args=compile_args_of(loaded),
        compute=dict(compute.options) if compute is not None else {},
        defines={},
        depth=make.get("depth", 4),
        handlers=folder / f"{name}.h",
    )
