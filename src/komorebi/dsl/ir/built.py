"""A class's program built through the Komorebi IR, beside today's make (unchanged):
its module compiled, emitted on Tensix and loaded with the make's tensors, its files
in today's folder, a `km_program.Program` of the output (docs/ir/frontends.md). It
imports the komorebi-ir wheel; `komorebi.dsl.ir` does not import it."""

import inspect
from dataclasses import dataclass, field
from pathlib import Path

import komorebi_ir
from komorebi import abi
from komorebi.dsl.compile.source import file_of
from komorebi.dsl.files import put, text_key
from komorebi.dsl.front.front import word_of
from komorebi.dsl.ir.walk import given_tensors, module
from komorebi.dsl.program import BUILD
from komorebi.errors import ProgramError
from komorebi.program import description as km_program

# a scalar parameter's kind as today's command word takes it (front.word_of)
WORD_KINDS = {"f32": "float", "u8": "int", "u16": "int", "u32": "int", "i32": "int"}


@dataclass
class Built(km_program.Program):
    """A `km_program.Program` built through the IR, with its handlers made from the
    output's entry points: `entry(name)` is what Run.submit takes."""

    entries: dict = field(default_factory=dict, compare=False)

    def entry(self, name):
        """The handler `name` of this program (an `Entry`)."""
        if name not in self.entries:
            raise KeyError(f"{self.name} has no handler {name!r}")
        return self.entries[name]


class Entry(km_program.Handler):
    """A handler from its `komorebi_ir.EntryPoint`: its number (a Handler) and
    `args(...)`, the command words of a call, each range the IR states checked
    against its tensor before anything is sent (docs/ir/frontends.md)."""

    def __new__(cls, point, program, names):
        made = super().__new__(cls, point.number, program)
        made.point, made.names = point, list(names)
        made.short = point.name.split(".")[-1]
        return made

    def args(self, *values, group=None):
        """The command words of a call with these arguments, in the entry's order;
        `group`, the submit's (first, count), when a range reads the rank or size."""
        point, short = self.point, self.short
        if len(values) != len(self.names):
            raise ProgramError(
                f"{short}: {len(values)} arguments, the handler takes {len(self.names)}"
            )
        words = [self.word(kind, p, values[p]) for kind, p in point.words]
        for tensor, first, count in point.bounds:
            ranked = reads_group(first) or reads_group(count)
            if ranked and group is None:
                raise ProgramError(
                    f"{short}: a range of {self.names[tensor]} reads the rank: give "
                    "args(..., group=(first, count)), the submit's group"
                )
            size = int(group[1]) if ranked else None
            for rank in range(size) if ranked else (None,):
                at = {"rank": rank, "size": size}
                self.bound_checked(
                    values[tensor], self.names[tensor], first, count, words, at
                )
        return words

    def word(self, kind, p, value):
        """The word of parameter `p` given `value`: a tensor's address or tiles, a
        scalar's word as today's make gives it."""
        name = self.names[p]
        if kind == "address":
            return word_of(self.short, name, "tensor", value)
        if kind == "tiles":
            return tiles_of(value) & abi.M32
        scalar = self.point.params[p]
        if scalar == "bool":
            if not isinstance(value, bool):
                raise ProgramError(f"{self.short}: {name} is a bool, got {value!r}")
            return int(value)
        if scalar not in WORD_KINDS:
            raise ProgramError(
                f"{self.short}: {name} is a {scalar}, a word the IR's handler does not "
                "send yet"
            )
        return word_of(self.short, name, WORD_KINDS[scalar], value)

    def bound_checked(self, tensor, name, first, count, words, at):
        """Refuses a call whose tiles `first` to `first + count - 1` of `tensor` (at
        rank `at["rank"]` of `at["size"]`, when they read it) are not all in it, in the
        words of today's ranges (pages.ranges.at_submit)."""
        start, many = value_of(first, words, at), value_of(count, words, at)
        held = tiles_of(tensor)
        if many < 1 or start + many <= held:
            return
        per_page = getattr(tensor, "page_tiles", 1)
        size = tensor.page_bytes // per_page
        written = f"{name}.tiles({self.text(first)}, {self.text(count)})"
        rank = at["rank"]
        where = "" if rank is None else f" at rank {rank} of {at['size']}"
        raise ProgramError(
            f"{self.short}: {written}: items {start} to {start + many - 1} of {name}"
            f"{where} at this call, and {name} holds {held} items of {size} B "
            f"({tensor.pages} page(s) of {per_page}): keep the range inside the tensor"
        )

    def text(self, expr, inner=False):
        """An end of a range as the handler's parameters and the IR's words write it."""
        if expr[0] == "const":
            return str(expr[1])
        if expr[0] in ("rank", "size"):
            return {"rank": "worker.rank", "size": "group.size"}[expr[0]]
        if expr[0] in ("add", "mul"):
            sign = {"add": "+", "mul": "*"}[expr[0]]
            text = f"{self.text(expr[1], True)} {sign} {self.text(expr[2], True)}"
            return f"({text})" if inner else text
        kind, p = self.point.words[expr[1]]
        return self.names[p] if kind == "word" else f"{self.names[p]}'s {kind}"


def tiles_of(tensor):
    """The tiles a tensor holds: its pages of `page_tiles` tiles each."""
    return int(tensor.pages) * getattr(tensor, "page_tiles", 1)


def reads_group(expr):
    """Whether an end of a range reads the worker's rank or the group's size."""
    if expr[0] in ("rank", "size"):
        return True
    return expr[0] in ("add", "mul") and (reads_group(expr[1]) or reads_group(expr[2]))


def value_of(expr, words, at):
    """An end of a range at a call: of its words, constants, the rank and size in
    `at`, sums and products."""
    kind = expr[0]
    if kind == "word":
        return words[expr[1]]
    if kind == "const":
        return expr[1]
    if kind in ("rank", "size"):
        return at[kind]
    left, right = value_of(expr[1], words, at), value_of(expr[2], words, at)
    return left + right if kind == "add" else left * right


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
    """The `Built` program of class `cls` made with `make` (a `km.Program`'s keywords,
    as `module` takes them), built through the IR; its files written to
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
    made = Built(
        name=name,
        kernels={k.processor: folder / k.source for k in loaded.kernels},
        buffers=buffers_of(loaded),
        compile_args=compile_args_of(loaded),
        compute=dict(compute.options) if compute is not None else {},
        defines={},
        depth=make.get("depth", 4),
        handlers=folder / f"{name}.h",
    )
    for point in out.entries:
        short = point.name.split(".")[-1]
        names = list(inspect.signature(getattr(cls, short).function).parameters)[1:]
        made.entries[short] = Entry(point, made, names)
    return made
