"""What a program written as a class names: the handler decorator, fields, formats,
and the calls of a handler's lines. The lines are read by the compiler, never run in
Python: calling one outside a class's handler raises ProgramError.
"""

import inspect
import operator
from dataclasses import dataclass

from komorebi import abi
from komorebi.dsl.names import NAMES, spelling
from komorebi.dsl.tile import trace
from komorebi.errors import ProgramError
from komorebi.movement import Pattern
from komorebi.program import Handler
from komorebi.tensor import Tensor


@dataclass(frozen=True)
class Format:
    """A data format of a tile or a field: its dtype name and bytes an item."""

    dtype: str
    itemsize: int


FORMATS = {
    "fp32": Format("float32", 4),
    "bf16": Format("bfloat16", 2),
    "u32": Format("uint32", 4),
}


@dataclass(frozen=True)
class Field:
    """A field resident in each core's L1: its format and shape (ints, or symbols the
    build solves); the annotation of `km.L1[format, *shape]`."""

    format: Format
    shape: tuple


class Symbol:
    """`K = km.symbol()` in a class body: a size of a field's shape the plan gives
    or the build solves; its name is the attribute's."""

    def __init__(self):
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __repr__(self):
        return f"km.{spelling('symbol')}() {self.name}"


def symbol():
    return Symbol()


# The CPUs that may push an own field's buffer: a mover, or compute's (its pack CPU).
OWN_PUSHERS = ("mover_a", "mover_b", "pack")


class Own:
    """`p: km.L1[format, *shape] = km.own(push=...)`: a field only bodies use, never
    filled, waited for or freed by a line. `push` the one CPU whose bodies push it
    (OWN_PUSHERS), None for one written in place and never pushed; not given, the
    build finds it from the bodies (one CPU at most)."""

    def __init__(self, push="found"):
        if push != "found" and push is not None and push not in OWN_PUSHERS:
            raise ProgramError(
                f"km.{spelling('own')}(push={push!r}): push= is the one CPU that "
                f"pushes it ({', '.join(map(repr, OWN_PUSHERS))}) or None"
            )
        self.push = push


def own(push="found"):
    return Own(push)


class Tile:
    """`gives=km.Tile`: what an inline body of compute leaves in DST 0."""


class L1:
    """`km.L1[format, *shape]`: the annotation of a field resident in the L1, each
    size an int or a km.symbol() of the class."""

    def __class_getitem__(cls, key):
        key = key if isinstance(key, tuple) else (key,)
        format_, *shape = key
        if not isinstance(format_, Format):
            raise ProgramError(
                f"km.{spelling('field')}[format, *shape]: the format first, km.fp32, "
                "km.bf16 or km.u32"
            )
        if not shape or not all(
            isinstance(n, Symbol) or (isinstance(n, int) and not isinstance(n, bool))
            for n in shape
        ):
            raise ProgramError(
                f"a field's sizes are ints and km.{spelling('symbol')}()s of the class "
                f"(K = km.{spelling('symbol')}() in its body): got {tuple(shape)!r}"
            )
        return Field(format_, tuple(shape))


class Message:
    """`class Visit(km.Message)` with int annotations: a dynamic handler's message,
    its named words 1, 2, ... in order (word 0 is the runtime's mark)."""

    words = {}  # name -> its word

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        names = list(inspect.get_annotations(cls))
        if not all(inspect.get_annotations(cls)[name] is int for name in names):
            raise ProgramError(f"{cls.__name__}: a message's words are ints")
        if not 1 <= len(names) <= MESSAGE_WORDS - 1:
            raise ProgramError(
                f"{cls.__name__}: 1 to {MESSAGE_WORDS - 1} named words, "
                f"{len(names)} given"
            )
        cls.words = {name: word for word, name in enumerate(names, start=1)}

    def __init__(self, **words):
        raise ProgramError(
            f"km.{spelling('message')} is a line of a handler: the compiler reads it, "
            "Python never runs it"
        )


MESSAGE_WORDS = 16  # a message of 64 bytes, the runtime's default


class ClassHandler(Handler):
    """A handler of a program made from a class: a Handler (its number and program)
    whose `args(...)` gives the command words of a call."""

    def __new__(cls, number, program, method):
        made = super().__new__(cls, number, program)
        made.method = method
        return made

    def args(self, *values, **named):
        """The command words of a call with these arguments."""
        return self.method.args(*values, **named)


class HandlerMethod:
    """A method decorated with `km.handler`: on the class, its number once the class
    is prepared (an int) and `args(...)`; on a made program, a ClassHandler."""

    def __init__(self, function, overlay=False, stamps=None):
        self.function = function
        self.overlay = overlay  # its code is an overlay: loaded into a slot, called
        self.stamps = stamps  # the tensor parameter its load and call cycles go to
        self.name = function.__name__
        self.number = None
        self.words = None  # [(parameter, kind)] in word order, set by the compiler
        self.owner = None  # the class it is a handler of
        self.returns = False  # it returns a word: its commands ask for the result

    def __set_name__(self, owner, name):
        self.name = name
        self.owner = owner

    def __get__(self, program, owner=None):
        if program is None:
            return self
        return ClassHandler(self.number, program, self)

    def __int__(self):
        if self.number is None:
            raise ProgramError(f"{self.name}: its class is not prepared")
        return self.number

    def __call__(self, *args, **kwargs):
        raise ProgramError(
            f"{self.name} is a handler: it runs on the cores; submit it "
            f"(run.submit(Class.{self.name}, group, args=(...)))"
        )

    def args(self, *values, **named):
        """The command words of a call: a tensor gives its reference word (and its
        page count, if the handler counts its pages), an int itself, a pattern its
        word."""
        if self.words is None:
            raise ProgramError(f"{self.name}: its program is not made yet")
        given = dict(zip([p for p, _ in self.parameters()], values))
        if len(values) > len(given):
            raise ProgramError(f"{self.name}: {len(values)} arguments, too many")
        given.update(named)
        words = []
        for parameter, kind in self.words:
            value = given.get(parameter)
            if value is None:
                raise ProgramError(f"{self.name}: no argument {parameter!r}")
            words.append(word_of(self.name, parameter, kind, value))
        return words

    def parameters(self):
        """The (name, kind) of each parameter, in order, without page counts."""
        return [(p, kind) for p, kind in self.words if kind != "pages"]


def is_tensor(value):
    """A DRAM tensor of a session or an island session: what a class reads of it (its
    reference word, format, page bytes and accessor arguments)."""
    return isinstance(value, Tensor) or all(
        hasattr(value, name) for name in ("address", "dtype", "page_bytes", "placement")
    )


def word_of(handler, parameter, kind, value):
    """One command word for an argument of `kind`."""
    if kind == "tensor":
        if not is_tensor(value):
            raise ProgramError(f"{handler}: {parameter} is a tensor, got {value!r}")
        return int(value.address) & abi.M32
    if kind == "pages":
        return int(value.pages) & abi.M32
    if kind == "pattern":
        pattern = value if isinstance(value, Pattern) else Pattern(value)
        return pattern.word
    if isinstance(value, bool) or not hasattr(value, "__index__"):
        raise ProgramError(f"{handler}: {parameter} is an int, got {value!r}")
    value = operator.index(value)
    if not 0 <= value <= abi.M32:
        raise ProgramError(f"{handler}: {parameter} = {value}: a 32-bit word")
    return value


def handler(function=None, *, overlay=False, stamps=None):
    """Makes a method of a class deriving km.Program a handler; `overlay=True`: its
    code is built apart as an overlay, loaded into a slot when a command needs it;
    `stamps="t"`: each mover's load and call cycles to tensor parameter t."""
    if function is None:
        return lambda function: HandlerMethod(function, overlay, stamps)
    return HandlerMethod(function, overlay, stamps)


# The matrix unit's fidelities (tt-metal's MathFidelity) and their phases.
FIDELITIES = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}


class TileFunction:
    """A function decorated with `km.tile_function`: tile arithmetic the tile
    compiler traces and emits for compute when a handler calls it on a stream's
    tiles; `tolerance` its stated bound against float64 (printed at the build);
    `fidelity` a product's matrix-unit fidelity (tt-metal's word)."""

    def __init__(self, function, tolerance=None, fidelity=None):
        if fidelity not in (None, *FIDELITIES):
            raise ProgramError(
                f"{function.__name__}: fidelity={fidelity!r} is not one of "
                f"{', '.join(FIDELITIES)}"
            )
        self.function = function
        self.tolerance = tolerance
        self.fidelity = fidelity
        self.name = function.__name__

    def __get__(self, program, owner=None):
        return self

    def __call__(self, *args, **kwargs):
        raise ProgramError(
            f"{self.name} is a tile function: a handler calls it on a stream's "
            "tiles, and the compiler emits it for compute"
        )


def tile_function(function=None, *, tolerance=None, fidelity=None):
    """Makes a function of tiles a tile function: called from a handler on the tiles
    of its streams, traced and emitted for compute by the tile compiler;
    `tolerance`: its stated largest relative error against float64; `fidelity`: a
    product's ("HiFi4" when not given)."""
    if function is None:
        return lambda function: TileFunction(function, tolerance, fidelity)
    return TileFunction(function, tolerance, fidelity)


def line_only(kind):
    """A call that is a line of a handler, read by the compiler."""

    def call(*args, **kwargs):
        raise ProgramError(
            f"km.{spelling(kind)} is a line of a handler: the compiler reads it, "
            "Python never runs it"
        )

    call.__name__ = spelling(kind)
    return call


class TileNamespace:
    """`km.tile`: the tile values of compute (`km.tile.zeros(km.fp32)`)."""

    def __getattr__(self, name):
        if name == spelling("zeros"):
            return line_only("zeros")
        raise AttributeError(f"km.{spelling('tile')} has no {name!r}")


def tile_math(kind):
    """The tile compiler's trace function a km. name of a tile function's arithmetic
    resolves to."""
    return getattr(trace, NAMES[kind].released.removeprefix("trace."))


def exports():
    """The module-level names of `komorebi` for programs written as classes."""
    made = {
        "handler": handler,
        "tile_function": tile_function,
        "message": Message,
        "field": L1,
        "symbol": symbol,
        "own": own,
        "tile_value": Tile,
        "tensor": Tensor,
        "stream": line_only("stream"),
        "reduce": line_only("reduce"),
        "broadcast": line_only("broadcast"),
        "cxx": line_only("cxx"),
        "tile": TileNamespace(),
        **{kind: FORMATS[kind] for kind in FORMATS},
        **{kind: tile_math(kind) for kind in NAMES if kind.startswith("math_")},
    }
    return {NAMES[kind].spelling: value for kind, value in made.items()}
