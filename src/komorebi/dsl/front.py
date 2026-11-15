"""What a program written as a class names: the handler decorator, fields, formats,
and the calls of a handler's lines. The lines are read by the compiler, never run in
Python: calling one outside a class's handler raises ProgramError.
"""

import operator
from dataclasses import dataclass

from komorebi import abi
from komorebi.dsl.names import NAMES, spelling
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

    def __init__(self, function, overlay=False):
        self.function = function
        self.overlay = overlay  # its code is an overlay: loaded into a slot, called
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


def handler(function=None, *, overlay=False):
    """Makes a method of a class deriving km.Program a handler; `overlay=True`: its
    code is built apart as an overlay, loaded into a slot when a command needs it."""
    if function is None:
        return lambda function: HandlerMethod(function, overlay)
    return HandlerMethod(function, overlay)


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


def exports():
    """The module-level names of `komorebi` for programs written as classes."""
    made = {
        "handler": handler,
        "field": L1,
        "symbol": symbol,
        "tile_value": Tile,
        "tensor": Tensor,
        "stream": line_only("stream"),
        "reduce": line_only("reduce"),
        "broadcast": line_only("broadcast"),
        "cxx": line_only("cxx"),
        "tile": TileNamespace(),
        **{kind: FORMATS[kind] for kind in FORMATS},
    }
    return {NAMES[kind].spelling: value for kind, value in made.items()}
