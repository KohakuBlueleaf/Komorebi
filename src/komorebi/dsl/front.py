"""What a program written as a class names: the handler decorator, fields, formats,
and the calls of a handler's lines. The lines are read by the compiler, never run in
Python: calling one outside a class's handler raises ProgramError.
"""

import importlib
import inspect
import numbers
import operator
import struct
from dataclasses import dataclass

from komorebi import abi
from komorebi.device.tensor import Tensor
from komorebi.dsl.names import NAMES, spelling
from komorebi.dsl.tile import trace
from komorebi.errors import ProgramError
from komorebi.program.description import Handler
from komorebi.program.movement import Pattern


@dataclass(frozen=True)
class Format:
    """A data format of a tile or a field: its dtype name, bytes an item (0: a block
    format, which has a tile's bytes only) and bytes a 32 x 32 tile."""

    dtype: str
    itemsize: int
    tile_bytes: int


FORMATS = {
    "fp32": Format("float32", 4, 4096),
    "bf16": Format("bfloat16", 2, 2048),
    "u32": Format("uint32", 4, 4096),
    "fp16": Format("float16", 2, 2048),
    "bfp8": Format("bfp8", 0, 1088),
    "bfp4": Format("bfp4", 0, 576),
    "bfp2": Format("bfp2", 0, 320),
    "bfp8_b": Format("bfp8_b", 0, 1088),
    "bfp4_b": Format("bfp4_b", 0, 576),
    "bfp2_b": Format("bfp2_b", 0, 320),
}


@dataclass(frozen=True)
class Field:
    """A field resident in each core's L1: its format and shape (ints, or symbols the
    build solves); the annotation of `km.L1[format, *shape]`."""

    format: Format
    shape: tuple


class Constant:
    """`K_TILES = km.constant(128)` in a class body: a number fixed when the program
    is built, its default overridden at the make; on the class the declaration, on
    a made program its value (constant_values). Arithmetic makes a Derived one."""

    def __init__(self, default):
        if isinstance(default, tuple):
            raise ProgramError(
                f"km.{spelling('constant')}({default!r}): a tuple is not a constant: "
                "the compiler emits no table of constants; declare each number"
            )
        if not isinstance(default, int | float):  # bool is an int
            raise ProgramError(
                f"km.{spelling('constant')}({default!r}): an int, a float or a bool"
            )
        self.default = default
        self.kind = type(default)
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, program, owner=None):
        if program is None:
            return self
        values = program.__dict__.get("constant_values")
        if values is None or self.name not in values:
            raise ProgramError(
                f"{type(program).__name__}.{self.name}: a constant's value is the "
                "made program's; make the program first"
            )
        return values[self.name]

    def __set__(self, program, value):
        raise ProgramError(
            f"{type(program).__name__}.{self.name}: a constant is fixed when the "
            f"program is made; give {self.name}={value!r} to the make"
        )

    def value(self, values):
        """This declaration's value among a program's `values` by name."""
        return values[self.name]

    def __repr__(self):
        return f"km.{spelling('constant')}({self.default!r}) {self.name}"

    def derived(self, op, other, mirrored=False):
        if not isinstance(other, Constant | int | float) or isinstance(other, bool):
            return NotImplemented
        return Derived(op, other, self) if mirrored else Derived(op, self, other)

    def __add__(self, other):
        return self.derived(operator.add, other)

    def __radd__(self, other):
        return self.derived(operator.add, other, True)

    def __sub__(self, other):
        return self.derived(operator.sub, other)

    def __rsub__(self, other):
        return self.derived(operator.sub, other, True)

    def __mul__(self, other):
        return self.derived(operator.mul, other)

    def __rmul__(self, other):
        return self.derived(operator.mul, other, True)

    def __floordiv__(self, other):
        return self.derived(operator.floordiv, other)

    def __rfloordiv__(self, other):
        return self.derived(operator.floordiv, other, True)

    def __mod__(self, other):
        return self.derived(operator.mod, other)

    def __rmod__(self, other):
        return self.derived(operator.mod, other, True)

    def __truediv__(self, other):
        return self.derived(operator.truediv, other)

    def __rtruediv__(self, other):
        return self.derived(operator.truediv, other, True)

    def __neg__(self):
        return Derived(operator.neg, self, None)


class Derived(Constant):
    """`W_BLOCK = CHUNK * WIDTH` in a class body: a constant of others, evaluated
    from each made program's values (a subclass's override of WIDTH included)."""

    def __init__(self, op, left, right):
        self.op, self.left, self.right = op, left, right
        self.default = self.kind = None
        self.name = None

    def value(self, values):
        def of(side):
            return side.value(values) if isinstance(side, Constant) else side

        if self.right is None:
            return self.op(of(self.left))
        return self.op(of(self.left), of(self.right))

    def __repr__(self):
        return f"{self.name or 'a derived constant'} (of the class's constants)"


class Symbol(Constant):
    """`K = km.symbol()` (or `km.constant()`) in a class body: a size of a field's
    shape the plan gives or the build solves; its name is the attribute's."""

    def __init__(self):
        self.default = None
        self.kind = int
        self.name = None

    def __repr__(self):
        return f"km.{spelling('symbol')}() {self.name}"


def symbol():
    return Symbol()


class ConstantParameter:
    """`block: km.constant[int] = 8`: a handler's parameter fixed when the program is
    built, not a command word; each listed set of values a handler of its own."""

    def __init__(self, kind):
        self.kind = kind


class ConstantSpelling:
    """`km.constant`: called, a constant of the class (no default: a symbol);
    subscripted by int, float or bool, a handler's constant parameter."""

    def __call__(self, *default):
        if len(default) > 1:
            raise ProgramError(f"km.{spelling('constant')}(default): one default")
        return Constant(default[0]) if default else Symbol()

    def __getitem__(self, kind):
        if kind not in (int, float, bool):
            raise ProgramError(
                f"km.{spelling('constant')}[{getattr(kind, '__name__', kind)}]: a "
                "constant parameter is of int, float or bool"
            )
        return ConstantParameter(kind)


class FirstUse:
    """`km.ANY` in a make's handlers=: any set of the handler's constant parameters
    is taken at submit, a set not built compiled then (komorebi.dsl.first_use);
    `km.ANY(p=n)`: the load's buffers also sized for the set with p=n (`reserve`)."""

    def __init__(self, reserve=None):
        self.reserve = dict(reserve or {})

    def __call__(self, **reserve):
        return FirstUse(reserve)

    def __repr__(self):
        given = ", ".join(f"{k}={v!r}" for k, v in self.reserve.items())
        return f"km.{spelling('first_use')}" + (f"({given})" if given else "")


def constants_of(cls):
    """The constants declared on `cls` and its parents by name, a subclass's over
    its parent's; a name a parent declared rebound to a plain value is refused."""
    declared = {}
    for klass in reversed(cls.__mro__):
        for name, value in vars(klass).items():
            if isinstance(value, Constant):
                declared[name] = value
            elif name in declared:
                raise ProgramError(
                    f"{klass.__name__}.{name}: a constant of a parent class; declare "
                    f"it {name} = km.{spelling('constant')}({value!r})"
                )
    return declared


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


# compute_kernel_hw_startup's source orders: how its two inputs map on SrcA and SrcB.
STARTUP_ORDERS = {"regular": "SrcOrder::Regular", "reverse": "SrcOrder::Reverse"}


class Startup:
    """`startup=km.Startup("a", "b", "out", order="reverse")` of a class: compute's
    start-up (compute_kernel_hw_startup) by those fields' buffers, two inputs and
    an output (or one input and the output), in the source order given."""

    def __init__(self, *fields, order="regular"):
        if not 2 <= len(fields) <= 3 or not all(isinstance(f, str) for f in fields):
            raise ProgramError(
                f"km.{spelling('startup')}(...): the names of two input fields and an "
                "output field (or one input and the output)"
            )
        if order not in STARTUP_ORDERS:
            raise ProgramError(
                f"km.{spelling('startup')}(order={order!r}): one of "
                f"{', '.join(map(repr, STARTUP_ORDERS))}"
            )
        if len(fields) == 2 and order != "regular":
            raise ProgramError(
                f"km.{spelling('startup')}: a source order is of two inputs; name "
                "both"
            )
        self.fields, self.order = fields, order


class L1:
    """`km.L1[format, *shape]`: the annotation of a field resident in the L1, each
    size an int or a km.symbol() of the class."""

    def __class_getitem__(cls, key):
        key = key if isinstance(key, tuple) else (key,)
        format_, *shape = key
        if not isinstance(format_, Format):
            raise ProgramError(
                f"km.{spelling('field')}[format, *shape]: the format first, km.fp32, "
                "km.bf16, km.u32, km.fp16 or a block format (km.bfp8_b, ...)"
            )
        if not shape or not all(
            isinstance(n, Constant) or (isinstance(n, int) and not isinstance(n, bool))
            for n in shape
        ):
            raise ProgramError(
                f"a field's sizes are ints and constants of the class (K = "
                f"km.{spelling('constant')}(n) or km.{spelling('symbol')}() in its "
                f"body): got {tuple(shape)!r}"
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


class Entry:
    """`class Job(km.Entry)` with int annotations: an entry of a page's table of words
    (`pages.entries(Job, ...)`), its named words 0, 1, ... in order."""

    words = {}  # name -> its word in the entry

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        names = list(inspect.get_annotations(cls))
        if not names or not all(
            inspect.get_annotations(cls)[name] is int for name in names
        ):
            raise ProgramError(
                f"{cls.__name__}: an entry's words are ints, one at least"
            )
        cls.words = {name: word for word, name in enumerate(names)}

    def __init__(self, **words):
        raise ProgramError(
            f"km.{spelling('entry')} is read by the compiler from a handler's lines; "
            "Python never makes one"
        )


class ClassHandler(Handler):
    """A handler of a program made from a class: a Handler (its number and program)
    whose `args(...)` gives the command words of a call."""

    def __new__(cls, number, program, method):
        made = super().__new__(cls, number, program)
        made.method = method
        return made

    def args(self, *values, **named):
        """The command words of a call with these arguments."""
        made = getattr(self.program, "up_to_given", {}).get(self.method.name, {})
        if made != up_to_of(self.method.function):
            raise ProgramError(
                f"{type(self.program).__name__}.{self.method.name}: made with "
                f"km.{spelling('up_to')} {made or 'none'}, and the class was made "
                f"since with {up_to_of(self.method.function) or 'none'}; a handler's "
                "words are its last make's: make this program again"
            )
        return self.method.args(*values, **named)

    def __call__(self, *values, **named):
        """A call of the handler in a chain (`run.chain([program.scale(x, y)])`)."""
        return HandlerCall(self, values, named)


class Words(list):
    """A call's command words, a list as before, that also carry the constant values
    the call stated (`constants`, by name) and the handler's name. A slice, a copy, or
    a plain list added on either side keeps them; words of two handlers or two sets
    joined, or repeated, are refused naming the handler."""

    def __init__(self, words, constants, handler=None):
        super().__init__(words)
        self.constants = dict(constants)
        self.handler = handler

    def _kept(self, words):
        return Words(words, self.constants, self.handler)

    def _stated(self):
        sets = ", ".join(f"{k}={v}" for k, v in self.constants.items())
        return f"{self.handler or 'a handler'}'s words for {sets or 'its defaults'}"

    def _joined(self, other):
        """Refuses `other` beside these when either states a set the other does not."""
        if (
            isinstance(other, Words)
            and (self.constants or other.constants)
            and (other.handler, other.constants) != (self.handler, self.constants)
        ):
            raise ProgramError(
                f"{self._stated()} joined with {other._stated()}: one command runs "
                "one handler's set; call args again for the words you mean"
            )

    def __getitem__(self, index):
        got = super().__getitem__(index)
        return self._kept(got) if isinstance(index, slice) else got

    def copy(self):
        return self._kept(self)

    def __add__(self, other):
        self._joined(other)
        return self._kept([*self, *other])

    def __radd__(self, other):
        self._joined(other)
        return self._kept([*other, *self])

    def __iadd__(self, other):
        self._joined(other)
        return super().__iadd__(other)

    def __mul__(self, count):
        if self.constants:
            raise ProgramError(
                f"{self._stated()} repeated: one command's words state one call"
            )
        return list(self) * count

    __rmul__ = __mul__

    def __imul__(self, count):
        self * count  # refuses a set's words
        return super().__imul__(count)


class HandlerMethod:
    """A method decorated with `km.handler`: on the class, its number once the class
    is prepared (an int) and `args(...)`; on a made program, a ClassHandler."""

    def __init__(self, function, overlay=None, stamps=None, direct=(), in_place=False):
        self.function = function
        # its code is an overlay (loaded into a slot, called); False: resident, kept
        # so in a merged program too; None: resident, which a merged build may move
        self.overlay = overlay
        self.stamps = stamps  # the tensor parameter its load and call cycles go to
        self.direct = tuple(direct)  # its inputs a chain's call may send direct
        self.in_place = in_place  # a direct input lands in its stream's own halves
        self.name = function.__name__
        self.number = None
        self.words = None  # [(parameter, kind)] in word order, set by the compiler
        self.widths = {}  # tensor parameter -> the tile width of its rectangles
        # tensor parameter -> (turned, the product's place) of each product reading it
        # as W: set when a program of the handler is made (dsl.program.place_handlers)
        self.placed = {}
        self.owner = None  # the class it is a handler of
        self.returns = False  # it returns a word: its commands ask for the result
        # its constant parameters' values this handler is built for (km.constant[int])
        self.constant_set = {}

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

    def __call__(self, *values, **named):
        """A call of the handler in a chain (`run.chain([Class.scale(x, y)])`): its
        arguments as a submit's, `cores=(first, count)` its group when not the
        chain's; outside a chain's list it runs nothing."""
        return HandlerCall(self, values, named)

    def args(self, *values, **named):
        """The command words of a call: a tensor gives its reference word (and its
        page count, if the handler counts its pages), an int itself, a pattern its
        word; a constant parameter's value, at its place, is no word."""
        if self.words is None:
            raise ProgramError(f"{self.name}: its program is not made yet")
        given = dict(zip(self.places(), values))
        if rect_given(self.function) is not None and len(values) > len(given):
            given["rect"] = values[len(given)]  # the rectangle after the parameters
        if len(values) > len(given):
            raise ProgramError(f"{self.name}: {len(values)} arguments, too many")
        given.update(named)
        for constant in constant_parameters(self.function):
            given.pop(constant, None)
        sized = sized_parameters(self.function)
        words = []
        for parameter, kind in self.words:
            value = given.get(parameter)
            if value is None or isinstance(value, SizedHook):
                raise ProgramError(f"{self.name}: no argument {parameter!r}")
            if parameter in sized and kind == "int" and not isinstance(value, AtSubmit):
                sized_checked(self.name, parameter, value, sized[parameter])
            if kind == "columns":
                columns_checked(self.name, parameter, value, sized[parameter])
            if kind == "rect words":  # the rectangle's last three, given with it
                continue
            if kind == "rect":  # its four words: x0, y0, columns, rows
                words += rect_checked(self.name, parameter, value, sized[parameter])
                continue
            width = self.widths.get(parameter)
            shape = getattr(value, "shape", None)
            if width is not None and shape is not None and shape[-1] // 32 != width:
                raise ProgramError(
                    f"{self.name}: {parameter} is {shape[-1] // 32} tiles wide, and "
                    f"the handler's rectangles of {parameter} were built for {width} "
                    "(the example tensor's): make the program with a tensor of this "
                    "width"
                )
            self.placed_check(parameter, value)
            words.append(word_of(self.name, parameter, kind, value))
        owner = self.owner.__name__ if self.owner else "its class"
        return Words(words, self.constants_given(values, named), f"{owner}.{self.name}")

    def placed_check(self, parameter, value):
        """Refuses a command's tensor that a product reads as its W stored otherwise
        than the product's placement takes it (transposed tiles: turned)."""
        placed = self.placed.get(parameter)
        if placed is None or not isinstance(value, Tensor):
            return
        turned, where = placed
        if value.transposed_tiles != turned:
            stored = "stored" if value.transposed_tiles else "not stored"
            raise ProgramError(
                f"{self.name}: tensor {parameter} is {stored} with its tiles "
                f"transposed, and {where} reads it as W of a "
                f"{'turned' if turned else 'plain'} product; write it with "
                f"session.tensor(..., transposed_tiles={turned})"
            )

    def places(self):
        """The names of the parameters after self in order, constant ones too."""
        return list(inspect.signature(self.function).parameters)[1:]

    def constants_given(self, values, named=None):
        """The constant parameters' values a call gives at their places (or by
        name), by name; one left out takes its default."""
        constants = constant_parameters(self.function)
        given = {
            name: value
            for name, value in zip(self.places(), values)
            if name in constants
        }
        given.update({k: v for k, v in (named or {}).items() if k in constants})
        return given

    def parameters(self):
        """The (name, kind) of each parameter, in order, without page counts."""
        return [
            (p, kind)
            for p, kind in self.words
            if kind not in ("pages", "bank", "columns")
        ]


class Region:
    """km.Region, a handler parameter's type: an X280 thread's region, its bank address
    and bank two command words. `Run.submit` takes it by the region's name, or as
    (bank, bank address), what a run's `threads[c].regions.address(name)` gives."""

    def __init__(self, bank, address, island=0):
        self.bank, self.address, self.island = int(bank), int(address), int(island)


class ThreadRegion:
    """`groups = km.region(4096)` in a class body: a region of the class's X280
    thread, its bytes an int or of the class's constants; ids in body order."""

    def __init__(self, size):
        if isinstance(size, bool) or not isinstance(size, int | Constant):
            raise ProgramError(
                f"km.{spelling('thread_region')}({size!r}): its bytes, an int or of "
                f"the class's constants (km.{spelling('constant')})"
            )
        self.size, self.name = size, None

    def __set_name__(self, owner, name):
        self.name = name

    def bytes(self, values):
        """Its bytes among a made program's constant `values`."""
        return self.size.value(values) if isinstance(self.size, Constant) else self.size


class ClassThread:
    """`thread = km.x280_thread("t.c", cluster=1)` in a class body: the class's X280
    thread, C files beside the class's source; on a made program, its Thread."""

    def __init__(self, sources, options, line):
        self.chains = dict(options.pop("chains", None) or {})  # the header's chains
        self.sources, self.options, self.line = sources, options, line
        self.cluster = options.get("cluster", 1)
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, program, owner=None):
        if program is None:
            return self
        return program.threads[self.cluster]


THREAD_OPTIONS = ("cluster", "deadline", "includes", "defines", "prefetch",
                  "host_fallback", "chains")  # fmt: skip


def thread(*sources, **options):
    """The spelling of a class's thread: its C files, then Program.thread's options
    but its regions (km.region in the body) and words (thread_words= at the make)."""
    unknown = sorted(set(options) - set(THREAD_OPTIONS))
    if not sources or unknown:
        raise ProgramError(
            f"km.{spelling('class_thread')}(*sources, {', '.join(THREAD_OPTIONS)}): "
            + (f"{', '.join(unknown)} is not one of them" if unknown else "no source")
        )
    return ClassThread(sources, options, inspect.currentframe().f_back.f_lineno)


def region_words(handler, parameter, value):
    """A region argument's (bank address, bank)."""
    if isinstance(value, Region):
        return value.address, value.bank
    if isinstance(value, tuple) and len(value) == 2:
        return int(value[1]), int(value[0])
    raise ProgramError(
        f"{handler}: {parameter} is a region: give its name (Run.submit) or "
        f"(bank, address), got {value!r}"
    )


class SizedHook:
    """The internal hook of a size given at each submit (a handler parameter's
    default, `block: int = _sized(8)`): its most sizes the buffers. No public name
    yet; the spelling is the owner's to choose."""

    def __init__(self, most):
        rect = (  # a rectangle's (columns, rows) at most
            isinstance(most, tuple)
            and len(most) == 2
            and all(isinstance(n, int) and n >= 1 for n in most)
        )
        if not rect and (not isinstance(most, int) or most < 1):
            raise ProgramError(
                f"a size given at each submit: most {most!r}, an int or (columns, rows)"
            )
        self.most = most

    def __repr__(self):
        return f"a size given at each submit, at most {self.most}"


def _sized(most):
    """The one internal hook: a handler parameter given at each submit, at most
    `most` (see SizedHook)."""
    return SizedHook(most)


class UpTo:
    """`km.UPTO(block=8, x=64, rect=(4, 4))` among a handler's sets: one build serves
    every value up to the most. A key names a constant int parameter (a word), a
    tensor parameter (its tile columns) or, with no such parameter, `rect` (the
    rectangle, an argument after the parameters). It sets SizedHook on the handler."""

    def __init__(self, mosts=None):
        self.mosts = dict(mosts or {})

    def __call__(self, **mosts):
        return UpTo(mosts)

    def __repr__(self):
        given = ", ".join(f"{k}={v!r}" for k, v in self.mosts.items())
        return f"km.{spelling('up_to')}" + (f"({given})" if given else "")


UP_TO = "km_up_to"  # a handler function's attribute: {name: SizedHook} from km.UPTO


def up_to_set(cls, method, up_to):
    """Sets a handler's sizes given at each submit from the make's km.UPTO (None:
    none), each checked against the handler's parameters; a class made before with
    other ones is refused (the handler's words are its class's)."""
    label = f"{cls.__name__}.{method.name}"
    word = f"km.{spelling('up_to')}"
    hooks = {}
    signature = inspect.signature(method.function).parameters
    for name, most in (up_to.mosts if up_to else {}).items():
        parameter = signature.get(name)
        if parameter is None and name != "rect":
            raise ProgramError(
                f"{label}: {word}({name}=...) names no parameter of it, and is not "
                f"rect (its parameters: {', '.join(list(signature)[1:])})"
            )
        annotation = None if parameter is None else parameter.annotation
        if parameter is None:
            fits = isinstance(most, tuple) and len(most) == 2
            what = "the rectangle's most is (columns, rows)"
        elif isinstance(annotation, ConstantParameter) and annotation.kind is int:
            fits = isinstance(most, int) and not isinstance(most, bool)
            what = f"{name} is a constant int: its most is an int"
        elif annotation is Tensor or getattr(annotation, "__name__", "") == "Tensor":
            fits = isinstance(most, int) and not isinstance(most, bool)
            what = f"{name} is a tensor: its most is its tile columns, an int"
        else:
            raise ProgramError(
                f"{label}: {word}({name}=...): {name} is neither a constant int "
                "parameter nor a tensor; a size given at each submit is one of those "
                "or the rectangle (rect)"
            )
        if not fits:
            raise ProgramError(f"{label}: {word}({name}={most!r}): {what}")
        hooks[name] = _sized(most)
    method.function.__dict__[UP_TO] = hooks  # the last make's, as its words are
    return {k: h.most for k, h in hooks.items()}


def up_to_of(function):
    """A handler's sizes given at each submit, its last make's: {name: most}."""
    return {k: h.most for k, h in getattr(function, UP_TO, {}).items()}


def sized_parameters(function):
    """A handler's parameters whose size is given at each submit by name: their
    most (km.UPTO at the make, or the internal hook as a parameter's default)."""
    found = {
        name: parameter.default.most
        for name, parameter in inspect.signature(function).parameters.items()
        if isinstance(parameter.default, SizedHook)
    }
    found.update(
        {k: h.most for k, h in getattr(function, UP_TO, {}).items() if k not in found}
    )
    return found


def rect_given(function):
    """The most of a rectangle km.UPTO gives a handler with no parameter of that name
    (an argument after its parameters), else None."""
    hooks = getattr(function, UP_TO, {})
    if "rect" in hooks and "rect" not in inspect.signature(function).parameters:
        return hooks["rect"].most
    return None


def sized_checked(handler, parameter, value, most):
    """Refuses a size given at submit that does not divide its most (its buffers'
    pages are whole blocks), by name."""
    if not isinstance(value, int) or not 1 <= value <= most or most % value:
        raise ProgramError(
            f"{handler}: {parameter}={value!r}; a size given at each submit, at most "
            f"{most}: an int that divides {most} (the buffers hold whole blocks)"
        )


def rect_checked(handler, parameter, value, most):
    """A rectangle given at each submit, (x0, y0, columns, rows) within its most
    (columns, rows): its four words; else refused by name."""
    if (
        not isinstance(value, tuple | list)
        or len(value) != 4
        or not all(isinstance(n, int) and n >= 0 for n in value)
        or not 1 <= value[2] <= most[0]
        or not 1 <= value[3] <= most[1]
    ):
        raise ProgramError(
            f"{handler}: {parameter}={value!r}; a rectangle given at each submit is "
            f"(x0, y0, columns, rows), at most {most[0]} columns and {most[1]} rows"
        )
    return [int(n) for n in value]


def columns_checked(handler, parameter, value, most):
    """Refuses a tensor whose width, given at each submit, is past its most (or not
    of whole tiles), by name."""
    columns = (getattr(value, "shape", None) or (0,))[-1]
    if columns % 32 or not 1 <= columns // 32 <= most:
        raise ProgramError(
            f"{handler}: {parameter} is {columns} columns wide; its width is given at "
            f"each submit, at most {most} tiles of 32 columns"
        )


def constant_parameters(function):
    """A handler's constant parameters (`p: km.constant[int] = n`) by name: (their
    type, their default; None: no default)."""
    found = {}
    given = getattr(function, UP_TO, {})  # km.UPTO: a command word, no constant
    for name, parameter in inspect.signature(function).parameters.items():
        if name in given:
            continue
        if isinstance(parameter.annotation, ConstantParameter):
            default = parameter.default
            found[name] = (
                parameter.annotation.kind,
                None if default is inspect.Parameter.empty else default,
            )
    return found


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
    if kind == "columns":  # its tile columns: a width given at each submit
        return int(value.shape[-1]) // 32
    if kind in ("region", "bank"):
        address, bank = region_words(handler, parameter, value)
        return (address if kind == "region" else bank) & abi.M32
    if kind == "pattern":
        pattern = value if isinstance(value, Pattern) else Pattern(value)
        return pattern.word
    if kind == "float":
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ProgramError(f"{handler}: {parameter} is a float, got {value!r}")
        try:
            return struct.unpack("<I", struct.pack("<f", float(value)))[0]
        except OverflowError:
            raise ProgramError(
                f"{handler}: {parameter} = {value!r}: past float32's range"
            ) from None
    if isinstance(value, bool) or not hasattr(value, "__index__"):
        raise ProgramError(f"{handler}: {parameter} is an int, got {value!r}")
    value = operator.index(value)
    if not 0 <= value <= abi.M32:
        raise ProgramError(f"{handler}: {parameter} = {value}: a 32-bit word")
    return value


def handler(function=None, *, overlay=None, stamps=None, direct=(), in_place=False):
    """Makes a method of a class deriving km.Program a handler; `overlay=True`: built
    apart as an overlay, loaded into a slot when a command needs it (False: resident,
    also when merged); `stamps="t"`: each mover's cycles to tensor t; `direct`: the
    inputs a chain's call may send into its cores' L1 (`in_place`: its stream's)."""
    if isinstance(direct, str):
        direct = (direct,)
    if function is None:
        return lambda function: HandlerMethod(
            function, overlay, stamps, direct, in_place
        )
    return HandlerMethod(function, overlay, stamps, direct, in_place)


class HandlerCall:
    """A call of a class's handler in a chain: the handler (a class's, or a made
    program's), its arguments as a submit's (`km.past(t)`, `km.direct(t)` marking a
    tensor) and `cores=(first, count)` when its group is not the chain's."""

    def __init__(self, handler, values, named):
        self.handler = handler
        self.method = handler if isinstance(handler, HandlerMethod) else handler.method
        named = dict(named)
        self.cores = None
        if "cores" in named and "cores" not in self.method.places():
            self.cores = tuple(named.pop("cores"))
        self.values, self.named = tuple(values), named

    def __repr__(self):
        return f"{self.method.name}({len(self.values)} arguments)"


class Marked:
    """A tensor argument of a chain's call marked `km.past(t)` (written by a call
    further back, proven complete) or `km.direct(t)` (sent into the cores' L1)."""

    def __init__(self, tensor, how):
        self.tensor, self.how = tensor, how

    def __getattr__(self, name):
        return getattr(self.__dict__["tensor"], name)


def past(tensor):
    """km.past(t) at a chain's call: t was written by a call further back than the
    one before; the build proves each tile complete where it is read, or refuses."""
    return Marked(tensor, "past")


def direct(tensor):
    """km.direct(t) at a chain's call: the call before's roots send t straight into
    this call's cores' L1 (the handler declares t in km.handler(direct=))."""
    return Marked(tensor, "direct")


class AtSubmit:
    """km.at_submit("name") at an int parameter of a chain's call: the word is not
    in the chain's records but given at each run.submit(chain, name=value)."""

    def __init__(self, name):
        if not isinstance(name, str) or not name.isidentifier():
            raise ProgramError(
                f"km.{spelling('at_submit')}({name!r}): a word's name, an identifier"
            )
        self.name = name

    def __repr__(self):
        return f"km.{spelling('at_submit')}({self.name!r})"


def at_submit(name):
    """km.at_submit("position") at a chain's call: the handler's int word given at
    each submit of the chain (run.submit(chain, position=n)), the records unchanged."""
    return AtSubmit(name)


class ChainMethod:
    """A method decorated with `km.chain`: calls of the class's handlers in order,
    read when the class is made; called on a program with its arguments, the same
    list of calls the host form gives (`run.chain(program.layer(x, y))`)."""

    def __init__(self, function):
        self.function = function
        self.name = function.__name__
        self.owner = None

    def __set_name__(self, owner, name):
        self.name, self.owner = name, owner

    def __get__(self, program, owner=None):
        if program is None:
            return self
        return lambda *values, **named: chain_calls_module().expand(
            program, self, values, named
        )


def chain_calls_module():
    """komorebi.dsl.chain_calls (it imports the compiler, which imports this)."""
    return importlib.import_module("komorebi.dsl.chain_calls")


def chain(function):
    """Makes a method of a class deriving km.Program a chain: its lines calls of the
    class's handlers in order (a loop of a constant count unrolled). Spelled
    `km.chain`, the framework's module komorebi.chain called."""
    return ChainMethod(function)


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


def placement(side):
    """km.srcb / km.srca: marks a product's operand for that source register; read by
    the compiler from the source, and its operand itself while a tile function is
    traced (the compiler has refused a marker that turns a traced product)."""

    def call(*args, **kwargs):
        if trace._STACK and len(args) == 1 and not kwargs:
            return args[0]
        raise ProgramError(
            f"km.{spelling(side)} marks an operand of a tile function `return a @ b` "
            "that a handler's product line adds; the compiler reads it, Python never "
            "runs it"
        )

    call.__name__ = spelling(side)
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
        "past": past,
        "direct": direct,
        "at_submit": at_submit,
        "tile_function": tile_function,
        "message": Message,
        "entry": Entry,
        "field": L1,
        "symbol": symbol,
        "constant": ConstantSpelling(),
        "first_use": FirstUse(),
        "up_to": UpTo(),
        "own": own,
        "tile_value": Tile,
        "startup": Startup,
        "tensor": Tensor,
        "region": Region,
        "class_thread": thread,
        "thread_region": ThreadRegion,
        "stream": line_only("stream"),
        "reduce": line_only("reduce"),
        "all_reduce": line_only("all_reduce"),
        "broadcast": line_only("broadcast"),
        "cxx": line_only("cxx"),
        "srcb": placement("srcb"),
        "srca": placement("srca"),
        "tile": TileNamespace(),
        **{kind: FORMATS[kind] for kind in FORMATS},
        **{kind: tile_math(kind) for kind in NAMES if kind.startswith("math_")},
    }
    return {NAMES[kind].spelling: value for kind, value in made.items()}
