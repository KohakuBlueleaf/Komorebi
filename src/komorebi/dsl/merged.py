"""Classes merged into ONE program in the kernel form: `km.merge` of made class
programs, or `class Layer(Attention, Linear): pass`. Each handler keeps its own class's
constants, plan and tensors; the handlers are numbered over the classes and compiled
together (compiler.compile_merged); a handler is the unit that may be an overlay."""

import copy
import inspect

from komorebi.dsl import compiler, first_use
from komorebi.dsl import program as made
from komorebi.dsl.compile.records import Compiled, Refused
from komorebi.dsl.front import front
from komorebi.dsl.front.front import ClassHandler, HandlerMethod
from komorebi.dsl.front.names import spelling
from komorebi.dsl.front.nodes import prune, render
from komorebi.dsl.thread import class_thread
from komorebi.errors import KernelSizeError, ProgramError
from komorebi.program import description as km_program
from komorebi.program.description import Handler
from komorebi.program.overlay import OverlayError

USER_MAX = 239  # KM_HANDLER_USER_MAX


class Source:
    """One class of a merged program: its name, its class, what it is made with
    (komorebi.dsl.program.make's make_given) and its made program, if any."""

    def __init__(self, name, cls, given, program=None):
        self.name, self.cls, self.given, self.program = name, cls, given, program


class MergedClass(km_program.Program):
    """Classes merged into one program: ONE program field, one handler numbering, one
    L1 layout, in the kernel form. `numbers` {merged: (source, own number)}."""

    def handler(self, handler, name=None):
        """The merged handler of `handler`: a name one class alone has (or the merged
        name of a shared one), a number, a source's own handler (`a.split`), or
        (source, name) when two classes share the name."""
        if name is not None:
            return self._handler_of(self._index(handler), name)
        if isinstance(handler, HandlerMethod):
            if handler.owner is type(self):  # the merged class's own (Layer.split)
                return self.handler(handler.name)
            for index, source in enumerate(self.merged_sources):
                owner = handler.owner
                if isinstance(source, type) and owner and issubclass(source, owner):
                    return self._handler_of(index, handler.name)
                if owner and isinstance(source, owner):  # a made program's (km.merge)
                    return self._handler_of(index, handler.name)
            raise ValueError(f"{handler.name}: not a handler of {self.name}'s classes")
        if isinstance(handler, Handler) and handler.program is not self:
            index = self._index(handler.program)
            method = getattr(handler, "method", None)
            if method is not None:
                return self._handler_of(index, method.name)
            for own_name, number in handler.program.handlers.items():
                if number == int(handler):
                    return self._handler_of(index, own_name)
            raise ValueError(f"{handler!r}: no such handler of {handler.program.name}")
        if isinstance(handler, str):
            if handler in self.shared_names:
                owners = [self.classes[i].name for i in self.shared_names[handler]]
                named = [f"{self.classes[i].prefix}_{handler}" for i in
                         self.shared_names[handler]]  # fmt: skip
                raise ValueError(
                    f"{self.name}: handler {handler!r} is in {' and '.join(owners)}; "
                    f"name its source, merged.handler({owners[0]}, {handler!r}), or "
                    f"its merged name ({', '.join(named)})"
                )
            for method in self.methods.values():
                if method.name == handler:
                    return ClassHandler(method.number, self, method)
        number = km_program.Program.handler(self, handler)
        for method in self.methods.values():
            if method.number == number:
                return ClassHandler(number, self, method)
        return number

    def _handler_of(self, index, name):
        method = self.methods.get((index, name))
        if method is None:  # a handler with listed sets: its first set's
            listed = getattr(self, "constant_handlers", {}).get(name, [])
            mine = [m for _, m in listed if m.source_index == index]
            method = mine[0] if mine else None
        if method is None:
            raise ValueError(f"{self.classes[index].name} has no handler {name!r}")
        return ClassHandler(method.number, self, method)

    def _index(self, source):
        for index, given in enumerate(self.merged_sources):
            if given is source:
                return index
        raise ValueError(f"{getattr(source, 'name', source)!r}: not merged here")

    def merged_number(self, method):
        """A class's own handler (Layer.split, Linear.split) as this program's
        number (run._handler)."""
        return int(self.handler(method))

    def report(self):
        """Each class's handlers and their merged numbers."""
        lines = [f"merged program {self.cname}: handler numbers"]
        for (index, own_name), method in self.methods.items():
            source = self.classes[index]
            lines.append(
                f"  {method.number:>3}: {source.name}.{own_name} as {method.name} "
                f"({source.name}'s {self.own_numbers[index, own_name]})"
            )
        return lines

    # The kernels must fit: a handler placed as an overlay, largest first ------------
    def fits_kernels(self, session, arguments):
        """session.load(**arguments), each time the kernels do not fit the kernel
        buffer with the largest resident handler that may move placed as an overlay;
        when none is left, the load's KernelSizeError. A rebuild keeps the defines
        the load put on the build's (an island load's KM_PAGE_LINKS...)."""
        built = getattr(self, "built_defines", {})
        added = {k: v for k, v in self.defines.items() if built.get(k, ()) != v}
        self.fitting = True
        try:
            while True:
                try:
                    return session.load(**arguments)
                except KernelSizeError as error:
                    if not self.spill(error):
                        raise
                except OverlayError as error:
                    if not self.unspill(error):
                        raise
                self.defines = {**self.defines, **added}
        finally:
            self.fitting = False

    def unspill(self, error):
        """A handler the build placed as an overlay whose part the overlay builder
        refuses (`error`): kept resident, the program built again; False for any
        other refusal."""
        for name, method in self.by_name.items():
            placed = method.overlay and not getattr(method, "marked", False)
            if placed and str(error).startswith(f"overlay {name},"):
                self.log(f"  {name} stays resident: {error}")
                method.overlay = False
                self.build()
                return True
        return False

    def spill(self, error):
        """Places the largest resident handler that may move as an overlay (its
        generated C++, the kernels' and its tile functions'); False when none may."""
        sizes = self.code_sizes()
        movable = [
            (size, name)
            for name, size in sizes.items()
            if self.by_name[name].overlay is None
            and name not in self.compiled.broadcast_lines
        ]
        if not movable:
            return False
        size, name = max(movable)
        sizes_text = ", ".join(f"{n} {s:,} B" for n, s in sorted(sizes.items()))
        total = f"{error.sizes.total:,} B" if error.sizes.total else "more"
        self.log(
            f"{self.name}: its kernels take {total} of the "
            f"{error.sizes.limit:,} B kernel buffer; {name}, the largest resident "
            f"handler that may move ({size:,} B of generated C++; {sizes_text}), is "
            "placed as an overlay"
        )
        self.by_name[name].overlay = True
        try:
            self.build()
        except Refused as refused:  # it cannot be an overlay: kept resident
            self.log(f"  {name} stays resident: {refused}")
            self.by_name[name].overlay = False
            self.build()
        return True

    def code_sizes(self):
        """Each resident handler's generated C++ in bytes: its lines in both kernels
        and the bodies of the tile functions only it calls."""
        compiled = self.compiled
        callers = {}
        for call in compiled.tile_calls:
            callers.setdefault(call_function(call), set()).add(call_handler(call))
        sizes = {}
        for name in compiled.handlers:
            if name in compiled.overlays:
                continue
            text = render(prune(compiled.movement.get(name, [])), 0)
            text += render(prune(compiled.compute.get(name, [])), 0)
            size = sum(len(line) + 1 for line in text)
            for cxx, (_, unit) in compiled.units.items():
                if callers.get(cxx) == {name}:
                    size += len(made.emit(unit))
            sizes[name] = size
        return sizes

    def build(self):
        """Compiles the classes as one program and makes this a Program of the files
        (as komorebi.dsl.program.make does for one class)."""
        build(self)


def call_function(call):
    """The C++ name of the tile function a call of compiled.tile_calls names."""
    return call[4]


def call_handler(call):
    return call[0]


# km.merge of made class programs -----------------------------------------------------
def merge_made(programs, name=None, overlay=(), resident=(), session=None, l1=None,
               log=print):  # fmt: skip
    """ONE program of made class programs (km.merge): each handler with its own
    program's constants, plan and tensors."""
    for program in programs:
        if getattr(program, "merged_sources", None) is not None:
            raise ProgramError(
                f"{program.name}: a merged program; merge its sources with the others"
            )
        if not hasattr(program, "make_given"):
            raise ProgramError(f"{program.name}: not a made class program")
        if getattr(program, "first_use", None) is not None:
            raise ProgramError(
                f"{program.name}: made with km.{spelling('first_use')}, whose sets are "
                "compiled on first use against its own program's load; a merged "
                "program builds its handlers once: list the sets "
                f"({spelling('handlers')}=) to merge them"
            )
    sources = [
        Source(program.name, type(program), program.make_given, program)
        for program in programs
    ]
    merged = MergedClass.__new__(MergedClass)
    # the chain's walk when a source was made with it
    merged.chain_wanted = any(getattr(p, "chain_handler", 0) for p in programs)
    first = programs[0].make_given
    start(
        merged,
        sources,
        name or "_".join(program.name for program in programs),
        overlay,
        resident,
        session if session is not None else first["session"],
        l1 if l1 is not None or session is not None else first["l1"],
        first["trace"],
        log,
    )
    return merged


# class Layer(Attention, Linear): pass -------------------------------------------------
def prepare(cls, parents, compute, defines, startup):
    """A class deriving two or more classes with handlers: their handlers merged
    (numbered over the classes, the first keeping its numbers), made by make_class."""
    if compute is not None or defines is not None or startup is not None:
        raise ProgramError(
            f"{cls.__name__}: compute=, defines= and startup= are each merged class's; "
            "give them there"
        )
    own = [
        value for value in vars(cls).values() if isinstance(value, HandlerMethod)
    ] + [value for value in vars(cls).values() if isinstance(value, front.Constant)]
    if own or any(isinstance(v, front.Field) for v in front_fields(cls)):
        raise ProgramError(
            f"{cls.__name__}: a class merging {', '.join(p.__name__ for p in parents)} "
            "adds no handler, constant or field of its own; merge a class that has them"
        )
    cls.merged_parents = tuple(parents)
    cls.__init__ = make_class
    for key, value in vars(MergedClass).items():  # a merged program's methods
        if callable(value) and not key.startswith("__"):
            setattr(cls, key, value)
    threads = [class_thread.declared(parent)[0] for parent in parents]
    threads = [thread for thread in threads if thread is not None]
    if len(threads) == 1:  # program.thread: the Thread made after the merge
        setattr(cls, threads[0].name, threads[0])
    sources = [Source(parent.__name__.lower(), parent, None) for parent in parents]
    methods, shared = numbered(cls.__name__, sources)
    cls.built_handlers = [method.name for method in methods.values()]
    for (_, own_name), method in methods.items():
        if own_name in shared:
            setattr(cls, own_name, Shared(own_name, [p.__name__ for p in parents]))
        else:
            method.owner = cls
            setattr(cls, own_name, method)
    cls.merged_methods = methods
    cls.overlays_built = property(made.overlays_built, set_overlays_built)


def front_fields(cls):
    return list(getattr(cls, "__annotations__", {}).values())


class Shared:
    """A handler name two merged classes have: refused on the class, which names
    both."""

    def __init__(self, name, owners):
        self.name, self.owners = name, owners

    def __get__(self, program, owner=None):
        raise ProgramError(
            f"{owner.__name__ if owner else ''}.{self.name}: a handler of "
            f"{' and '.join(self.owners)}; name its class, "
            f"program.handler({self.owners[0]}, {self.name!r})"
        )


def make_class(self, plan=None, l1=None, depth=4, log=print, rect=None, nocs=None,
               session=None, trace=0, defines=None, handlers=None,
               broadcast_wait=False, guard=True, overlay=(), resident=(),
               chain=False,
               **given):  # fmt: skip
    """Makes a class merging classes: the make's keywords as one class's (a constant
    to each class that declares it; a constant two classes declare differently is
    refused naming both), then the classes compiled as one program."""
    cls = type(self)
    form_refused(cls, given)
    if handlers is not None:
        raise ProgramError(
            f"{cls.__name__}: {spelling('handlers')}= of a merged class is not built "
            "(its handlers take their defaults); make each class with its sets and "
            "merge the made programs (km.merge)"
        )
    parents = cls.merged_parents
    self.__dict__["thread_given"] = (  # the thread one class declares: class_thread
        class_thread.words_given(given),
        class_thread.tensors_given(given),
    )
    declared = {}  # each class keeps its own constants; an override names one
    for parent in parents:
        for key, declaration in parent.constants_declared.items():
            declared.setdefault(key, []).append((parent, declaration))
    for key, value in [*given.items(), *(plan or {}).items()]:
        owners = declared.get(key, [])
        if isinstance(value, int | float) and not owners and key in given:
            raise ProgramError(
                f"{cls.__name__}: {key}={value!r} is not a constant of the classes "
                f"(their constants: {', '.join(declared) or 'none'})"
            )
        if len({repr(declaration) for _, declaration in owners}) > 1:
            raise ProgramError(
                f"{cls.__name__}: {key} is declared "
                + " and ".join(f"{d!r} in {p.__name__}" for p, d in owners)
                + ": each class keeps its own, and one value given for both is "
                "refused; merge made programs (km.merge) to give each its own"
            )
    sources = []
    for parent in parents:
        mine = {
            key: value
            for key, value in given.items()
            if key in parent.constants_declared or not isinstance(value, int | float)
        }
        own_plan = dict(plan or {})
        values, tensors = made.constants_given(parent, own_plan, mine)
        kept = {**parent.defines_given, **made.defines_checked(cls.__name__, defines)}
        sources.append(
            Source(
                parent.__name__.lower(),
                parent,
                {
                    "plan": own_plan, "tensors": tensors, "constants": values,
                    "sets": {}, "depth": depth, "rect": rect, "nocs": nocs,
                    "defines": kept, "broadcast_wait": broadcast_wait,
                    "guard": guard,
                },  # fmt: skip
            )
        )
    self.chain_wanted = chain  # the chain's walk (km.chain methods give it too)
    start(self, sources, cls.__name__, overlay, resident, session, l1, trace, log)


# Both ways ---------------------------------------------------------------------------
def form_refused(cls, given):
    """Refuses the runtime's form (form=, beside=) at a merged class's make: it is
    built in the kernel form; made classes merge in the runtime's form (km.merge)."""
    asked = [key for key in ("form", "beside") if key in given]
    if asked and (given.get("form", "kernel") != "kernel" or "beside" in given):
        names = ", ".join(parent.__name__ for parent in cls.merged_parents)
        raise ProgramError(
            f"{cls.__name__}: {'= and '.join(asked)}= at the make of a class merging "
            f"{names}: it is built as one program in the kernel form; for the "
            'runtime\'s form merge the made classes, each made with form="runtime" '
            f"(km.merge({names.split(', ')[0]}(..., form=\"runtime\"), ...))"
        )


def start(self, sources, name, overlay, resident, session, l1, trace, log):
    """The merged program's numbering, markers and first build."""
    for source in sources:
        source.prefix = source.name.lower().replace("-", "_").replace("+", "_")
    self.classes = sources
    self.merged_sources = [s.program if s.program is not None else s.cls
                           for s in sources]  # fmt: skip
    self.cname = name
    self.session_made, self.l1_made, self.trace_made = session, l1, trace
    self.fitting = False
    self.log = log
    methods, self.shared_names = numbered(name, sources)
    class_methods = getattr(type(self), "merged_methods", None)
    if class_methods is not None:  # copies of the class's: Layer.split numbers alike
        methods = {key: copy.copy(method) for key, method in class_methods.items()}
    self.methods = methods
    self.by_name = {method.name: method for method in methods.values()}
    self.own_numbers = {key: method.own for key, method in methods.items()}
    markers(self, overlay, True)
    markers(self, resident, False)
    self.build()


def own_handler(cls, name):
    """[the class's handler of that name], or [] (a listed set's name)."""
    return [m for m in compiler.handler_methods(cls) if m.name == name]


def numbered(name, sources):
    """({(source, own name): its copy, numbered and named in the merged program},
    {a name several classes have: [their indices]}): the first class keeps its
    numbers, each next one takes the next numbers in its own order; each copy's
    `declared` its own class (its `self.f` looked up there)."""
    owners = {}
    for index, source in enumerate(sources):
        for method in compiler.handler_methods(source.cls):
            owners.setdefault(method.name, []).append(index)
    shared = {key: value for key, value in owners.items() if len(value) > 1}
    methods, top = {}, 0
    for index, source in enumerate(sources):
        own = set_handlers(source)
        for place, method in own:
            number = place if index == 0 else top + place
            copied = copy.copy(method)
            copied.number, copied.source_index = number, index
            copied.declared = source.cls
            if method.name in shared:
                prefix = source.name.lower().replace("-", "_").replace("+", "_")
                copied.name = f"{prefix}_{method.name}"
            methods[index, method.name] = copied
        top = max(top, *(m.number for m in methods.values()))
    if top > USER_MAX:
        raise ProgramError(
            f"{name}: {top} handlers over the classes; a program numbers {USER_MAX} "
            "at most"
        )
    return methods, shared


def set_handlers(source):
    """[(own number, method)] of a class's handlers as its program built them: a
    handler with constant parameters once for each set its make listed (named by the
    values, at the numbers the class's own build gives them) or once at its defaults."""
    sets = (source.given or {}).get("sets") or {}
    methods = [copy.copy(m) for m in compiler.handler_methods(source.cls)]
    built = compiler.handler_variants(Compiled(source.name, ""), methods, sets)
    for number, method in built:
        method.own = number
    return sorted(built, key=lambda pair: pair[0])


def set_lists(self):
    """{a class's handler name: [(its constant values, the merged copy)]} of each
    handler with constant parameters (komorebi.dsl.first_use.constant_handler)."""
    lists, owner = {}, {}
    for (index, _), method in self.methods.items():
        values = getattr(method, "constant_set", None)
        if values is not None and front.constant_parameters(method.function):
            original = method.function.__name__
            if owner.setdefault(original, index) != index:
                first, other = self.classes[owner[original]], self.classes[index]
                names = f"{first.name} and {other.name}"
                raise ProgramError(
                    f"{self.cname}: {original} has constant parameters in {names}; "
                    "a set is picked by the handler's name: rename one of them"
                )
            lists.setdefault(original, []).append((dict(values), method))
    return lists


def markers(self, given, overlay):
    """`overlay=[...]` (True) or `resident=[...]` (False): each a handler (a name, a
    merged name, a class's or a program's handler) or a whole source (its program or
    class); a handler marked both ways is refused."""
    word = "overlay" if overlay else "resident"
    for marker in given:
        if any(marker is s.program or marker is s.cls for s in self.classes):
            index = next(i for i, s in enumerate(self.classes)
                         if marker is s.program or marker is s.cls)  # fmt: skip
            chosen = [m for (i, _), m in self.methods.items() if i == index]
        else:
            try:
                chosen = [self.handler(marker).method]
            except ValueError as error:
                raise ProgramError(
                    f"{self.cname}: {word}= {marker!r}: {error}"
                ) from None
        for method in chosen:
            changed = method.overlay is not None and bool(method.overlay) != overlay
            if changed and getattr(method, "marked", False):
                raise ProgramError(
                    f"{self.cname}: {method.name} is marked both overlay and "
                    "resident"
                )
            method.overlay = overlay
            method.marked = True


def build(self):
    """The classes compiled as one program; the files written; this made a Program
    of them, as komorebi.dsl.program.make makes one class."""
    sources, log = self.classes, self.log
    session, l1, trace = self.session_made, self.l1_made, self.trace_made
    if session is not None:
        l1 = first_use.reserve_of(session)[0].l1()[1] if l1 is None else l1
        room = l1 - made.runtime_bytes(session, trace)
    else:
        room = l1
    parts = [
        compiler.MergedPart(
            source.cls,
            source.given["plan"],
            source.given["tensors"],
            source.given["constants"],
            [
                (method.number, method)
                for (i, _), method in self.methods.items()
                if i == index
            ],
            source.prefix,
        )
        for index, source in enumerate(sources)
    ]
    handlers = {method.name: method for method in self.methods.values()}
    given = {}  # each class's example tensors, for a direct input's landing area
    for source in sources:
        given.update(source.given["tensors"])
    given_room = room  # the buffers get it less a direct input's landing area
    room = made.direct_room(
        type(self), handlers.values(), given, self.chain_wanted, room
    )
    compiled, tensors = compiler.compile_merged(self.cname, parts, room)
    self.compiled = compiled
    compiled.handler_sources = handler_sources(self, compiled)
    walked = made.chain_made(
        self, type(self), compiled, self.chain_wanted, given, handlers
    )
    made.place_handlers(  # a command by the merged copy or by its class's handler
        compiled,
        {  # a listed set's copy alone: its class's handler names several sets
            method.name: [method, *own_handler(sources[index].cls, own)]
            for (index, own), method in self.methods.items()
        },
    )
    defines, rect, nocs = {}, None, None
    for source in sources:
        for key, value in (source.given.get("defines") or {}).items():
            if defines.setdefault(key, value) != value:
                raise ProgramError(
                    f"{self.cname}: define {key} is {defines[key]!r} and {value!r} in "
                    f"{source.name}; one value a merged program"
                )
        for key, kept in (("rect", rect), ("nocs", nocs)):
            given = source.given.get(key)
            if given is not None and kept is not None and given != kept:
                raise ProgramError(
                    f"{self.cname}: {source.name} is made with {key}={given!r} and "
                    f"another class with {kept!r}; one rectangle a merged program"
                )
        rect = source.given.get("rect") if rect is None else rect
        nocs = source.given.get("nocs") if nocs is None else nocs
    waits = {bool(source.given.get("broadcast_wait")) for source in sources}
    if len(waits) > 1:
        raise ProgramError(f"{self.cname}: broadcast_wait differs between the classes")
    wait = waits.pop()
    for source in sources[1:]:
        if source.given.get("guard", True) != sources[0].given.get("guard", True):
            raise ProgramError(
                f"{self.cname}: {sources[0].name} has guard="
                f"{sources[0].given.get('guard', True)} and {source.name} guard="
                f"{source.given.get('guard', True)}: one merged program, one guard"
            )
    guard = sources[0].given.get("guard", True)
    depth = sources[0].given.get("depth", 4)
    plan = {}
    for source in sources:
        plan.update(source.given["plan"])
    keyed = {
        f"{source.prefix}.{key}": value
        for source in sources
        for key, value in source.given["constants"].items()
    }
    keyed["overlays"] = sorted(compiled.overlays)
    if walked:
        keyed[spelling("chain")] = True
    solved = {n: v for n, v in compiled.constants.items() if n not in plan}
    key = made.sources_key(plan, tensors, solved, depth, rect, nocs, wait, keyed, guard)
    base = class_thread.header_text(class_thread.merged_parts(self, compiled, tensors))
    built = made.write(None, compiled, given_room, key, thread=base)
    for line in self.report():
        log(line)
    for line in built.report:
        log(line)
    self.rectangle = {"rect": rect, "nocs": nocs, "args": None, "group": None}
    self.broadcast_lines = compiled.broadcast_lines
    self.reduce_groups = compiled.reduce_groups
    self.reduce_local = compiled.reduce_local
    self.port_handlers = set(compiled.port_handlers)
    self.constant_handlers = set_lists(self)  # each listed set's merged handler
    program_args = made.rectangle_of(self, compiled) if compiled.rectangle else None
    if compiled.word_fields:
        program_args = made.words_of(compiled, tensors, program_args)
    buffers = []
    for buffer in compiled.buffers:
        if buffer.index_b is None:
            buffers.append(
                km_program.Buffer(
                    buffer.index_a, buffer.dtype, buffer.page_bytes, buffer.pages[0]
                )
            )
        else:
            buffers.append(
                km_program.BufferPair(
                    buffer.index_a,
                    buffer.index_b,
                    buffer.dtype,
                    buffer.page_bytes,
                    buffer.pages,
                )
            )
    movement_args = []
    for tensor in compiled.tensors:
        movement_args += tensors[tensor].tensor_args()
    km_program.Program.__init__(
        self,
        name=compiled.name.lower(),
        kernels=built.folder,
        buffers=buffers,
        compile_args={"movement": movement_args} if movement_args else {},
        compute=made.compute_options(compiled),
        args=program_args,
        defines=defines,
        depth=depth,
        handlers=built.folder / made.header_name(compiled),
        broadcast_wait=wait,
        guard=guard,
        port_handlers=self.port_handlers,  # the init would set it back to none
    )
    self.built_defines = dict(self.defines)  # the build's own (fits_kernels)
    self.built = built
    self.built_handlers = list(compiled.handlers)
    for key, method in getattr(type(self), "merged_methods", {}).items():
        mine = self.methods[key]  # the class's handler gives a call's words too
        method.words, method.widths, method.returns, method.placed = (
            mine.words,
            mine.widths,
            mine.returns,
            mine.placed,
        )
        method.paged = mine.paged
        method.bounds = mine.bounds
        method.padding = mine.padding
    for name in compiled.overlays:
        sources_of = {"reader": built.folder / name / "movement.cpp"}
        sources_of["writer"] = sources_of["reader"]
        if f"{name}/compute.cpp" in built.files:
            sources_of["compute"] = built.folder / name / "compute.cpp"
        self.overlay(name, sources_of, defines=self.defines or None)
    if compiled.overlays:
        self.slots()
    class_thread.merged(self, compiled, tensors)  # after the words: its chains'
    if session is not None:
        made.l1_checked(self, session, l1, trace, log)


def handler_sources(self, compiled):
    """handler -> the file of its class's lines (a chain's waits read each handler's
    statements in its own file)."""
    numbers = {number: name for name, number in compiled.handlers.items()}
    return {
        numbers[method.number]: inspect.getsourcefile(method.function)
        for method in self.methods.values()
        if method.number in numbers
    }


def set_overlays_built(self, built):
    """The session's build of the overlays: kept, and where each handler's code
    lives logged, resident ones too."""
    self.__dict__["_overlays_built"] = built
    if built is not None:
        for line in made.code_table(self, built):
            self.log(line)


MergedClass.overlays_built = property(made.overlays_built, set_overlays_built)
MergedClass.thread = class_thread.added  # its header written, its C built
