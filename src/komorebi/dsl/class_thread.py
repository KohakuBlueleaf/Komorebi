"""The X280 thread of a program written as a class. The class declares it in its body
(km.x280_thread, km.region), or it is added after the make (program.thread). Either
way its C is built with the class's thread header, whose functions fill each
handler's command, so the C and the class are checked against each other.
"""

import hashlib
import importlib
import inspect
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from komorebi import _env, abi, build, native
from komorebi import program as km_program
from komorebi.dsl import front
from komorebi.dsl.names import spelling
from komorebi.errors import ProgramError
from komorebi.thread import BANK_OF_CLUSTER, layout

# C words a parameter's name may be: its argument takes a "_" after it.
C_WORDS = frozenset(
    ("auto", "break", "case", "char", "const", "continue", "default", "do", "double",
     "else", "enum", "extern", "float", "for", "goto", "if", "inline", "int", "long",
     "register", "restrict", "return", "short", "signed", "sizeof", "static",
     "struct", "switch", "typedef", "union", "unsigned", "void", "volatile", "while")
)  # fmt: skip


@dataclass
class Handler:
    """A built handler as the thread's header gives it: its name in the program, its
    C function, its words, its tensor parameters' tensors among the load's words,
    its line and whether its command asks for the leader's result."""

    name: str
    function: str
    words: list
    tensors: dict
    line: int
    result: bool


@dataclass
class Parts:
    """What a program's thread header is made of: the Compiled, the program's tensors
    in word order, its handlers, its constants by define, the handlers given km.ANY."""

    compiled: object
    tensors: dict
    handlers: list
    constants: dict
    opted: tuple


def declared(cls):
    """(the class's km.x280_thread or None, its km.region declarations by name), a
    subclass's over its parents'; two threads refused."""
    threads, regions = {}, {}
    for klass in reversed(cls.__mro__):
        for name, value in vars(klass).items():
            if isinstance(value, front.ClassThread):
                threads[name] = value
            elif isinstance(value, front.ThreadRegion):
                regions[name] = value
    if len(threads) > 1:
        raise ProgramError(
            f"{cls.__name__}: {len(threads)} threads ({', '.join(threads)}); a class "
            f"declares one km.{spelling('class_thread')}"
        )
    if regions and not threads:
        raise ProgramError(
            f"{cls.__name__}: km.{spelling('thread_region')} declares a region of the "
            f"class's thread, and it declares none: add `thread = "
            f'km.{spelling("class_thread")}("thread.c", cluster=1)`'
        )
    return next(iter(threads.values()), None), regions


def prepared(cls):
    """At the class's definition: its declarations checked; without a declared thread
    named `thread`, program.thread(...) adds one and writes its header (added)."""
    declared(cls)
    if not isinstance(inspect.getattr_static(cls, "thread", None), front.ClassThread):
        cls.thread = added


def words_given(tensors):
    """The make's thread_words= taken from its other keywords (None: not given)."""
    return tensors.pop(spelling("thread_words"), None)


# The make -----------------------------------------------------------------------------
def made(program, cls, compiled, tensors, thread_words, opted):
    """At the make: the thread header written without a thread; with a declared one,
    the thread made (its regions sized by the program's constants, its words, then
    the load's), its own header written and its C built."""
    thread, regions = declared(cls)
    word = spelling("thread_words")
    if thread is None and thread_words is not None:
        raise ProgramError(
            f"{cls.__name__}: {word}= gives the words of the class's thread, and it "
            f"declares none (km.{spelling('class_thread')})"
        )
    parts = class_parts(program, compiled, tensors, opted)
    program.__dict__["thread_parts"] = parts
    write_base(program, parts)
    if thread is None:
        return
    for handler in parts.handlers:
        for parameter, kind in handler.words:
            if kind == "region" and parameter not in regions:
                raise ProgramError(
                    f"{cls.__name__}.{handler.name}, line {handler.line}: {parameter} "
                    "is a region parameter, and the class's thread has no region of "
                    f"that name: declare `{parameter} = "
                    f"km.{spelling('thread_region')}(bytes)` in the class"
                )
    values = program.constant_values
    here = Path(compiled.source).resolve().parent
    options = dict(thread.options)
    includes = [here / path for path in options.pop("includes", ())]
    try:
        made_thread = km_program.Program.thread(
            program,
            sources=[here / source for source in thread.sources],
            regions={name: region.bytes(values) for name, region in regions.items()},
            words=list(thread_words or ()),
            includes=[*includes, program.built.folder],
            **options,
        )
    except ValueError as error:
        raise ProgramError(
            f"{cls.__name__}, line {thread.line}: its thread: {error}"
        ) from None
    joined(program, made_thread, parts, f"{cls.__name__}, line {thread.line}")


def added(self, cluster=1, clusters=None, **kwargs):
    """Program.thread of a class program: the thread declared as a hand program's,
    then its header written and its C built as a declared thread's."""
    thread, _ = declared(type(self))
    if thread is not None:
        raise ProgramError(
            f"{type(self).__name__}: the class declares its thread "
            f"(km.{spelling('class_thread')}, line {thread.line}); one thread a class"
        )
    made_threads = km_program.Program.thread(self, cluster, clusters, **kwargs)
    parts = self.__dict__.get("thread_parts")
    if parts is not None:
        for one in made_threads if isinstance(made_threads, list) else [made_threads]:
            joined(self, one, parts, f"{self.name}: program.thread")
    return made_threads


def merged(program, compiled, tensors):
    """A merged program's thread header: the merged numbers, each class's functions
    under its own name; a class declaring a thread refused (add one after)."""
    parts = merged_parts(program, compiled, tensors)
    program.__dict__["thread_parts"] = parts
    write_base(program, parts)


def merged_parts(program, compiled, tensors):
    """The parts of a merged program's thread header (merged)."""
    for source in program.classes:
        if declared(source.cls)[0] is not None:
            raise ProgramError(
                f"{program.cname}: {source.cls.__name__} declares its thread; a merged "
                "program takes one after the merge, program.thread(...)"
            )
    handlers, constants = [], {}
    for (index, own), method in program.methods.items():
        source = program.classes[index]
        given = source.given["tensors"]
        named = {
            parameter: (
                parameter
                if tensors.get(parameter) is given.get(parameter)
                else f"{source.prefix}_{parameter}"
            )
            for parameter, kind in compiled.words[method.name]
            if kind == "tensor"
        }
        handlers.append(
            Handler(
                method.name,
                f"{source.cls.__name__.lower()}_{own}",
                compiled.words[method.name],
                {p: name for p, name in named.items() if name in tensors},
                compiled.lines[method.name],
                method.name in compiled.results,
            )
        )
        for name, value in source.given["constants"].items():
            constants[f"{source.cls.__name__.upper()}_{name.upper()}"] = value
    return Parts(compiled, ordered(compiled, tensors), handlers, constants, ())


def class_parts(program, compiled, tensors, opted):
    prefix = compiled.name.upper()
    handlers = [
        Handler(
            name,
            f"{compiled.name.lower()}_{name}",
            compiled.words[name],
            {
                p: p
                for p, kind in compiled.words[name]
                if kind == "tensor" and p in tensors
            },
            compiled.lines[name],
            name in compiled.results,
        )
        for name in compiled.handlers
    ]
    constants = {
        f"{prefix}_{name.upper()}": value
        for name, value in program.constant_values.items()
    }
    return Parts(
        compiled, ordered(compiled, tensors), handlers, constants, tuple(opted)
    )


def ordered(compiled, tensors):
    return {name: tensors[name] for name in compiled.tensors if name in tensors}


def write_base(program, parts):
    """The header in the build folder: every thread's but the words' count and the
    regions, which a thread's own copy gives. The build wrote this very text (its
    `thread=`: the folder is named by it), so this leaves the file as it is."""
    made = importlib.import_module("komorebi.dsl.program")  # it imports this module
    name = made.thread_header_name(parts.compiled)
    text = header_text(parts)
    made.put(program.built.folder / name, text)
    program.built.files[name] = text


def joined(program, thread, parts, where):
    """A thread joined to its program: the load's words after its own, its header in
    a folder of its own ahead of its includes, its C built."""
    made = importlib.import_module("komorebi.dsl.program")
    own = len(thread.words)
    thread.words = [
        *thread.words,
        *(int(tensor.address) & abi.M32 for tensor in parts.tensors.values()),
    ]
    most = layout()["KMX_WORDS_MAX"]
    if len(thread.words) > most:
        raise ProgramError(
            f"{where}: {own} words of the thread's and {len(parts.tensors)} of the "
            f"load's (its tensors' addresses); a thread takes {most}"
        )
    text = header_text(parts, thread, own)
    folder = program.built.folder / (
        "thread-" + hashlib.sha256(text.encode()).hexdigest()[:8]
    )
    folder.mkdir(exist_ok=True)
    (folder / made.thread_header_name(parts.compiled)).write_text(text)
    thread.includes = [folder, *thread.includes]
    compiled_with(program, thread, parts, where)


# The C built at the make --------------------------------------------------------------
def compiled_with(program, thread, parts, where):
    """The thread's C compiled with its header, each error named (a handler, tensor
    or region the C names and the program lacks), then its image built."""
    tree = _env.find_tree()[0]
    if tree is None or _env.x280_compiler(tree) is None:
        program.log(
            f"  thread on cluster {thread.cluster}: not built (no X280 compiler)"
        )
        return
    includes = [native.kmx_directory(), *native.includes(), *map(Path, thread.includes)]
    defines = [f"{k}={v}" for k, v in sorted(thread.all_defines().items())]
    sources = thread.source_paths()
    errors = []
    with tempfile.TemporaryDirectory() as scratch:
        commands = build.image_commands(
            tree, Path(scratch) / "thread.bin", sources, includes, defines
        )
        for command in commands[: len(sources)]:
            checked = [*command[:-4], "-fsyntax-only", command[-3]]
            done = subprocess.run(checked, capture_output=True, text=True, check=False)
            if done.returncode:
                said = done.stderr.splitlines()
                errors += [line for line in said if ": error:" in line] or said[-3:]
    if errors:
        header = made_name(parts)
        lines = [f"  {line}" for line in errors[:12]] + hints(errors, parts, thread)
        raise ProgramError(
            f"{where}: the thread's C does not build with {header}:\n"
            + "\n".join(lines)
        )
    image = native.thread_image(thread, tree)
    program.log(
        f"  thread on cluster {thread.cluster}: {', '.join(s.name for s in sources)} "
        f"built with {made_name(parts)} ({image.stat().st_size} B)"
    )


def made_name(parts):
    made = importlib.import_module("komorebi.dsl.program")
    return made.thread_header_name(parts.compiled)


def hints(errors, parts, thread):
    """What each error names that the program lacks."""
    text = "\n".join(errors)
    upper = parts.compiled.name.upper()
    functions = {handler.function for handler in parts.handlers}
    prefixes = {function.split("_")[0] + "_" for function in functions}
    prefixes.add(parts.compiled.name.lower() + "_")
    lacks = {
        "handler": sorted(handler.name for handler in parts.handlers),
        "tensor": sorted(parts.tensors),
        "region": sorted([*thread.regions, *thread.shared]),
    }
    out = []
    for name in re.findall(r"implicit declaration of function '(\w+)'", text):
        prefix = next((p for p in prefixes if name.startswith(p)), None)
        if prefix is None or name in functions:
            continue
        rest = name.removeprefix(prefix)
        kind = "tensor" if rest.endswith("_dram") else "handler"
        rest = rest.removesuffix("_dram")
        out.append(f"  {name}: the program has no {kind} {rest} ({lacks[kind]})")
    for name in re.findall(r"'(\w+)' undeclared", text):
        for place, kind in (("_REGION_", "region"), ("_WORD_", "tensor"),
                            ("_HANDLER_", "handler")):  # fmt: skip
            if name.startswith(upper + place):
                rest = name.removeprefix(upper + place).lower()
                out.append(
                    f"  {name}: the program has no {kind} {rest} ({lacks[kind]})"
                )
    return list(dict.fromkeys(out))


# The header ---------------------------------------------------------------------------
def header_text(parts, thread=None, own=0):
    """The thread header: today's places (komorebi.dsl.program.thread_header), then
    the program field, the constants, the load's words, the regions and a function
    per handler that fills its command."""
    made = importlib.import_module("komorebi.dsl.program")
    compiled = parts.compiled
    upper, lower = compiled.name.upper(), compiled.name.lower()
    program = "kProgram" + made.camel(lower)
    lines = [
        made.thread_header(compiled).rstrip("\n"),
        "",
        "/* For the thread's C, from the same build: the program field, the class's",
        " * constants, the load's words, the regions, and a function per handler that",
        " * fills its command for kmx_submit. */",
        '#include "kmx.h"',
        "",
        f"/* The program field: {program} in a run of several programs, else 0. */",
        f"#ifdef {program}",
        f"#define {upper}_PROGRAM {program}",
        "#else",
        f"#define {upper}_PROGRAM 0u",
        "#endif",
    ]
    if parts.constants:
        lines += ["", "/* The class's constants at this make. */"]
        lines += [
            f"#define {name} {constant_text(value)}"
            for name, value in parts.constants.items()
        ]
    if parts.tensors:
        lines += tensor_lines(parts, upper, lower, own)
    lines += region_lines(parts, thread, upper, lower)
    regions = list(thread.regions) + list(thread.shared) if thread else []
    for handler in parts.handlers:
        lines += [""] + fill_function(handler, upper, lower, regions)
    for name in parts.opted:
        lines += [
            "",
            f"/* {name.upper()}: a set added on its first use "
            f"(km.{spelling('first_use')}) has no number at the build; the host",
            " * submits it, never this thread. */",
        ]
    return "\n".join(lines) + "\n"


def constant_text(value):
    if isinstance(value, bool):
        return "1u" if value else "0u"
    if isinstance(value, int):
        return f"{value}u" if value >= 0 else f"({value})"
    return repr(float(value))


def tensor_lines(parts, upper, lower, own):
    lines = [
        "",
        "/* The load's words, after the thread's own: each tensor of the make, its",
        " * address; its pages. */",
        f"#define {upper}_WORDS_OWN {own}u",
    ]
    for place, (name, tensor) in enumerate(parts.tensors.items()):
        lines.append(
            f"#define {upper}_WORD_{name.upper()} ({upper}_WORDS_OWN + {place}u)"
        )
        lines.append(f"#define {upper}_{name.upper()}_PAGES {int(tensor.pages)}u")
    lines += [
        f"static inline uint32_t {lower}_word(uint32_t place) {{",
        "    return kmx_args_get()->words[place];",
        "}",
    ]
    for name, tensor in parts.tensors.items():
        page = getattr(tensor, "aligned_page_bytes", None)
        if page is None or not getattr(tensor, "in_dram", True):
            continue
        lines += [
            f"/* {name}'s pages from `first`, for kmx_fetch and kmx_store. */",
            f"static inline kmx_dram {lower}_{name}_dram(uint32_t first) {{",
            "    const kmx_dram dram = {",
            f"        {lower}_word({upper}_WORD_{name.upper()}),",
            "        KM_COPY_INTERLEAVED,",
            f"        {int(page)}u,",
            "        first,",
            "    };",
            "    return dram;",
            "}",
        ]
    return lines


def region_lines(parts, thread, upper, lower):
    used = any(kind == "region" for h in parts.handlers for _, kind in h.words)
    if thread is None and not used:
        return []
    lines = [
        "",
        "/* The thread's regions: ids for kmx_region and their bytes; a region's",
    ]
    lines.append(" * bank address and the bank, a region parameter's two words. */")
    if thread is not None:
        for place, name in enumerate([*thread.regions, *thread.shared]):
            lines.append(f"#define {upper}_REGION_{name.upper()} {place}u")
            if name in thread.regions:
                lines.append(
                    f"#define {upper}_REGION_{name.upper()}_BYTES "
                    f"{thread.region_bytes(name)}u"
                )
    banks = {}
    for cluster, bank in sorted(BANK_OF_CLUSTER.items()):
        banks.setdefault(bank, []).append(cluster)
    *tests, (last, _) = banks.items()
    chosen = "".join(
        " || ".join(f"cluster == {c}u" for c in clusters) + f" ? {bank}u : "
        for bank, clusters in tests
    )
    lines += [
        f"static inline uint32_t {lower}_region_address(uint32_t id) {{",
        "    return (uint32_t)kmx_args_get()->region[id].offset;",
        "}",
        f"static inline uint32_t {lower}_bank(void) {{",
        "    const uint32_t cluster = kmx_args_get()->cluster;",
        f"    return {chosen}{last}u;",
        "}",
    ]
    return lines


def argument(name):
    return f"{name}_" if name in C_WORDS or name.startswith("km_") else name


def fill_function(handler, upper, lower, regions):
    """The C function that fills `handler`'s command from typed arguments: the
    make's tensors and the thread's regions from the load, the rest given."""
    arguments, said, body = [], [], []
    for place, (parameter, kind) in enumerate(handler.words):
        word = f"    km_command->a{place} = "
        tensor = handler.tensors.get(parameter)
        if kind == "tensor" and tensor is not None:
            body.append(word + f"{lower}_word({upper}_WORD_{tensor.upper()});")
            said.append(f"{parameter} the make's tensor")
        elif kind == "pages" and tensor is not None:
            body.append(word + f"{upper}_{tensor.upper()}_PAGES;")
        elif kind == "pages":
            arguments.append(f"{argument(parameter)}_pages")
            body.append(word + f"{argument(parameter)}_pages;")
        elif kind == "region" and parameter in regions:
            body.append(
                word + f"{lower}_region_address({upper}_REGION_{parameter.upper()});"
            )
            said.append(f"{parameter} the thread's region")
        elif kind == "region":
            arguments.append(f"{argument(parameter)}_region")
            body.append(word + f"{lower}_region_address({argument(parameter)}_region);")
            said.append(f"{parameter} a region's id")
        elif kind == "bank":
            body.append(word + f"{lower}_bank();")
        else:
            arguments.append(argument(parameter))
            body.append(word + f"{argument(parameter)};")
            said.append(parameter)
    flags = "(km_flags | KM_FLAG_RESULT)" if handler.result else "km_flags"
    asked = "; the leader's result asked" if handler.result else ""
    signature = [
        "km_command_t* km_command",
        "uint32_t km_flags",
        "uint32_t km_first",
        "uint32_t km_count",
        *(f"uint32_t {name}" for name in arguments),
    ]
    return [
        f"/* {handler.name.upper()}, line {handler.line}: "
        f"{', '.join(said) or 'no words'}{asked}. */",
        f"static inline void {handler.function}(",
        *(f"    {one}," for one in signature[:-1]),
        f"    {signature[-1]}",
        ") {",
        f"    kmx_command(km_command, {upper}_PROGRAM, "
        f"{upper}_HANDLER_{handler.name.upper()}, km_first, km_count);",
        f"    km_command->hdr |= {flags} << 8;",
        *body,
        "}",
    ]
