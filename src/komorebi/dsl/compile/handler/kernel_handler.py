"""A handler whose body is hand-written kernel files (`@km.handler(km.kernel(...))`):
its checks, one call a role in its case (lowered as km.cxx bodies are: `take`'s
names, Lines pruned and rendered), the files' include lines, the header's km.Pair and
constant lines, the state a DSL handler sets at its entry beside one, its refusals."""

import ast
import os
from pathlib import Path

from komorebi.dsl.compile.casts.messages import arrive_of
from komorebi.dsl.compile.cxx import camel
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compile.source import refuse, source_of
from komorebi.dsl.compile.tiles.bodies import CPU_GUARDS, take
from komorebi.dsl.front import front
from komorebi.dsl.front.names import spelling
from komorebi.dsl.front.nodes import Group, Line

# The kernel source each role's file is included in, and the CPUs of each source.
ROLE_KERNEL = {
    "movement": "movement",
    "reader": "movement",
    "writer": "movement",
    "compute": "compute",
    "unpack": "compute",
    "math": "compute",
    "pack": "compute",
}
KERNEL_CPUS = {
    "movement": ("mover_a", "mover_b"),
    "compute": ("unpack", "math", "pack"),
}
# Dst cleared whole on the math CPU (projects/raster/kernels/apply_compute.hpp).
CLEAR_DST = (
    "#if defined(TRISC_MATH)\n"
    "ckernel::addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, "
    ".dest = {.incr = 0}}\n"
    "    .set(ADDR_MOD_1);\n"
    "TTI_ZEROACC(p_zeroacc::CLR_ALL, DST_ACCUM_MODE, 0, ADDR_MOD_1, 0);\n"
    "TTI_STALLWAIT(p_stall::STALL_SFPU | p_stall::STALL_SYNC, p_stall::MATH);\n"
    "#endif"
)
FORMATS_HEADER = '#include "api/compute/reconfig_data_format.h"'
# The packer set for a buffer whose index is a constant: the library's call with the
# index a template argument, forced inline (device/core/km_pack_format.hpp).
PACK_FORMAT = "km::pack_format"
PACK_FORMAT_HEADER = '#include "komorebi/device/core/km_pack_format.hpp"'


def lowered(frame):
    """(the movers' nodes, compute's) of a km.kernel handler: its files checked and
    recorded, then `<entry>_<role>(movement or compute, command, <args>)` on each
    role's CPUs, its #line the def's."""
    method, definition = frame.method, frame.definition
    kernel, where = method.kernel, frame.where
    checked(frame)
    frame.compiled.kernel_handlers[method.name] = kernel
    entry = kernel.entry or method.function.__name__
    movement, compute = [], []
    for role, path in kernel.roles.items():
        source = ROLE_KERNEL[role]
        files = frame.compiled.kernel_files[source]
        files.setdefault(str(resolved(where, path)), set()).update(
            front.KERNEL_ROLES[role]
        )
        mover = {"reader": "A", "writer": "B"}.get(role)
        bound, uses = [], set()
        names = []
        for param in frame.params:
            if source == "compute" and param.kind == "tensor":
                continue  # in DRAM: the movers'
            item = ast.Name(id=param.name, ctx=ast.Load())
            place = mover or ("A" if source == "movement" else role)
            uses |= take(frame, definition, item, place, bound, [], [])
            names.append(param.name)
        args = ", ".join([source, "command", *names])
        text = f"{entry}_{role}({args});"
        if source == "movement":
            text = f"if (!{entry}_{role}({args})) {{\n    return false;\n}}"
        body = f'#line {definition.lineno} "{where.file}"\n{text}\n#line RESUME'
        if bound:
            inner = "\n".join("    " + line for line in [*bound, *body.split("\n")])
            body = "{\n" + inner + "\n}"
        if role in KERNEL_CPUS["compute"]:  # one compute CPU's alone
            body = f"#if {CPU_GUARDS[role]}\n{body}\n#endif"
        comment = f"line {definition.lineno}: the hand kernel's {role}, {entry}_{role}"
        group = Group(comment, [Line(body, uses, mover=mover)])
        (movement if source == "movement" else compute).append(group)
    return movement, compute


def resolved(where, path):
    """A role's file: its path as given, a relative one from the class's file."""
    return (Path(where.file).parent / path).resolve()


def checked(frame):
    """Refuses a km.kernel handler the build does not take, at its line: km.handler's
    other options, roles given twice or none, a file not there, a body of lines, a
    parameter that is no command word of a hand kernel's."""
    method, definition, where = frame.method, frame.definition, frame.where
    kernel = method.kernel
    decorator = definition.decorator_list[0] if definition.decorator_list else None
    node = decorator or definition
    named = f"km.{spelling('kernel')}"
    if method.kernel_given:
        given = ", ".join(f"{name}=" for name in method.kernel_given)
        refuse(
            where,
            node,
            f"{given} with {named}: a hand kernel's handler is resident and called "
            "as written, by its command alone",
            f"drop {given}, or write the handler in the DSL",
        )
    roles = set(kernel.roles)
    if not roles:
        refuse(
            where,
            node,
            f"{named}(...) names no file: a hand kernel has a file for a role at least",
            "give movement=, reader=, writer=, compute=, unpack=, math= or pack=",
        )
    wholes = (("movement", ("reader", "writer")), ("compute", KERNEL_CPUS["compute"]))
    for whole, parts in wholes:
        again = sorted(roles & set(parts), key=parts.index)
        if whole in roles and again:
            refuse(
                where,
                node,
                f"{whole}= is the file of {', '.join(front.KERNEL_ROLES[whole])}, and "
                f"{', '.join(f'{r}=' for r in again)} names one of them again",
                f"give {whole}= alone, or {' and '.join(f'{r}=' for r in parts)}",
            )
    for role, path in kernel.roles.items():
        if not isinstance(path, str | Path):
            refuse(
                where,
                node,
                f"{role}={path!r}: a file's path is a str or a Path",
                f'write {role}="kernels/name.hpp"',
            )
        if not resolved(where, path).is_file():
            refuse(
                where,
                node,
                f"{role}= names {resolved(where, path)}, which is no file",
                f"give a path relative to {Path(where.file).parent}, or an "
                "absolute one",
            )
    for statement in definition.body:
        docstring = isinstance(statement, ast.Expr) and isinstance(
            statement.value, ast.Constant
        )
        if not (docstring or isinstance(statement, ast.Pass)):
            refuse(
                where,
                statement,
                f"a handler is a kernel or lines; a hand-written line in a DSL "
                f"handler is km.{spelling('cxx')}",
                f"leave the body a docstring (or pass), or drop {named}",
            )
    constants = front.constant_parameters(method.function)
    unworded = [p.name for p in frame.params if p.kind in ("region", "rect")]
    if front.rect_given(method.function) is not None:
        unworded.append("rect")
    for name in [*constants, *unworded]:
        what = "a constant parameter" if name in constants else "a region or rectangle"
        refuse(
            where,
            definition,
            f"{name} is {what}: a hand kernel's call passes tensors, ints, floats "
            "and patterns, and builds no set",
            "pass its words as ints, or write the handler in the DSL",
        )


def constants_kept(compiled, listed):
    """kConst<Name> of each constant the classes holding a km.kernel handler declare,
    this program's value (`listed`: (class, values)); a name of two values refused."""
    if not compiled.kernel_handlers:
        return
    kept, owners = {}, {}
    for owner, values in listed:
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                continue  # a word's constants alone
            if name in kept and kept[name] != value:
                raise Refused(
                    f"{compiled.name}: constant {name} is {kept[name]} in "
                    f"{owners[name]} and {value} in {owner}, and the header holds one "
                    f"kConst{camel(name.lower())} for the hand kernels; give both one "
                    "value, or rename one"
                )
            kept[name], owners[name] = value, owner
    compiled.kernel_constants = kept


def entered(compiled, formats, builds):
    """Before the files are written: the build folders' folder (`builds`, the include
    paths' base), and, beside a km.kernel handler not clean=True, each DSL handler's
    compute case begun with the start-up's formats (`formats`, start_formats') and Dst
    cleared."""
    compiled.builds = str(builds)
    if compiled.entered or not entry_state(compiled):
        return
    compiled.entered = True
    for header in (FORMATS_HEADER, PACK_FORMAT_HEADER):
        if header not in compiled.body_includes["compute"]:
            compiled.body_includes["compute"].append(header)
    lines = [
        "// the start-up's formats and a clear Dst again: a hand kernel may have "
        "changed them"
    ]
    startup = compiled.startup
    if startup is not None:  # km.Startup: its inputs and its output
        names = [f"kField{camel(name)}" for name in startup.fields]
        second = names[1] if len(names) == 3 else names[0]
        lines.append(f"reconfig_data_format({names[0]}, {second});")
        lines.append(f"{PACK_FORMAT}<{names[-1]}>();")
    else:
        lines += [line.strip() for line in formats[1:]]
    head = Line("\n".join([*lines, CLEAR_DST]))
    for name in compiled.handlers:
        if name not in compiled.kernel_handlers:
            compiled.compute[name] = [head, *compiled.compute[name]]


def entry_state(compiled):
    """Whether the DSL handlers set the start-up's state at their entry: a km.kernel
    handler not clean=True and a DSL handler in one program."""
    kernels = compiled.kernel_handlers
    return any(not k.clean for k in kernels.values()) and any(
        name not in kernels for name in compiled.handlers
    )


def include_lines(compiled, source):
    """The #include of each hand file of `source` ("movement" or "compute"), after
    the class's header: once, under the #if of each CPU using it unless all do."""
    lines = []
    base = Path(compiled.builds or Path(compiled.source).resolve().parent) / "-"
    for path, cpus in compiled.kernel_files[source].items():
        text = f'#include "{Path(os.path.relpath(path, base)).as_posix()}"'
        if set(cpus) == set(KERNEL_CPUS[source]):
            lines.append(text)
            continue
        for cpu in KERNEL_CPUS[source]:
            if cpu in cpus:
                lines += [f"#if {CPU_GUARDS[cpu]}", text, "#endif"]
    return lines


def pair_lines(buffer):
    """A km.Pair's header lines: its pair and the pages of each half."""
    return [
        f"constexpr km::BufferPair {buffer.name} = "
        f"{{{buffer.index_a}, {buffer.index_b}}};",
        f"constexpr uint32_t {buffer.name}Pages = {buffer.pages[0]};",
    ]


def constant_lines(compiled):
    """The header's kConst<Name> of each constant, for the hand kernels."""
    if not compiled.kernel_constants:
        return []
    return [
        "",
        f"// The class's constants (km.{spelling('constant')}), for its hand kernels.",
        *(
            f"constexpr uint32_t kConst{camel(name.lower())} = {value};"
            for name, value in compiled.kernel_constants.items()
        ),
    ]


def made_refused(name, methods, form, chain, chained, islands, handlers):
    """Refuses the make of a program with a km.kernel handler (`methods`) in what it
    does not support: the runtime's form, a chain build or a km.chain method calling
    it (`chained`), a dynamic handler, an island session, `handlers=` of it."""
    kernels = [m for m in methods if m.kernel is not None]
    if not kernels:
        return
    first = kernels[0]
    at = f"{name}.{first.name} (line {source_of(first.function)[0].lineno})"
    named = f"a km.{spelling('kernel')} handler"
    if form == "runtime":
        raise Refused(
            f'{at}: {named} with {spelling("program_form")}="runtime": its hand '
            "files are included in the class's own kernels, which the runtime's form "
            'does not build; make it in the kernel form (form="kernel")'
        )
    if chain:
        raise Refused(
            f"{at}: {named} with {spelling('chain')}=True: a chain's records state "
            "what each call reads and writes, and a hand kernel's are its own C++; "
            "submit its commands, or write the handler in the DSL"
        )
    for chain_name, value in chained.items():
        called = calls_of(value.function, {m.name for m in kernels})
        if called:
            raise Refused(
                f"{name}.{chain_name}: the km.{spelling('chain')} method calls "
                f"{called[0]}, {named}: a chain's records state what each call reads "
                "and writes, and a hand kernel's are its own C++; call DSL handlers "
                "alone in the chain"
            )
    for method in methods:
        if arrive_of(source_of(method.function)[0]):
            raise Refused(
                f"{at}: {named} beside the dynamic handler {method.name}: a dynamic "
                "class's kernels are its message loop; make them two programs"
            )
    if islands[0] > 1:
        raise Refused(
            f"{at}: {named} in a session of {islands[0]} islands: what may cross "
            "a link is found from the lines' moves, and a hand kernel's are its own "
            "C++; make it in a session of one card"
        )
    for kernel in kernels:
        if kernel.name in (handlers or {}):
            raise Refused(
                f"{name}.{kernel.name}: {spelling('handlers')}= lists sets of "
                f"{named}, which has no constant parameters: one build serves every "
                "call; drop it from handlers="
            )


def calls_of(function, names):
    """The handlers of `names` a km.chain method calls as self.name(...)."""
    return [
        node.func.attr
        for node in ast.walk(source_of(function)[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr in names
    ]
