"""Where a handler's line is in its source: its def parsed, a line's text, a refusal
that names it, a name the generated C++ takes, a wait of the generated code there."""

import ast
import inspect
import keyword
import re
import textwrap
from dataclasses import dataclass

from komorebi.dsl.compile.records import Refused, Wait

# Every local the generated code declares starts with GENERATED_PREFIX, which a
# parameter or a variable may not; the runtime's names bodies use are taken too.
GENERATED_PREFIX = "km_"
GENERATED = {"command", "movement", "compute"}
# The runtime's where names (komorebi.session.state) of each kind of wait.
WAIT_DATA, WAIT_INPUT, WAIT_OUTPUT = "data", "cinput", "coutput"
WAIT_PORT, WAIT_EMIT = "port", "emit"
WAIT_VALID, WAIT_DONE = "broadcastvalid", "broadcastdone"
WAIT_FREE = "broadcastfree"
CXX_KEYWORDS = {
    "auto",
    "bool",
    "case",
    "char",
    "const",
    "default",
    "delete",
    "do",
    "double",
    "enum",
    "float",
    "goto",
    "int",
    "long",
    "new",
    "private",
    "public",
    "register",
    "short",
    "signed",
    "sizeof",
    "static",
    "struct",
    "switch",
    "template",
    "this",
    "throw",
    "typedef",
    "union",
    "unsigned",
    "using",
    "virtual",
    "void",
    "volatile",
    "km",
    "uint32_t",
}


def source_of(function):
    """The function's def, parsed, with its file's line numbers."""
    lines, first = inspect.getsourcelines(function)
    tree = ast.parse(textwrap.dedent("".join(lines)))
    ast.increment_lineno(tree, first - 1)
    return tree.body[0], lines, first


@dataclass(frozen=True)
class Where:
    """A handler in its source: its class's name, its own name, its file's lines from
    its def, and the file's number of the first of them."""

    cls: str
    handler: str
    lines: list
    first: int


def line_text(where, node):
    """The node's source on one line: a compound statement's head, else all of it,
    each line stripped."""
    body = getattr(node, "body", None)
    last = None
    if isinstance(body, list) and body and body[0].lineno > node.lineno:
        last = body[0].lineno - 1
    last = last or getattr(node, "end_lineno", None) or node.lineno
    lines = where.lines[node.lineno - where.first : last - where.first + 1]
    text = " ".join(line.strip() for line in lines if not line.strip().startswith("#"))
    for _ in range(2):  # `a, )` closes twice: its space, then its comma
        text = re.sub(r"([(\[]) | ([)\]])|,([)\]])", r"\1\2\3", text)
    return text


def refuse(where, node, why, instead):
    """Raises Refused naming the class, the handler, the line and its text: `why`,
    then `instead`."""
    text = line_text(where, node)
    if (node.end_lineno or node.lineno) - node.lineno > 2 and not hasattr(
        node, "body"
    ):  # a long call (an inline body): its first line
        text = where.lines[node.lineno - where.first].strip() + " ..."
    raise Refused(
        f"{where.cls}.{where.handler}, line {node.lineno}, `{text}`: {why}; {instead}"
    )


def check_name(where, node, name):
    """Refuses a name the generated C++ takes."""
    if (
        name in GENERATED
        or name.startswith(GENERATED_PREFIX)
        or keyword.iskeyword(name)
        or name in CXX_KEYWORDS
    ):
        refuse(
            where,
            node,
            f"the name {name} is taken by the generated C++ (its own names start "
            f"{GENERATED_PREFIX}, and {', '.join(sorted(GENERATED))} are the "
            "runtime's)",
            "give it another name",
        )


def wait(where, waits, node, cpu, kind):
    """A wait of the generated code on `cpu` at `node`, the runtime's where `kind`,
    added to `waits`."""
    waits.append(Wait(where.handler, cpu, kind, node.lineno, line_text(where, node)))
