"""A handler's messages: a dynamic handler's message loop (the cases of a class's
dynamic handlers), a message sent or forwarded, a direct call, and the seeding
branch."""

import ast

from komorebi.dsl import front
from komorebi.dsl.compile import expressions
from komorebi.dsl.compile.cxx import camel
from komorebi.dsl.compile.records import Value
from komorebi.dsl.compile.source import Where, check_name, refuse, self_call, source_of
from komorebi.dsl.names import spelling
from komorebi.dsl.nodes import Block, Glue, Line


def arrive_loop(frame, node):
    """`for message in self.arrive():` each message given to this core, in packets
    (copied from the queue), each then forwarded, consumed or left."""
    if not isinstance(node.target, ast.Name):
        refuse(
            frame.where,
            node,
            "one message name",
            "write for message in self.arrive()",
        )
    args = node.iter.args
    cls = None
    if len(args) == 1:
        cls = message_class(frame, node, ast.Call(func=args[0], args=[], keywords=[]))
    elif not args:
        known = [
            v for v in frame.names.values()
            if isinstance(v, type) and issubclass(v, front.Message)
            and v is not front.Message
        ]  # fmt: skip
        cls = known[0] if len(known) == 1 else None
    if cls is None:
        refuse(
            frame.where,
            node,
            "the loop's message class is not known",
            "write self.arrive(Visit), Visit a km.Message of the module",
        )
    name = node.target.id
    check_name(frame.where, node, name)
    messages_of(frame, cls)
    frame.scope[name] = Value("message", name, words=cls.words)
    _, body = frame.statements(node.body)
    frame.scope.pop(name)
    if frame.compiled.folded:
        body = [handler_cases(frame, name, cls, body)]
    copy = (
        "const volatile tt_l1_ptr uint32_t* km_incoming = km_packet[km_arrived];\n"
        f"uint32_t {name}[kMessageWords] = {{}};  // word 0: the runtime's\n"
        "for (uint32_t km_word = 1; km_word < kMessageWords; km_word++) {\n"
        f"    {name}[km_word] = km_incoming[km_word];\n"
        "}"
    )
    inner = Block(
        "for (uint32_t km_arrived = 0; "
        "km_arrived < km_packet.count; km_arrived++) {",
        "}",
        [Line(copy), *body],
    )
    outer = Block(
        "while (km::Packet km_packet = km_messages.next()) {",
        "}",
        [inner, Line("km_messages.done(km_packet);")],
        {"km_messages"},
    )
    return [], [outer]


def arrives(frame):
    """Whether the handler has `for message in self.arrive():` (it is dynamic:
    compute's lines alone, the runtime's movers carry the messages)."""
    for node in ast.walk(frame.definition):
        if isinstance(node, ast.For) and self_call(node.iter, "arrive_messages"):
            return True
    return False


def direct_call(frame, node, name, call):
    """`went = self.direct(rank, message)`: 1 when the message went into its slot
    at rank (a declared pair), 0 when the slot was busy and nothing was sent."""
    if not frame.dynamic:
        refuse(frame.where, node, "a message call is a dynamic handler's", "use one")
    arguments, uses = message_arguments(frame, node, call, "direct_message")
    text = f"(km_messages.direct({arguments}) ? 1u : 0u)"
    if name in frame.scope:
        if frame.scope[name].kind != "int" or name not in frame.assigned:
            refuse(frame.where, node, f"{name} is already defined", "use a new name")
        return [], [Line(f"{name} = {text};", uses | {name})]
    frame.scope[name] = Value("int", name)
    declared = "uint32_t" if name in frame.assigned else "const uint32_t"
    # a line, not glue: the call sends, so it stays when the name goes unread
    return [], [Line(f"{declared} {name} = {text};", uses)]


def dynamic_start(frame, compute):
    """A dynamic handler: one in a class, no moves; its message loop's state, the
    rank and size of the loop's ranks."""
    if frame.compiled.dynamic is not None:
        refuse(
            frame.where,
            frame.definition,
            f"{frame.compiled.dynamic} is the class's dynamic handler already",
            "make one class for each dynamic handler",
        )
    frame.compiled.dynamic = frame.method.name
    for node in ast.walk(frame.definition):
        moving = isinstance(node, ast.Call) and (
            expressions.construct(frame, node)
            in ("stream", "reduce", "all_reduce", "broadcast")
            or isinstance(node.func, ast.Attribute)
            and node.func.attr in (spelling("push"), spelling("fill"))
            or any(k.arg == spelling("on") for k in node.keywords)
        )
        if moving:
            refuse(
                frame.where,
                node,
                "a dynamic handler is compute's lines alone: the runtime's movers "
                "carry its messages",
                "move pages in a handler of another class",
            )
    compute.insert(
        0, Glue("km::Messages km_messages(compute, command);", {"km_messages"})
    )
    compute[1] = Glue(
        "const uint32_t km_rank = km_messages.rank();", {"km_rank"}, {"km_messages"}
    )
    compute.append(
        Glue(
            "const uint32_t km_size = km_messages.count();",
            {"km_size"},
            {"km_messages"},
        )
    )


def handler_cases(frame, name, cls, body):
    """The message loop's body of a class of several dynamic handlers: a case of
    each, by the handler the message names (its own loop's, compiled here with its
    own lines and names); a message of no handler of the class is consumed."""
    cases = [
        Block(
            f"case kHandler{camel(frame.method.name)}: {{",
            "}",
            [*body, Line("break;")],
        )
    ]
    for folded in frame.compiled.folded.values():
        saved = (frame.method, frame.definition, frame.where, frame.scope)
        frame.method = folded
        frame.definition, lines, first = source_of(folded.function)
        frame.where = Where(frame.compiled.name, folded.name, lines, first)
        frame.scope = {}
        try:
            loop = arrive_of(frame.definition)
            if not isinstance(loop.target, ast.Name):
                refuse(
                    frame.where,
                    loop,
                    "one message name",
                    "write for m in self.arrive()",
                )
            check_name(frame.where, loop, loop.target.id)
            frame.scope[loop.target.id] = Value("message", name, words=cls.words)
            if loop.target.id != name:
                refuse(
                    frame.where,
                    loop,
                    f"the loops of one class's dynamic handlers name their "
                    f"message alike ({name})",
                    f"write for {name} in self.arrive()",
                )
            _, lines = frame.statements(loop.body)
        finally:
            frame.method, frame.definition, frame.where, frame.scope = saved
        cases.append(
            Block(
                f"case kHandler{camel(folded.name)}: {{",
                "}",
                [*lines, Line("break;")],
            )
        )
    cases.append(Line("default:\n    km_messages.consume();", {"km_messages"}))
    return Block("switch (km_messages.handler()) {", "}", cases, {"km_messages"})


def handler_named(frame, node, value):
    """The C++ of `to=Class.handler`: a dynamic handler of this class."""
    named = (
        isinstance(value, ast.Attribute)
        and isinstance(value.value, ast.Name)
        and value.value.id == frame.compiled.name
        and value.attr in (*frame.compiled.folded, frame.compiled.dynamic)
    )
    if not named:
        refuse(
            frame.where,
            node,
            f"to= names a dynamic handler of {frame.compiled.name}",
            f"write to={frame.compiled.name}.<handler>, a handler with "
            "`for message in self.arrive():`",
        )
    return f"kHandler{camel(value.attr)}"


def made_message(frame, node, name, cls, call):
    """`message = Visit(id=..., ...)`: a message of named words, the others 0."""
    if not frame.dynamic:
        refuse(
            frame.where, node, "a message is a dynamic handler's", "send it from one"
        )
    if call.args or any(k.arg not in cls.words for k in call.keywords):
        refuse(
            frame.where,
            node,
            f"a {cls.__name__} is made of its named words",
            f"write {cls.__name__}({', '.join(f'{w}=...' for w in cls.words)})",
        )
    messages_of(frame, cls)
    frame.scope[name] = Value("message", name, words=cls.words)
    text = [f"uint32_t {name}[kMessageWords] = {{}};"]
    uses = set()
    for given in call.keywords:
        word = f"k{cls.__name__}{camel(given.arg)}"
        text.append(f"{name}[{word}] = {expressions.integer(frame, given.value)};")
        uses |= expressions.uses(frame, given.value)
    return [], [Line("\n".join(text), uses)]


def message_arguments(frame, node, call, kind):
    """The C++ arguments of self.<kind>(rank, message[, to=]) and their uses."""
    to = [k for k in call.keywords if k.arg == spelling("to")]
    if len(call.args) != 2 or len(call.keywords) != len(to) or len(to) > 1:
        refuse(
            frame.where,
            node,
            "a rank and a message (and the handler there, to=)",
            f"write self.{spelling(kind)}(rank, m) or "
            f"self.{spelling(kind)}(rank, m, to={frame.compiled.name}.handler)",
        )
    message = call.args[1]
    if not (
        isinstance(message, ast.Name)
        and frame.scope.get(message.id, Value("")).kind == "message"
    ):
        refuse(frame.where, node, "the second is a message", "give a message's name")
    rank = expressions.integer(frame, call.args[0])
    uses = expressions.uses(frame, call.args[0]) | {"km_messages", message.id}
    handler = f", {handler_named(frame, node, to[0].value)}" if to else ""
    return f"{rank}, {message.id}{handler}", uses


def message_call(frame, node, call, kind):
    """self.send(rank, message), self.forward(rank, message), self.consume()."""
    if not frame.dynamic:
        refuse(frame.where, node, "a message call is a dynamic handler's", "use one")
    if kind == "consume":
        if call.args or call.keywords:
            refuse(frame.where, node, "consume() takes nothing", "write self.consume()")
        return [], [Line("km_messages.consume();", {"km_messages"})]
    arguments, uses = message_arguments(frame, node, call, kind)
    return [], [Line(f"km_messages.{kind}({arguments});", uses)]


def message_class(frame, node, call):
    """The km.Message class `call` names, or None."""
    if not isinstance(call, ast.Call):
        return None
    func = call.func
    found = frame.names.get(func.id) if isinstance(func, ast.Name) else None
    if isinstance(found, type) and issubclass(found, front.Message):
        return found
    return None


def messages_of(frame, cls):
    """The class's message words, for the header."""
    known = frame.compiled.messages.setdefault(cls.__name__, cls.words)
    if known != cls.words:
        refuse(frame.where, frame.definition, f"two {cls.__name__}s", "name one")


def seeding_branch(frame, node):
    """`if self.seeding():` the first messages, then the loop is told they are
    all sent."""
    if node.orelse:
        refuse(frame.where, node, "seeding has no else", "write the if alone")
    _, body = frame.statements(node.body)
    return [], [
        Block(
            "if (km_messages.seeding()) {",
            "}",
            [*body, Line("km_messages.seeded();")],
            {"km_messages"},
        )
    ]


def arrive_of(definition):
    """The `for message in self.arrive():` loop of a handler's def, or None."""
    for node in ast.walk(definition):
        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Attribute)
            and isinstance(node.iter.func.value, ast.Name)
            and node.iter.func.value.id == "self"
            and node.iter.func.attr == spelling("arrive_messages")
        ):
            return node
    return None
