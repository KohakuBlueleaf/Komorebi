"""A handler's expressions: an integer, a word and its sum, a constant of the class,
a field's use and its word, an attribute, an operand, a construct's kind, and the
names an expression reads."""

import ast

from komorebi.dsl.compile import scope
from komorebi.dsl.compile.cxx import camel
from komorebi.dsl.compile.source import refuse, self_call
from komorebi.dsl.front.names import kind_of, spelling
from komorebi.dsl.front.nodes import Line

OPERATORS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.FloorDiv: "/",
    ast.Mod: "%",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
}
FOLDS = {  # the operators a build-known int folds through
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.FloorDiv: lambda a, b: a // b,
}
COMPARISONS = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}


def attribute(frame, node):
    base = node.value
    if is_class_constant(frame, node):
        return str(class_constant(frame, node))
    if isinstance(base, ast.Name) and base.id == "self":
        kind = kind_of(node.attr, ("rank", "size", "leader", "x", "y", "core"))
        if kind == "core":
            return "km_self_core"
        if kind == "rank":
            return "km_rank"
        if kind == "size":
            return "km_size"
        if kind == "leader":
            return frame.leader_text
        if kind in ("x", "y"):
            frame.compiled.rectangle = True
            frame.compiled.rect_plain |= frame.rect_param is None
            frame.rect_read = True
            return f"km_self_{kind}"
        field_use(frame, node)
        refuse(frame.where, node, f"self.{node.attr} is not known", "use self.rank")
    if isinstance(base, ast.Name) and base.id in frame.scope:
        value = frame.scope[base.id]
        if value.kind in ("entry", "entries"):
            return entries_word(frame, node, value)
        if value.kind == "partition":
            if node.attr == "start":
                return value.first_text(value.cxx)
            if node.attr == "count":
                return f"{value.cxx}.len"
        if value.kind == "range" and node.attr in ("first", "count"):
            return value.first if node.attr == "first" else value.count
        if value.kind == "tensor" and node.attr == spelling("pages"):
            return frame.counts[base.id]
    refuse(
        frame.where, node, "an attribute level 1 does not know", "use .start, .count"
    )


def class_constant(frame, node):
    """self.X of a constant of the class: this program's value, lowered as a
    module constant is (a non-negative int)."""
    name = node.attr
    if name not in frame.compiled.class_constants:
        refuse(
            frame.where,
            node,
            f"self.{name} is a symbol the build solves after the handlers' lines",
            f"give {name} in the {spelling('plan')}, or as {name}= at the make",
        )
    value = frame.compiled.class_constants[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        refuse(
            frame.where,
            node,
            f"self.{name} = {value!r}: not a word",
            "make it an int",
        )
    return value


def constant(frame, node, name):
    """A name of the module: an int constant, or the plan's value for it."""
    if name in frame.plan:
        value = frame.plan[name]
    elif name in frame.names and isinstance(frame.names[name], int):
        value = frame.names[name]
    else:
        refuse(frame.where, node, f"{name} is not known here", "define it, or pass it")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        refuse(frame.where, node, f"{name} = {value!r}: not a word", "make it an int")
    frame.compiled.constants[name] = value
    return value


def constant_int(frame, node):
    """An int known at build: a literal, a module constant or a plan value, or
    + - * // of them."""
    if isinstance(node, ast.BinOp) and type(node.op) in FOLDS:
        left, right = constant_int(frame, node.left), constant_int(frame, node.right)
        if isinstance(node.op, ast.FloorDiv) and right == 0:
            refuse(frame.where, node, "a division by 0", "give a size the build knows")
        return FOLDS[type(node.op)](left, right)
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if is_class_constant(frame, node):
        return class_constant(frame, node)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return constant(frame, node, node.value)  # a symbol: the plan's
    if isinstance(node, ast.Name) and node.id not in frame.scope:
        return constant(frame, node, node.id)
    word_given(frame, node, "a size the build needs")
    refuse(frame.where, node, "a size the build needs", "give an int, or a plan name")


def construct(frame, node):
    """The construct a call is (a key of the names table), or None."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name) and func.value.id == "self":
            return kind_of(func.attr, ("partition", "result"))
        if (
            isinstance(func.value, ast.Attribute)
            and func.value.attr == spelling("tile")
            and func.attr == spelling("zeros")
        ):
            return "zeros"
        if isinstance(func.value, ast.Name) and func.value.id in ("km", "self"):
            return kind_of(
                func.attr, ("stream", "reduce", "all_reduce", "broadcast", "cxx")
            )
        value = isinstance(func.value, ast.Name) and frame.scope.get(func.value.id)
        if value and value.kind == "broadcast":
            return kind_of(func.attr, ("partition",))
    return None


def entries_word(frame, node, value):
    """The C++ of `e.word` (a word of an entry) or `table.count`."""
    cls = value.table["class"]
    if value.kind == "entries":
        if node.attr != spelling("entries_count"):
            refuse(
                frame.where,
                node,
                f"a table has its {spelling('entries_count')} and its entries",
                f"write {node.value.id}.{spelling('entries_count')}, or loop "
                f"over {node.value.id}",
            )
        return value.table["count"]
    if node.attr not in cls.words:
        refuse(
            frame.where,
            node,
            f"{node.value.id}, a {cls.__name__}, has no word {node.attr}",
            f"use one of {', '.join(cls.words)}",
        )
    return f"km_{node.value.id}_{node.attr}"


def field_sides(frame, node, field_):
    """A field of tiles' rows and columns, its symbols the plan's."""
    sides = []
    for side in frame.compiled.fields[field_].shape[:2]:
        if isinstance(side, str):
            if not isinstance(frame.plan.get(side), int):
                refuse(
                    frame.where,
                    node,
                    f"self.{field_}'s {side} is the plan's, and the plan has none",
                    f"make the program with {spelling('plan')}={{{side!r}: n}}",
                )
            side = frame.plan[side]
        sides.append(side)
    return tuple(sides)


def field_use(frame, node):
    """Refuses any other use of a field: `@`, slices and `.columns` of a field
    are tile expressions of level 2."""
    if scope.field_name(frame, node):
        refuse(
            frame.where,
            node,
            f"self.{node.attr} is a field: a line fills it from a stream and a "
            "km.cxx body takes it; `@`, slices and `.columns` of a field are "
            "level 2",
            f"write self.{node.attr}.fill(km.Stream(...)) and "
            f"km.cxx(..., takes=[self.{node.attr}])",
        )


def field_word(frame, node):
    """The C++ of `self.f[i]`, word i of word field f in this core's L1 (its
    index checked when it is a constant)."""
    name = word_name(frame, node)
    index = integer(frame, node.slice)
    count = frame.compiled.word_fields[name]
    if index.rstrip("u").isdigit() and int(index.rstrip("u")) >= count:
        refuse(
            frame.where,
            node,
            f"self.{name} has {count} words, 0 to {count - 1}",
            "index one of them",
        )
    return f"km_field_{name}[{index}]", uses(frame, node.slice) | {f"km_field_{name}"}


def integer(frame, node):
    """The C++ of an integer expression (glue)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        if isinstance(node.value, bool) or node.value < 0:
            refuse(frame.where, node, "a word is a non-negative int", "write one")
        return f"{node.value}u" if node.value > 0x7FFFFFFF else str(node.value)
    if isinstance(node, ast.Name):
        if node.id in frame.scope:
            value = frame.scope[node.id]
            if value.kind == "word" and frame.allow_words:
                return node.id
            if value.kind in ("streamed", "tile"):
                refuse(
                    frame.where,
                    node,
                    f"{node.id} is a tile: tile arithmetic is a tile function's",
                    "write it in a @km.tile_function and call that on the "
                    "stream's tiles",
                )
            if value.kind == "float":
                refuse(
                    frame.where,
                    node,
                    f"{node.id} is a float word (a float32's bits), and this is "
                    "the handler's integer arithmetic (a slice bound, a range, a "
                    "count)",
                    f"pass an int word for it, or use {node.id} in a tile "
                    "expression, a tile function's call or a body",
                )
            if value.kind != "int":
                refuse(
                    frame.where,
                    node,
                    f"{node.id} is a {value.kind}, not an int",
                    "use an int",
                )
            return node.id
        scope.not_ended(frame, node, node.id)
        return str(constant(frame, node, node.id))
    if isinstance(node, ast.Attribute):
        return message_word(frame, node) or attribute(frame, node)
    if word_name(frame, node):
        if not frame.field_line:
            refuse(
                frame.where,
                node,
                f"self.{word_name(frame, node)} is words the host reads, written "
                "by one CPU: its words are read in a line that sets one",
                f"keep the count in an int, or write self.{word_name(frame, node)}"
                "[i] += value",
            )
        return field_word(frame, node)[0]
    if isinstance(node, ast.BinOp):
        op = OPERATORS.get(type(node.op))
        if op is None:
            refuse(
                frame.where,
                node,
                "an operator level 1 has no C++ for",
                "use + - * //",
            )
        return f"({operand(frame, node.left)} {op} {operand(frame, node.right)})"
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        op = COMPARISONS.get(type(node.ops[0]))
        if op is None:
            refuse(
                frame.where,
                node,
                "a comparison level 1 has no C++ for",
                "use == < >",
            )
        left, right = node.left, node.comparators[0]
        mirror = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}
        for one, other, said in ((left, right, op), (right, left, mirror.get(op))):
            if (
                isinstance(one, ast.BinOp)
                and isinstance(one.op, ast.Sub)
                and isinstance(other, ast.Constant)
                and other.value == 0
            ):
                refuse(
                    frame.where,
                    node,
                    "words are unsigned: a subtraction below 0 wraps to a large "
                    "word, never below 0",
                    f"compare the two: write {ast.unparse(one.left)} {said or op} "
                    f"{ast.unparse(one.right)}",
                )
        return f"{operand(frame, left)} {op} {operand(frame, right)}"
    if isinstance(node, ast.BoolOp):
        truths = (ast.Compare, ast.BoolOp)
        for value in node.values:
            if not isinstance(value, truths) and not (
                isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not)
            ):
                word = "or" if isinstance(node.op, ast.Or) else "and"
                refuse(
                    frame.where,
                    node,
                    f"`{word}` of words is a word in Python ({ast.unparse(value)} "
                    "itself), and a truth in C++",
                    f"write {spelling('max')}(a, 1), or an if",
                )
        op = " && " if isinstance(node.op, ast.And) else " || "
        return "(" + op.join(integer(frame, v) for v in node.values) + ")"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return f"!({integer(frame, node.operand)})"
    if self_call(node, "direct_message"):
        direct_unnamed(frame, node)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in (spelling("min"), spelling("max"))
        and node.func.id not in frame.scope
    ):  # the smaller (larger) of ints, as words
        if len(node.args) < 2 or node.keywords:
            refuse(
                frame.where,
                node,
                f"{node.func.id}(...) of level 1 takes two ints or more",
                f"write {node.func.id}(a, b)",
            )
        op = "<" if node.func.id == spelling("min") else ">"
        text = operand(frame, node.args[0])
        for arg in node.args[1:]:
            other = operand(frame, arg)
            text = f"({text} {op} {other} ? {text} : {other})"
        return text
    refuse(
        frame.where, node, "not an integer of level 1", "write ints, names, + - * //"
    )


def is_class_constant(frame, node):
    """Whether `node` is self.X of a constant the class declares."""
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr in frame.compiled.declared
    )


def is_word(frame, node):
    """Whether `node` is a name or attribute the handler holds as an int word."""
    if isinstance(node, ast.Name):
        value = frame.scope.get(node.id)
        return value is not None and value.kind in ("int", "word")
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and (
            getattr(frame.scope.get(node.value.id), "kind", None)
            in ("entry", "entries")
            or node.value.id == "self"
        )
    )


def folded_int(frame, node):
    """An int the build knows, + - * // of them or one negated, or None."""
    if isinstance(node, ast.BinOp) and type(node.op) in FOLDS:
        left, right = folded_int(frame, node.left), folded_int(frame, node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.FloorDiv) and right == 0:
            return None
        return FOLDS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = folded_int(frame, node.operand)
        return None if value is None else -value
    return known_int(frame, node)


def known_int(frame, node):
    """An int the build knows (a literal, a module constant, a plan value), or
    None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if is_class_constant(frame, node):
        return class_constant(frame, node)
    if isinstance(node, ast.Name) and node.id not in frame.scope:
        value = frame.plan.get(node.id, frame.names.get(node.id))
        return value if isinstance(value, int) else None
    return None


def message_word(frame, node):
    """The C++ of `message.word` (a word of a message), or None."""
    base = node.value
    if not (isinstance(base, ast.Name) and base.id in frame.scope):
        return None
    value = frame.scope[base.id]
    if value.kind != "message":
        return None
    if node.attr not in value.words:
        refuse(
            frame.where,
            node,
            f"{base.id} has no word {node.attr}",
            f"use one of {', '.join(value.words)}",
        )
    cls = next(c for c, w in frame.compiled.messages.items() if w is value.words)
    return f"{base.id}[k{cls}{camel(node.attr)}]"


def mover_integer(frame, node, value, mover):
    """The C++ of an int of a line on `mover`: a word of that mover by its name,
    else an int of every CPU; a word of another CPU refused."""
    mover_word(frame, node, value, mover)
    if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name):
        page = frame.scope.get(value.value.id)
        if page is not None and page.kind == "page":
            return f"{value.value.id}_words[{integer(frame, value.slice)}]"
    if (
        isinstance(value, ast.Name)
        and getattr(frame.scope.get(value.id), "kind", None) == "word"
    ):
        return value.id
    return integer(frame, value)


def mover_word(frame, node, value, mover):
    """Refuses a word of another CPU where a line runs on `mover`."""
    for item in ast.walk(value):
        if not isinstance(item, ast.Name) or item.id not in frame.scope:
            continue
        known = frame.scope[item.id]
        if known.kind in ("word", "page") and known.mover != mover:
            where = {"A": "mover A", "B": "mover B"}.get(known.mover, known.mover)
            refuse(
                frame.where,
                node,
                f"{item.id} is a word of {where}, and this line runs on mover "
                f"{mover}",
                "give it a word of that mover, or an int of every CPU",
            )


def operand(frame, node):
    """integer() of an operator's operand: a comparison in parentheses, which C++
    binds looser than arithmetic (`a + (b < c)`)."""
    text = integer(frame, node)
    return f"({text})" if isinstance(node, ast.Compare) else text


def reads_word(frame, node, name):
    """Whether the value of `node` reads word field `name`."""
    return any(word_name(frame, item) == name for item in ast.walk(node.value))


def set_field_word(frame, node, target, op=""):
    """`self.f[i] = value` (or `+=` and the others): a word the host reads after
    the run, written by the unpack CPU alone (one write, not three); the value
    may read the field's words."""
    word, used = field_word(frame, target)
    frame.field_line = True
    try:
        value = integer(frame, node.value)
    finally:
        frame.field_line = False
    used |= uses(frame, node.value)
    used |= {
        f"km_field_{n}"
        for n in frame.compiled.word_fields
        if reads_word(frame, node, n)
    }
    return [], [Line(f"UNPACK(({word} {op}= {value}));", used)]


def set_word(frame, node, target, text, op=""):
    """`message.word = value` (or `+=` and the others): a word of a message."""
    word = message_word(frame, target)
    if word is None:
        refuse(frame.where, node, "an attribute set is a message's word", "set one")
    used = uses(frame, node.value) | {target.value.id}
    return [], [Line(f"{word} {op}= {text};", used)]


def stated_words(frame, node, what):
    """word_given of each `n` a rectangle's side states as its most (r:r + n,
    r:min(r + n, e))."""
    if not isinstance(node, ast.Subscript):
        return
    sides = node.slice.elts if isinstance(node.slice, ast.Tuple) else []
    for side in sides:
        if not isinstance(side, ast.Slice) or side.upper is None:
            continue
        upper = side.upper
        ends = upper.args if isinstance(upper, ast.Call) else [upper]
        for end in ends:
            if isinstance(end, ast.BinOp) and isinstance(end.op, ast.Add):
                word_given(frame, end.right, what)


def word_given(frame, node, what):
    """Refuses a command word where `what` needs a constant: the tier's error."""
    if isinstance(node, ast.Name) and any(p.name == node.id for p in frame.params):
        refuse(
            frame.where,
            node,
            f"{node.id} is a word of the command, known only when the command "
            f"runs, and {what} is fixed when the program is built: it needs a "
            "constant of the class",
            f"declare {node.id.upper()} = km.{spelling('constant')}(n) in the "
            f"class and write self.{node.id.upper()}",
        )


def word_name(frame, node):
    """The word field `node` indexes (`self.f[i]`), or None."""
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "self"
        and node.value.attr in frame.compiled.word_fields
    ):
        return node.value.attr
    return None


def word_sum(frame, node, value):
    """`word += x` (or another operator) on a mover's word: its mover's line; x
    reads ints and that mover's words."""
    op = OPERATORS.get(type(node.op))
    if op is None:
        refuse(
            frame.where, node, "an operator level 1 has no C++ for", "use + - * etc."
        )
    for item in ast.walk(node.value):
        other = isinstance(item, ast.Name) and frame.scope.get(item.id)
        if other and other.kind == "word" and other.mover != value.mover:
            refuse(
                frame.where,
                node,
                f"{item.id} is another CPU's word",
                "add words of one mover",
            )
    frame.allow_words = True
    text = f"{value.cxx} {op}= {integer(frame, node.value)};"
    frame.allow_words = False
    used = uses(frame, node.value) | {value.cxx}
    return [Line(text, used, mover=value.mover)], []


def uses(frame, node):
    """The names an expression reads (scope names and generated ones)."""
    found = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Name) and item.id in frame.scope:
            value = frame.scope[item.id]
            found.add(value.cxx or item.id)
        if isinstance(item, ast.Attribute) and isinstance(item.value, ast.Name):
            if item.value.id == "self":
                kind = kind_of(item.attr, ("rank", "size", "x", "y", "core"))
                if kind in ("x", "y", "core"):
                    found.add(f"km_self_{kind}")
                elif kind:
                    found.add(f"km_{kind}")
            if item.value.id in frame.counts and item.attr == spelling("pages"):
                found.add(frame.counts[item.value.id])
            entry = frame.scope.get(item.value.id)
            if entry is not None and entry.kind in ("entry", "entries"):
                found.add(entries_word(frame, item, entry))
    return found


def direct_unnamed(frame, node):
    """self.direct(...) anywhere but as a name's value: refused, by its bool."""
    refuse(
        frame.where,
        node,
        "self.direct(rank, message) says whether the message went (0: its slot "
        "is busy and nothing was sent), so its value is a name's",
        "write went = self.direct(rank, message), then forward it when went == 0",
    )
