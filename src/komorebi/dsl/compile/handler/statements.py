"""A handler's statements: each statement lowered to the movers' and compute's lines
(an assignment, an augmented one, a call, a for loop over a range or a zip, an if, a
return), and the ends a held block or stream leaves after them."""

import ast

from komorebi.dsl.compile import expressions, scope, tile_lookup
from komorebi.dsl.compile.casts import broadcasts, messages
from komorebi.dsl.compile.dst import dst_tiles
from komorebi.dsl.compile.expressions import OPERATORS
from komorebi.dsl.compile.handler import entry_tables, skew
from komorebi.dsl.compile.pages import (
    drained,
    field_fills,
    page_streams,
    page_words,
    puts,
    ranges,
)
from komorebi.dsl.compile.records import Value
from komorebi.dsl.compile.source import check_name, line_text, refuse, self_call
from komorebi.dsl.compile.tiles import (
    bodies,
    functions,
    products,
    reduce,
    steps,
    writes,
)
from komorebi.dsl.front.names import spelling
from komorebi.dsl.front.nodes import Block, Glue, Group, Line


def statements(frame, body):
    """The movement nodes and the compute nodes of a list of statements (the frame's
    `statements`, through which the families lower the bodies they hold)."""
    movement, compute = [], []
    frame.blocks.append(id(body))
    for statement in body:
        m, c = statement_lines(frame, statement)
        comment = f"line {statement.lineno}: {line_text(frame.where, statement)}"
        if statement.end_lineno - statement.lineno > 2 and not hasattr(
            statement, "body"
        ):  # a long call (an inline body): its first line
            first = frame.where.lines[statement.lineno - frame.where.first].strip()
            comment = f"line {statement.lineno}: {first} ..."
        if m:
            movement.append(Group(comment, m))
        if c:
            compute.append(Group(comment, c))
    compute += held_ends(frame, body)
    compute += held_block_ends(frame, body)
    frame.blocks.pop()
    return movement, compute


def assign(frame, node):
    if len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
        if tile_lookup.tile_function_of(frame, node.value) is not None:
            return functions.tile_results(frame, node, node.targets[0], node.value)
        return page_words.read_words(frame, node, node.targets[0], node.value)
    if len(node.targets) == 1 and expressions.word_name(frame, node.targets[0]):
        return expressions.set_field_word(frame, node, node.targets[0])
    if len(node.targets) == 1 and isinstance(node.targets[0], ast.Attribute):
        return expressions.set_word(
            frame, node, node.targets[0], expressions.integer(frame, node.value)
        )
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        refuse(frame.where, node, "one name a line", "assign one name")
    name = node.targets[0].id
    check_name(frame.where, node, name)
    scope.not_parameter(frame, node, name)
    value = node.value
    made = messages.message_class(frame, node, value)
    if made is not None:
        return messages.made_message(frame, node, name, made, value)
    if functions.composed_result(frame, node, name, value):
        return [], []
    if scope.field_name(frame, value) is not None:  # a DST tile of a field's
        return dst_tiles.from_field(frame, node, name, scope.field_name(frame, value))
    value = functions.tile_expression(frame, node, value) or value
    if steps.step_of_field(frame, value):
        return steps.started_from_field(frame, node, name, value)
    if tile_lookup.tile_function_of(frame, value) is not None:
        return functions.tile_call(frame, node, value, name)
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == spelling("read")
        and not value.keywords
        and ranges.is_range(frame, value.func.value)
    ):
        return page_words.read_word(
            frame, node, name, ranges.range_of(frame, value.func.value), value.args
        )
    if entry_tables.entries_call(frame, value) is not None:
        return entry_tables.entries_made(frame, node, name, value)
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == spelling("bump")
        and scope.field_name(frame, value.func.value)
    ):
        return puts.bump_line(frame, node, name, value)
    if self_call(value, "direct_message"):
        return messages.direct_call(frame, node, name, value)
    if self_call(value, "send"):
        return messages.direct_call(frame, node, name, value, "send")
    call = expressions.construct(frame, value)
    if call == "partition":
        return ranges.partition(frame, node, name, value)
    if call == "zeros":
        return dst_tiles.zeros(frame, node, name, value)
    if call == "stream" and scope.field_name(
        frame, value.args[0] if value.args else None
    ):
        return drained.drained_stream(frame, node, name, value)
    if call == "stream" and not write_keyword(frame, value):
        return page_streams.held_stream(frame, node, name, value)
    if call == "stream":
        return page_streams.write_stream(frame, node, name, value)
    if call == "reduce":
        return reduce.reduce(frame, node, name, value)
    if call == "all_reduce":
        return reduce.reduce(frame, node, name, value, everywhere=True)
    if call == "broadcast":
        return broadcasts.broadcast(frame, node, name, value)
    if call == "cxx":
        return bodies.inline(frame, node, value, name)
    if call:
        refuse(
            frame.where,
            node,
            f"km.{spelling(call)} gives no value to name",
            "write it on a line of its own",
        )
    if ranges.is_range(frame, value):
        frame.scope[name] = ranges.range_of(frame, value)
        return [], []
    text = expressions.integer(frame, value)
    declared = "uint32_t" if name in frame.assigned else "const uint32_t"
    if name in frame.scope:
        if frame.scope[name].kind != "int" or name not in frame.assigned:
            refuse(frame.where, node, f"{name} is already defined", "use a new name")
        line = f"{name} = {text};"
        uses = expressions.uses(frame, value) | {name}
        return [Line(line, uses)], [Line(line, uses)]
    frame.scope[name] = Value("int", name)
    glue = f"{declared} {name} = {text};"
    uses = expressions.uses(frame, value)
    return [Glue(glue, {name}, uses)], [Glue(glue, {name}, uses)]


def augmented(frame, node):
    target = node.target
    if expressions.word_name(frame, target):
        op = OPERATORS.get(type(node.op))
        if op is None:
            refuse(frame.where, node, "an operator level 1 has no C++ for", "use + - *")
        return expressions.set_field_word(frame, node, target, op)
    field_ = scope.field_name(frame, target)
    if (
        field_ is not None
        and tile_lookup.tile_function_of(frame, node.value) is not None
    ):
        return products.field_product(frame, node, field_, node.value)
    if isinstance(target, ast.Attribute):
        op = OPERATORS.get(type(node.op))
        if op is None:
            refuse(frame.where, node, "an operator level 1 has no C++ for", "use + - *")
        return expressions.set_word(
            frame, node, target, expressions.integer(frame, node.value), op
        )
    if isinstance(target, ast.Name):
        scope.not_parameter(frame, node, target.id)
        scope.not_ended(frame, node, target.id)
    if not isinstance(target, ast.Name) or target.id not in frame.scope:
        refuse(frame.where, node, "an unknown name", "assign it first")
    value = frame.scope[target.id]
    if (
        value.kind == "tile"
        and tile_lookup.tile_function_of(frame, node.value) is not None
    ):
        return products.product(frame, node, value, node.value)
    if value.kind == "tile":
        return dst_tiles.accumulate(frame, node, value)
    if value.kind == "word" and value.mover in ("A", "B"):
        return expressions.word_sum(frame, node, value)
    if value.kind != "int":
        refuse(frame.where, node, f"{target.id} is a {value.kind}", "add to an int")
    op = OPERATORS.get(type(node.op))
    if op is None:
        refuse(
            frame.where, node, "an operator level 1 has no C++ for", "use + - * etc."
        )
    text = f"{target.id} {op}= {expressions.integer(frame, node.value)};"
    uses = expressions.uses(frame, node.value) | {target.id}
    return [Line(text, uses)], [Line(text, uses)]


def branch(frame, node):
    if self_call(node.test, "seeding"):
        return messages.seeding_branch(frame, node)
    broadcast = broadcasts.broadcast_branch(frame, node)
    if broadcast is not None:
        return broadcast
    condition = expressions.integer(frame, node.test)
    uses = expressions.uses(frame, node.test)
    live = frame.dst_live
    leader, frame.in_leader = frame.in_leader, frame.in_leader or leader_test(
        frame, node
    )
    if leader_test(frame, node):
        frame.leader_condition = (condition, uses)  # a held sum's free (sum_held)
    before = set(frame.scope)
    held = unwritten(frame)
    entry = written_states(held)
    then_m, then_c = frame.statements(node.body)
    scope.scoped(frame, "if", node, before)
    frame.in_leader = leader
    after_then, then_live = written_states(held), frame.dst_live
    restore(held, entry)  # the else branch starts where the if did
    frame.dst_live = live
    else_m, else_c = frame.statements(node.orelse)
    scope.scoped(frame, "if", node, before)
    sides = (then_m, then_c, else_m, else_c)
    if one_branch_writes(frame, node, held, (entry, after_then), sides) and (
        then_live is None and frame.dst_live is None
    ):
        live = None  # the tile in DST written (or dropped) on each branch
    if frame.dst_live != live:
        refuse(
            frame.where,
            node,
            "a tile held in DST on one branch only",
            "write the tile after the if",
        )
    movement, compute = [], []
    for body_then, body_else, out in (
        (then_m, else_m, movement),
        (then_c, else_c, compute),
    ):
        if body_then:
            out.append(Block(f"if ({condition}) {{", "}", body_then, uses))
            if body_else:
                out.append(Block("else {", "}", body_else))
        elif body_else:
            out.append(Block(f"if (!({condition})) {{", "}", body_else, uses))
    return movement, compute


WRITTEN = ("kind", "popped", "waited", "buffer", "moved", "dropped")


def unwritten(frame):
    """The handler's tiles and reduce sums not yet written, by name."""
    return {
        name: value
        for name, value in frame.scope.items()
        if isinstance(value, Value)
        and value.kind in ("tile", "reduced")
        and not value.popped
    }


def written_states(held):
    return {name: [getattr(v, a, None) for a in WRITTEN] for name, v in held.items()}


def restore(held, saved):
    for name, value in held.items():
        for attribute, state in zip(WRITTEN, saved[name], strict=True):
            setattr(value, attribute, state)


def one_branch_writes(frame, node, held, saved, sides):
    """A tile or reduce sum written on each branch of an if: written after it; on
    one branch alone, the other empty: dropped there (compute's DST released, or
    mover B's sum freed), else refused. True when one was written or dropped."""
    then_m, then_c, else_m, else_c = sides
    entry, after_then = saved
    any_written = False
    for name, value in held.items():
        then_wrote = after_then[name][WRITTEN.index("popped")]
        else_wrote = value.popped
        if not then_wrote and not else_wrote:
            continue
        any_written = True
        if then_wrote and else_wrote:
            continue
        if value.kind == "reduced" and leader_test(frame, node):
            if then_wrote:  # the leader's sum: the other ranks hold none
                restore({name: value}, {name: after_then[name]})
            continue
        empty = not (node.orelse if then_wrote else node.body)
        if not empty or value.kind == "reduced" and value.summed:
            refuse(
                frame.where,
                node,
                f"{name} is written on one branch of this if only, and the other "
                "has lines of its own: compute and mover B would hold it on that "
                "branch",
                f"write {name} on each branch, or after the if",
            )
        kind = entry[name][WRITTEN.index("kind")]
        if then_wrote:
            restore({name: value}, {name: after_then[name]})
        movement, compute = dst_tiles.dropped(frame, value, kind)
        (else_m if then_wrote else then_m).extend(movement)
        (else_c if then_wrote else then_c).extend(compute)
        value.moved, value.dropped = None, node.lineno
    return any_written


def call_statement(frame, node, call):
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr == spelling("push"):
        return functions.push(frame, node, call)
    if isinstance(func, ast.Attribute) and func.attr == spelling("write"):
        return writes.write(frame, node, call)
    if isinstance(func, ast.Attribute) and func.attr == spelling("put"):
        return puts.put_line(frame, node, call)
    if (
        isinstance(func, ast.Attribute)
        and func.attr == spelling("stream_end")
        and isinstance(func.value, ast.Name)
        and getattr(frame.scope.get(func.value.id), "kind", None) == "drained"
    ):
        return drained.drained_end(frame, node, func.value.id)
    if (
        isinstance(func, ast.Attribute)
        and func.attr == spelling("fill")
        and scope.field_name(frame, func.value)
    ):
        return field_fills.fill(frame, node, call, scope.field_name(frame, func.value))
    if (
        isinstance(func, ast.Attribute)
        and func.attr == spelling("zero")
        and scope.field_name(frame, func.value)
    ):
        return field_fills.zero(frame, node, call, scope.field_name(frame, func.value))
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute):
        expressions.field_use(frame, func.value)
    if (
        isinstance(func, ast.Attribute)
        and func.attr == spelling("drain")
        and expressions.construct(frame, func.value) == "stream"
    ):
        return page_streams.drain(frame, node, call)
    for kind in ("send", "forward", "consume"):
        if self_call(call, kind):
            return messages.message_call(frame, node, call, kind)
    if self_call(call, "direct_message"):
        expressions.direct_unnamed(frame, node)
    construct = expressions.construct(frame, call)
    if construct == "cxx":
        return bodies.inline(frame, node, call)
    if construct == "reduce":
        return reduce.reduce(frame, node, None, call)
    if construct == "all_reduce":
        return reduce.reduce(frame, node, None, call, everywhere=True)
    if construct == "result":
        refuse(
            frame.where,
            node,
            "a command's result word is the handler's return",
            "write `return word` last under `if self.rank == self.leader:`",
        )
    refuse(
        frame.where,
        node,
        "a call level 1 does not know",
        "write a move, or an inline body",
    )


def held_block_ends(frame, body):
    """compute's frees of the blocks products took (taken_by_field_products),
    cast in `body`, at its end."""
    ends = frame.held_blocks.pop(id(body), [])
    nodes = [
        Line(f"if (km_ok) {{\n    km::broadcast_done({w}, {w}Pages);\n}}")
        for _, w in ends
    ]
    if not nodes:
        return []
    return [Group("the end of the cast's lines: the products' blocks freed", nodes)]


def held_ends(frame, body):
    """compute's frees of the held streams made in `body` (held_stream), at its
    end; their names end with it."""
    nodes = []
    for name, node in frame.held.pop(id(body), []):
        value = frame.scope.pop(name)
        frame.ended[name] = ("lines around its stream", node.lineno)
        buffer = value.buffer.name
        nodes.append(
            Line(
                f"if (km_ok) {{\n    cb_pop_front({value.cxx}, {buffer}Block);\n}}",
                {"km_ok"},
            )
        )
    if not nodes:
        return []
    return [Group("the end of the held streams' lines: their blocks freed", nodes)]


def held_not_iterated(frame, node, iterator):
    """Refuses a loop over a held stream: it is one block, taken or iterated."""
    if isinstance(iterator, ast.Name) and (
        frame.scope.get(iterator.id, Value("")).kind == "held"
    ):
        refuse(
            frame.where,
            node,
            f"{iterator.id} is a read stream that bodies take, one block: it is "
            "not iterated too",
            f"iterate a stream of its own, `for tile in km.{spelling('stream')}"
            "(...)`, or take this one in a body",
        )


def leader_test(frame, node):
    """True when the if's test is `self.rank == self.leader` (either way round)."""
    test = node.test
    if not (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
    ):
        return False
    sides = {
        side.attr
        for side in (test.left, test.comparators[0])
        if isinstance(side, ast.Attribute)
        and isinstance(side.value, ast.Name)
        and side.value.id == "self"
    }
    return sides == {spelling("rank"), spelling("leader")}


def for_loop(frame, node):
    if node.orelse:
        refuse(frame.where, node, "a loop with an else", "write the else after it")
    iterator = node.iter
    held_not_iterated(frame, node, iterator)
    if self_call(iterator, "arrive_messages"):
        return messages.arrive_loop(frame, node)
    if (
        isinstance(iterator, ast.Name)
        and frame.scope.get(iterator.id, Value("")).kind == "arrived"
    ):
        return broadcasts.block_loop(frame, node, frame.scope[iterator.id])
    if (
        isinstance(iterator, ast.Name)
        and frame.scope.get(iterator.id, Value("")).kind == "drained"
    ):
        return drained.drain_loop(frame, node, iterator.id)
    if expressions.construct(frame, iterator) == "stream":
        return page_streams.read_loop(frame, node, iterator)
    streams = zipped(frame, node)
    if streams is not None:
        return page_streams.read_loop(frame, node, None, streams)
    if entry_tables.entries_call(
        frame, iterator
    ) is not None or entry_tables.entries_named(frame, iterator):
        return entry_tables.entries_loop(frame, node)
    if (
        isinstance(iterator, ast.Call)
        and isinstance(iterator.func, ast.Name)
        and iterator.func.id == "range"
    ):
        return range_loop(frame, node, iterator)
    refuse(
        frame.where,
        node,
        "a loop of level 1 is over a stream or a range",
        f"write `for tile in km.{spelling('stream')}(...)` or " "`for i in range(n)`",
    )


def range_loop(frame, node, call):
    if not isinstance(node.target, ast.Name):
        refuse(frame.where, node, "one loop name", "write for i in range(n)")
    name = node.target.id
    check_name(frame.where, node, name)
    if not 1 <= len(call.args) <= 2:
        refuse(frame.where, node, "range(n) or range(first, n)", "give one or two")
    first = expressions.integer(frame, call.args[0]) if len(call.args) == 2 else "0"
    end = expressions.integer(frame, call.args[-1])
    uses = set().union(*(expressions.uses(frame, arg) for arg in call.args))
    frame.scope[name] = Value("int", name)
    loop = {"name": name, "first": first, "end": end, "casts": []}
    loop["body"] = node.body  # a running total's line stands in it (carries)
    frame.loops.append(loop)
    before = set(frame.scope)
    parts = skew.split(frame, node, name)
    if parts is None:
        body_m, body_c = frame.statements(node.body)
    else:  # two rows in flight: row k's step before row k - 1's products
        body_m, body_c = [], skew.halves(frame, node, name, first, end, parts)
    body_c += loop.pop("frees", [])
    body_c += dst_tiles.carried_back(frame, loop)
    scope.scoped(frame, "loop", node, before)
    frame.loops.pop()
    frame.scope.pop(name)
    head = f"for (uint32_t {name} = {first}; {name} < {end}; {name}++) {{"
    head_c = head
    if frame.compute_waits:
        head_c = (
            f"for (uint32_t {name} = {first}; km_ok && {name} < {end}; {name}++) {{"
        )
    if parts is not None:
        k, ok = skew.skew_counter(name), "km_ok && " if frame.compute_waits else ""
        head_c = f"for (uint32_t {k} = {first}; {ok}{k} < ({end}) + 1; {k}++) {{"
    movement = [Block(head, "}", body_m, uses)] if body_m else []
    compute = [Block(head_c, "}", body_c, uses | {"km_ok"})] if body_c else []
    compute += loop.pop("after", [])  # a sum held across the loop, freed
    for broadcast in loop["casts"]:
        m, c = broadcasts.broadcast_end(frame, broadcast, end, uses)
        movement += m
        compute += c
    m, c = broadcasts.counted_ends(frame, loop)
    movement, compute = movement + m, compute + c
    movement += writes.fed_moves(frame, node, loop, body_m)
    return movement, compute


def returned(frame, node):
    """`return word` inside `if self.rank == self.leader:`: the command's result
    word, set by the leader's mover B."""
    last = frame.definition.body[-1]
    if not (
        frame.in_leader
        and isinstance(last, ast.If)
        and last.body
        and last.body[-1] is node
        and node.value is not None
    ):
        refuse(
            frame.where,
            node,
            "a command's result word is the leader's, its last line",
            f"write `return word` last under `if self.{spelling('rank')} == "
            f"self.{spelling('leader')}:`, the handler's last line",
        )
    frame.allow_words = True
    text = expressions.integer(frame, node.value)
    frame.allow_words = False
    frame.returns = True
    frame.compiled.results.add(frame.method.name)
    return [
        Line(
            f"movement.result({text});",
            expressions.uses(frame, node.value),
            mover="B",
        )
    ], []


def statement_lines(frame, node):
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
        return [], []  # a docstring
    if isinstance(node, ast.Pass):
        return [], []
    if isinstance(node, ast.Assign):
        return assign(frame, node)
    if isinstance(node, ast.AugAssign):
        return augmented(frame, node)
    if isinstance(node, ast.For):
        return for_loop(frame, node)
    if isinstance(node, ast.With):
        return drained.drained_with(frame, node)
    if isinstance(node, ast.If):
        return branch(frame, node)
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        return call_statement(frame, node, node.value)
    if isinstance(node, ast.Return):
        return returned(frame, node)
    refuse(
        frame.where,
        node,
        "not a line of a handler at level 1",
        "write a move, an int, a loop over a stream or a range, an if, or "
        f"km.{spelling('cxx')}(...)",
    )


def write_keyword(frame, call):
    """True when a stream's call has write= (a write stream's, or refused there)."""
    return any(k.arg == spelling("write_stream") for k in call.keywords)


def zipped(frame, node):
    """The streams of `for a, b in zip(km.Stream(..), km.Stream(..)):`, each with
    its name, or None when the loop is not over a zip of streams."""
    iterator = node.iter
    if not (
        isinstance(iterator, ast.Call)
        and isinstance(iterator.func, ast.Name)
        and iterator.func.id == spelling("zip")
        and iterator.args
        and all(expressions.construct(frame, arg) == "stream" for arg in iterator.args)
    ):
        return None
    target = node.target
    if (
        iterator.keywords
        or not isinstance(target, ast.Tuple)
        or len(target.elts) != len(iterator.args)
        or not all(isinstance(name, ast.Name) for name in target.elts)
    ):
        refuse(
            frame.where,
            node,
            "streams read in step name one tile each",
            f"write `for a, b in {spelling('zip')}(km.{spelling('stream')}(...), "
            f"km.{spelling('stream')}(...))`",
        )
    return [(name.id, call) for name, call in zip(target.elts, iterator.args)]
