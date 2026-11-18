"""A class's command checked at submit: each tensor.pages(first, count) its handler's
lines name, against the tensor's pages, for the ranks that run the line, before
anything is sent. A range the lines do not tell (a name the card computes, a line
under an if) is left to the card's KM_GUARD check."""

import ast
import operator
from dataclasses import dataclass

from komorebi.dsl.chain import chain_calls
from komorebi.dsl.compile.source import source_of
from komorebi.dsl.front.names import spelling

CACHED = 4096  # commands remembered as checked, by their words and group
BENT = (ast.Mod, ast.FloorDiv, ast.Call)  # no sum of multiples: every value checked
EVERY = 4096  # the most values of a bent line's loops checked; past them the card's


@dataclass
class Range:
    """tensor.pages(first, count) on a line: the range loops around it, and whether
    rank 0 alone runs it (under `if self.rank == self.leader:`)."""

    tensor: str
    line: int
    first: ast.expr
    count: ast.expr
    over: tuple
    leader: bool


class Ranges(ast.NodeVisitor):
    """The page ranges of a handler's tensors, and the names given once (its lines'
    own values, as the chain's evaluator reads them)."""

    def __init__(self, method):
        self.tensors = set(method.places())  # those given a tensor at the check
        self.ranges, self.over, self.leader, self.guarded = [], [], False, 0
        self.lines = chain_calls.Lines()
        self.given = {}  # name -> times assigned

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.given[target.id] = self.given.get(target.id, 0) + 1
                self.lines.names[target.id] = node.value
        self.generic_visit(node)

    def visit_For(self, node):
        loop = (
            isinstance(node.target, ast.Name)
            and isinstance(node.iter, ast.Call)
            and getattr(node.iter.func, "id", None) == "range"
        )
        if isinstance(node.target, ast.Name):
            self.lines.loops.add(node.target.id)
        self.visit(node.iter)
        self.over.append((node.target.id, tuple(node.iter.args)) if loop else None)
        for statement in node.body + node.orelse:
            self.visit(statement)
        self.over.pop()

    def visit_While(self, node):
        self.over.append(None)  # its turns: the card's
        self.generic_visit(node)
        self.over.pop()

    def visit_If(self, node):
        self.visit(node.test)
        sides = [getattr(node.test, "left", None)]
        sides += getattr(node.test, "comparators", [])
        names = sorted(str(getattr(side, "attr", None)) for side in sides)
        leader = names == ["leader", "rank"] and not node.orelse
        before = self.leader
        if leader:
            self.leader = True
        else:
            self.guarded += 1
        for statement in node.body + node.orelse:
            self.visit(statement)
        if leader:
            self.leader = before
        else:
            self.guarded -= 1

    def visit_Call(self, node):
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == spelling("pages")
            and isinstance(func.value, ast.Name)
            and func.value.id in self.tensors
            and len(node.args) == 2
            and not self.guarded
            and None not in self.over
        ):
            self.ranges.append(
                Range(
                    func.value.id,
                    node.lineno,
                    node.args[0],
                    node.args[1],
                    tuple(self.over),
                    self.leader,
                )
            )
        self.generic_visit(node)


def ranges_of(method):
    """(the handler's page ranges, its lines' names) (cached on it)."""
    cached = getattr(method, "page_ranges", None)
    if cached is None:
        definition, _, _ = source_of(method.function)
        reader = Ranges(method)
        for statement in definition.body:
            reader.visit(statement)
        for name, times in reader.given.items():
            if times > 1:  # given again: its value is the card's
                reader.lines.names.pop(name, None)
        cached = (reader.ranges, reader.lines)
        method.page_ranges = cached
    return cached


def _ranks(found, size, bent):
    """The ranks whose range bounds the line's: rank 0 under a leader's if; every rank
    for a bent line; else the first and the last (a sum of multiples of the rank is
    largest or smallest at one of them)."""
    if found.leader:
        return [0]
    return list(range(size)) if bent else sorted({0, size - 1})


def _bindings(evaluator, over, bent):
    """The range loops around a line bound to each value (a bent line) or to their
    first and last; None: too many values (left to the card)."""
    bindings = [{}]
    for name, arguments in over:
        span = range(*[evaluator.value(a) for a in arguments])
        values = span if bent else ({span[0], span[-1]} if span else ())
        bindings = [{**b, name: v} for b in bindings for v in values]
        if len(bindings) > EVERY:
            return None
    return bindings


def problem(method, values, group, program):
    """Why a command of `method` with `values` on `group` reads or writes past a
    tensor's last page (None: it does not, or its lines do not tell)."""
    found_ranges, lines = ranges_of(method)
    given = dict(zip(method.places(), values))
    size = group[1]
    for found in found_ranges:
        try:  # a tensor's pages (a number the card's tensor holds), else no tensor
            pages = operator.index(getattr(given.get(found.tensor), "pages", None))
        except TypeError:
            continue
        bent = any(
            isinstance(node, BENT)
            for side in (found.first, found.count)
            for node in ast.walk(side)
        )
        try:
            for rank in _ranks(found, size, bent):
                evaluator = chain_calls.Evaluator(
                    method, lines, given, program, rank, size
                )
                for binding in _bindings(evaluator, found.over, bent) or ():
                    evaluator.bound = dict(binding)
                    first = evaluator.value(found.first)
                    count = evaluator.value(found.count)
                    if count > 0 and (first < 0 or first + count > pages):
                        at = ", ".join(f"{k} {v}" for k, v in binding.items())
                        return (
                            f"{type(program).__name__}.{method.name}, line "
                            f"{found.line}: pages {first} to {first + count - 1} of "
                            f"{found.tensor} (its pages: 0 to {pages - 1}) on "
                            f"{spelling('rank')} {rank}" + (f" at {at}" if at else "")
                        )
        except (chain_calls.Undescribed, ZeroDivisionError):
            continue  # the card's KM_GUARD check
    return None


class Checked:
    """A run's commands of class handlers checked once for each words and group."""

    def __init__(self):
        self._seen = set()

    def check(self, method, values, words, group, program):
        key = (id(method), tuple(int(w) for w in words), tuple(group))
        if key in self._seen:
            return
        found = problem(method, values, group, program)
        if found is not None:
            raise ValueError(f"{found}: refused before it runs")
        if len(self._seen) >= CACHED:
            self._seen.clear()
        self._seen.add(key)
