"""A tensor's padding through a tile function: which values' padding holds
zeros (KEEPS_ZERO), and the reductions and products along a padded axis that would
take it in (a max or a min always; a sum or a product when it may not be zero)."""

from dataclasses import dataclass

from komorebi import formats

# Zero padding out of zero padding in: "all" operands', "any" one's, "one" its own;
# an operation not named (exp, `/`, a per-row `.col()`, a reduction ...) does not.
KEEPS_ZERO = {
    "add": "all",
    "sub": "all",
    "max": "all",
    "min": "all",
    "matmul": "all",
    "mul": "any",
    "neg": "one",
    "abs": "one",
    "relu": "one",
    "square": "one",
    "silu": "one",
    "tanh": "one",
    "erf": "one",
    "sqrt": "one",
    "cast": "one",
    "row": "one",  # a field's row down the rows: its own lanes
}


@dataclass(frozen=True)
class Side:
    """A tensor a tile function's parameter reads: its name, its shape before the
    padding and after, whether its columns or rows are padded, whether its padding
    holds zeros."""

    name: str
    logical: tuple
    padded: tuple
    cols: bool
    rows: bool
    clean: bool


def side(name, tensor):
    """The Side of `tensor` (a Tensor or an IslandTensor), or None in whole tiles."""
    logical = getattr(tensor, "logical", None)
    if logical is None:
        return None
    # an island session's empty tensor keeps the shape it was given: padded here
    padded = formats.padded_shape(tensor.shape, tensor.layout)
    return Side(
        name,
        tuple(logical),
        padded,
        logical[-1] != padded[-1],
        logical[-2] != padded[-2],
        getattr(tensor, "padding", "dirty") == "clean",
    )


class Padding:
    """The padding of each value of a traced tile function, its parameters read from
    `sides` ({parameter: Side or None})."""

    def __init__(self, sides):
        self.sides = sides
        self.zero, self.cols, self.rows = {}, {}, {}

    def _sides(self, node):
        if node.kind in ("load", "resident"):
            found = self.sides.get(node.attrs["view"].tensor.name)
            return [found] if found is not None else []
        return [s for a in node.args for s in self._sides_of(a)]

    def _sides_of(self, node):
        if id(node) not in self.cols:
            found = self._sides(node)
            self.cols[id(node)] = [s for s in found if s.cols]
            self.rows[id(node)] = [s for s in found if s.rows]
        return self.cols[id(node)] + self.rows[id(node)]

    def padded_cols(self, node):
        """The column-padded tensors `node` reads."""
        self._sides_of(node)
        return self.cols[id(node)]

    def padded_rows(self, node):
        """The row-padded tensors `node` reads."""
        self._sides_of(node)
        return self.rows[id(node)]

    def zero_padding(self, node):
        """Whether `node`'s padding lanes hold zeros."""
        if id(node) not in self.zero:
            self.zero[id(node)] = self._zero(node)
        return self.zero[id(node)]

    def _zero(self, node):
        if node.kind == "resident":  # a field: its declared lanes (a mask's zeros)
            return True
        if node.kind == "load":  # a tensor in whole tiles: its data
            found = self.sides.get(node.attrs["view"].tensor.name)
            return found is not None and found.clean
        if node.kind == "const":
            return node.attrs["value"] == 0.0
        rule = KEEPS_ZERO.get(node.kind)
        if rule == "any":
            return any(self.zero_padding(a) for a in node.args)
        if rule in ("all", "one"):
            return all(self.zero_padding(a) for a in node.args)
        return False


def problems(ir, sides):
    """[(line, what, the tensors)] of each reduction or product of `ir` that reads
    padding along its axis: a max or a min along a padded row (any padding), a sum
    along it or a product along a padded K whose padding may not be zero."""
    pad, found = Padding(sides), []
    for node in ir.nodes():
        if node.kind == "reduce":
            x = node.args[0]
            padded = pad.padded_cols(x)
            if padded and node.attrs["op"] in ("max", "min"):
                found.append((node.line, f"reduce_{node.attrs['op']}", padded))
            elif padded and not pad.zero_padding(x):
                found.append((node.line, "reduce_sum", padded))
        elif node.kind == "matmul":
            a, w = node.args
            ka, kw = pad.padded_cols(a), pad.padded_rows(w)
            held = (ka and pad.zero_padding(a)) or (kw and pad.zero_padding(w))
            if (ka or kw) and not held:
                found.append((node.line, "a product `@`", ka + kw))
    return found


def keeps_zero(ir, sides):
    """Whether every value `ir` stores has zeros in its padding."""
    pad = Padding(sides)
    return all(pad.zero_padding(store.value) for store in ir.stores)
