"""The C++ of one kernel as a tree of nodes, before rendering: glue (a definition kept
only while a kept line uses it), lines, blocks and the groups of one Python line.
A node may belong to one mover (`mover` "A" or "B"); rendering puts it under #if.
"""

from dataclasses import dataclass, field

INDENT = "    "


@dataclass
class Glue:
    """A definition: kept when a kept node uses a name it defines."""

    text: str
    defines: set
    uses: set = field(default_factory=set)
    mover: str | None = None


@dataclass
class Line:
    """A statement: always kept."""

    text: str
    uses: set = field(default_factory=set)
    mover: str | None = None


@dataclass
class Block:
    """`opening`, the children indented, `closing`; kept when a child line is."""

    opening: str
    closing: str
    children: list = field(default_factory=list)
    uses: set = field(default_factory=set)
    mover: str | None = None


@dataclass
class Group:
    """The nodes of one Python line, under its comment; kept when a node is."""

    comment: str
    children: list = field(default_factory=list)
    mover: str | None = None


def has_line(node):
    """True when the node is or holds a Line."""
    if isinstance(node, Line):
        return True
    if isinstance(node, Block | Group):
        return any(has_line(child) for child in node.children)
    return False


def without_empty(nodes):
    """The nodes without blocks that hold no line (glue alone is no line); a group is
    looked through, its glue left to the glue's own pruning."""
    kept = []
    for node in nodes:
        if isinstance(node, Block) and not has_line(node):
            continue
        if isinstance(node, Block | Group):
            node.children = without_empty(node.children)
        kept.append(node)
    return kept


def walk(nodes):
    """Every node of the tree."""
    for node in nodes:
        yield node
        if isinstance(node, Block | Group):
            yield from walk(node.children)


def prune(nodes):
    """The tree with only the glue whose names a kept node uses (to a fixed point)."""
    nodes = without_empty(nodes)
    used = set()
    for node in walk(nodes):
        if isinstance(node, Line | Block):
            used |= node.uses
    glue = [node for node in walk(nodes) if isinstance(node, Glue)]
    kept = set()
    changed = True
    while changed:
        changed = False
        for item in glue:
            if id(item) not in kept and item.defines & used:
                kept.add(id(item))
                used |= item.uses
                changed = True
    return drop_glue(nodes, kept)


def drop_glue(nodes, kept):
    """The tree without the glue not in `kept` (ids), and without emptied groups."""
    out = []
    for node in nodes:
        if isinstance(node, Glue) and id(node) not in kept:
            continue
        if isinstance(node, Block | Group):
            node.children = drop_glue(node.children, kept)
            if isinstance(node, Group) and not node.children:
                continue
        out.append(node)
    return out


def render(nodes, depth):
    """The lines of the tree at `depth` indents; consecutive nodes of one mover go
    under one #if (mover A: COMPILE_FOR_BRISC), A then B under #if and #else; a
    third run (A, B, A) opens a new #if."""
    lines, current, flipped = [], None, False
    for node in nodes:
        mover = node.mover
        if mover != current:
            if current is not None and mover is not None and not flipped:
                lines.append(
                    "#else  // mover B" if mover == "B" else "#else  // mover A"
                )
                flipped = True
            else:
                if current is not None:
                    lines.append("#endif")
                if mover == "A":
                    lines.append("#if defined(COMPILE_FOR_BRISC)  // mover A")
                elif mover == "B":
                    lines.append("#if !defined(COMPILE_FOR_BRISC)  // mover B")
                flipped = False
            current = mover
        lines += render_node(node, depth)
    if current is not None:
        lines.append("#endif")
    return lines


def render_node(node, depth):
    pad = INDENT * depth
    if isinstance(node, Glue | Line):
        return [pad + text if text else "" for text in node.text.split("\n")]
    if isinstance(node, Group):
        head = [pad + "// " + node.comment] if node.comment else []
        return head + render(node.children, depth)
    return (
        [pad + node.opening] + render(node.children, depth + 1) + [pad + node.closing]
    )
