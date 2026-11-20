"""A block (32 tokens x 5,120, bf16, 160 pages) down a breadth-first tree of the islands
into each island's copy, and the islands' blocks up the same tree, added on Tensix at
each island (bf16 a hop), into the home's. Hand kernels (kernels/, the runtime's form),
S cores an island; the tree comes in the command (a parent an island). README.md."""

from collections import deque
from pathlib import Path

import komorebi as km

KERNELS = Path(__file__).resolve().parent / "kernels"
C = km.abi.constants(KERNELS / "tree.h", known=km.abi.headers().constants)
SYNC, DOWN, GO, UP, ACK = 1, 2, 4, 8, 16
NONE = 31
TILES = 160


def bfs_tree(graph, count, home):
    """{island: parent} of the breadth-first tree from `home` (the home its own parent);
    a node's parent the lowest-numbered neighbour one hop nearer."""
    parent, queue = {home: home}, deque([home])
    while queue:
        island = queue.popleft()
        for other in sorted(graph.neighbours(island)):
            if other not in parent and other < count:
                parent[other] = island
                queue.append(other)
    return parent


def pruned(parent, home, wanted):
    """The part of the tree that reaches the islands `wanted` (and the home)."""
    keep = {home: home}
    for island in wanted:
        while island not in keep:
            keep[island] = parent[island]
            island = parent[island]
    return keep


def children_most(parent, home):
    kids = {}
    for island, p in parent.items():
        if island != home:
            kids[p] = kids.get(p, 0) + 1
    return max([1, *kids.values()])


def depth(parent, home):
    def d(i):
        return 0 if i == home else d(parent[i]) + 1

    return max(d(i) for i in parent)


def words(parent, home, phase, senders, a7, a8):
    table = [NONE] * 36
    for island, p in parent.items():
        table[island] = p
    out = [home | phase << 5 | senders << 12]
    out += [sum(table[6 * k + j] << (5 * j) for j in range(6)) for k in range(6)]
    return out + [a7, a8]


def program(senders, kids, sample):
    """The tree's program, the dict's of the piece's one load: `senders` cores an
    island, `kids` child slots a core; `sample` a block tensor (its accessor's
    arguments)."""
    n = TILES // senders
    return km.Program(
        "moe_tree",
        kernels=KERNELS,
        buffers=[
            km.Buffer(C["kCbOut"], "bfloat16", C["kPageBytes"], n),
            km.Buffer(C["kCbDown"], "bfloat16", C["kPageBytes"], n),
            km.Buffer(C["kCbUp"], "bfloat16", C["kPageBytes"], kids * n),
            km.Buffer(C["kCbOwn"], "bfloat16", C["kPageBytes"], n),
            km.Buffer(C["kCbSum"], "bfloat16", C["kPageBytes"], n),
            km.Buffer(C["kCbTiny"], "bfloat16", C["kPageBytes"], 1),
            km.Buffer(C["kCbStamp"], "uint32", 64, 1),
            km.Buffer(C["kCbX"], "bfloat16", C["kPageBytes"], n),
            km.Buffer(C["kCbUpTiny"], "bfloat16", C["kPageBytes"], kids),
        ],
        compile_args={"reader": sample.accessor_args()},
        compute={
            "math_fidelity": "HiFi2",
            "fp32_dest_acc_en": True,
            "dst_full_sync_en": True,
        },
        defines={"TREE_PAGES": n, "TREE_KIDS": kids},
        handlers=KERNELS / "tree.h",
    )


def go(session, run, tree, parent, home, phase, senders, a7=0, a8=0, timeout=30.0):
    """One command of the program `tree` over every rank (islands outside `parent`
    return at once); its card time (us)."""
    ws = words(parent, home, phase, senders, a7, a8)
    handle = run.island(0).submit(tree.handler("tree"), (0, session.ranks), ws)
    (event,) = session.wait_together([(0, handle)], timeout=timeout)
    if event.kind != "done":
        raise km.KomorebiError(f"the tree, phase {phase}: {event}")
    return event.microseconds
