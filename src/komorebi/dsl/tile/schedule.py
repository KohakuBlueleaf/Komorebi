"""The schedule pass of a unit program's vector code: a run of instructions reordered
so a reader of a MAD result waits behind other ready words (lower and the row forms
both run it)."""

from komorebi.dsl.tile.unit import OPS, WORK


def interleave(code, start, end):
    """Schedule: reorders code[start:end] (a run under lane flags moves as one) so a
    reader of the MAD result issued just before it waits behind another ready word;
    a constant's loads go in front of its first reader. Returns how many moved."""
    whole = code[start:end]
    loaded = {}
    for x in code:
        for v in x.writes():
            loaded[v] = loaded.get(v, True) and x.op == "SFPLOADI"
    consts = {}
    for x in whole:
        v = x.args.get("lreg_ind") if x.op == "SFPLOADI" else None
        if isinstance(v, str) and v.startswith("v") and loaded.get(v):
            consts.setdefault(v, []).append(x)
    # a constant register's setup (loads into L0, SFPCONFIG) stays whole, in front
    setup = []
    for i, x in enumerate(whole):
        if x.op == "SFPCONFIG":
            j = i
            while j > 0 and whole[j - 1].op == "SFPLOADI":
                if whole[j - 1].args["lreg_ind"] != "L0":
                    break
                j -= 1
            setup += whole[j : i + 1]
    held = {id(x) for ws in consts.values() for x in ws} | {id(x) for x in setup}
    block = [x for x in whole if id(x) not in held]
    units = _units(block)
    if units is None:
        return 0
    preds = []
    for i, unit in enumerate(units):
        mine = set()
        for j in range(i):
            if any(_depends(x, y) for x in units[j] for y in unit):
                mine.add(j)
        preds.append(mine)
    height = [1] * len(units)
    for i in range(len(units) - 1, -1, -1):
        for j in range(i + 1, len(units)):
            if i in preds[j]:
                height[i] = max(height[i], height[j] + 1)
    later = {v for x in code[end:] for v in x.reads()}
    written = {v for x in block for v in x.writes()}
    outside = {
        v
        for x in block
        for v in x.reads()
        if _work(v) and v not in written and v not in consts
    }
    outside |= {v for x in code[:start] for v in x.writes() if v in later}
    left = {}
    for x in block:
        for v in x.reads():
            left[v] = left.get(v, 0) + 1
    start_live = {v for v in outside if left.get(v) or v in later}
    # a constant loaded at its reader is live beside the estimate: a tighter bound
    for limit in range(WORK, 2, -1):
        order = _schedule(
            units, preds, height, set(start_live), dict(left), later, limit
        )
        placed, out = set(), []
        for x in (x for i in order for x in units[i]):
            for v in x.reads():
                if v in consts and v not in placed:
                    out += consts[v]
                    placed.add(v)
            out.append(x)
        unread = [w for v, ws in consts.items() if v not in placed for w in ws]
        code[start:end] = setup + unread + out
        if _pressure(code) <= WORK:
            break
    else:
        code[start:end] = whole
        return 0
    return sum(1 for k, i in enumerate(order) if k != i)


def _schedule(units, preds, height, live, left, later, limit):
    """The issue order of `units`: the tallest ready one that keeps at most `limit`
    values live, else the one leaving fewest; not a reader of the MAD just issued."""
    order, issued, previous = [], set(), None
    while len(order) < len(units):
        ready = [i for i in range(len(units)) if i not in issued and preds[i] <= issued]
        ready.sort(key=lambda i: (-height[i], i))
        after = {i: len(_unit_after(units[i], live, left, later)) for i in ready}
        fits = [i for i in ready if after[i] <= limit]
        # none fits: the one that leaves the fewest values live
        ready = fits or sorted(ready, key=lambda i: after[i])
        pick = ready[0]
        if previous is not None and OPS[previous.op].kind == "mad":
            for i in ready:
                if not set(previous.writes()) & set(units[i][0].reads()):
                    pick = i
                    break
        live = _unit_after(units[pick], live, left, later, issue=True)
        order.append(pick)
        issued.add(pick)
        previous = units[pick][-1]
    return order


def _units(block):
    """The block as units that move whole: an instruction, or a run under lane flags
    (None when a run does not close inside the block)."""
    units, i = [], 0
    while i < len(block):
        if not block[i].sets_flags():
            units.append([block[i]])
            i += 1
            continue
        depth, j = 0, i
        while True:
            if j == len(block):
                return None
            x = block[j]
            if x.sets_flags():
                depth = 0 if x.op == "SFPENCC" else depth + 1
            j += 1
            if depth == 0:
                break
        units.append(block[i:j])
        i = j
    return units


def _depends(other, insn):
    """True when `insn` must stay after `other` (registers or a Dst row)."""
    raw = set(other.writes()) & set(insn.reads())
    war = set(other.reads()) & set(insn.writes())
    waw = set(other.writes()) & set(insn.writes())
    dst = "dest_reg_addr" in other.args and "dest_reg_addr" in insn.args
    same = dst and other.args["dest_reg_addr"] == insn.args["dest_reg_addr"]
    touches = "SFPSTORE" in (other.op, insn.op)
    return bool(raw or war or waw or (same and touches))


def _unit_after(unit, live, left, later, issue=False):
    """The live work values once every instruction of `unit` has issued; with
    `issue`, `left` counts their reads off."""
    out, counts = set(live), dict(left)
    for x in unit:
        for v in x.reads():
            counts[v] -= 1
        out = _after(x, out, counts, later, issue=True)
    if issue:
        left.update(counts)
    return out


def _pressure(code):
    """The most work registers the allocator needs at one instruction of `code` (as
    `allocate` counts: a value whose last read is an instruction frees its register
    for that instruction's result)."""
    first, last = {}, {}
    for i, insn in enumerate(code):
        for v in insn.writes():
            if _work(v):
                first.setdefault(v, i)
        for v in insn.reads():
            if _work(v):
                last[v] = i
    starts, ends = [0] * (len(code) + 2), [0] * (len(code) + 2)
    for v, i in first.items():
        starts[i] += 1
        end = last.get(v, i)
        if end == i or v in code[end].writes():  # written there: held through it
            end += 1
        ends[end] += 1
    most = live = 0
    for i in range(len(code)):
        live += starts[i] - ends[i]
        most = max(most, live)
    return most


def _work(value):
    """True for a value that takes a work register (not a fixed or constant one)."""
    return isinstance(value, str) and value.startswith("v")


def _after(insn, live, left, later, issue=False):
    """The live work values once `insn` has issued (`left`: reads still to come in
    the block, before this one's when not `issue`)."""
    out = set(live)
    for v in insn.reads():
        rest = left.get(v, 0) - (0 if issue else 1)
        if rest <= 0 and v not in later:
            out.discard(v)
    out |= {v for v in insn.writes() if _work(v)}
    return out
