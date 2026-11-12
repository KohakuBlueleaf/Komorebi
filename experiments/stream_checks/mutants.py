"""Mutation run: every plausible slip in the simulator or the check must be caught.

Each mutant is a text patch of `stream.py`, run on the saved reference streams; one must
then give another answer. It lists the survivors and exits non-zero if there is one.
"""

# ruff: noqa: E501
# Patch strings are source text and keep the line breaks of the code they patch.

import json
import struct
import sys
import types
from pathlib import Path

import stream_reference as ref

from komorebi.engines import stream as st

SOURCE = Path(st.__file__).read_text()
SLOT_LOOP = "        for slot, busy in enumerate(self.table):\n"
LATE = "                and self.now >= busy.deadline_pass\n"
TEAM = "        if team_outside(first, count, self.n_cores):\n"
SUBMIT_COUNT = "        if count == 0 or count > self.n_cores:\n"
QUEUE_FULL = '        if len(self.queue) >= QUEUE_JOBS:\n            return "stall"\n'

MUTANTS = {
    # queue window and overtaking limit
    "window_15": [("QUEUE_LOOK = 16", "QUEUE_LOOK = 15")],
    "window_17": [("QUEUE_LOOK = 16", "QUEUE_LOOK = 17")],
    "window_1": [("QUEUE_LOOK = 16", "QUEUE_LOOK = 1")],
    "window_refills": [
        ("for job in list(self.queue[:QUEUE_LOOK]):", "for job in self.live_window():"),
        (
            "    def step_queue(self):",
            "    def live_window(self):\n"
            "        for i in range(QUEUE_LOOK):\n"
            "            if i < len(self.queue):\n"
            "                yield self.queue[i]\n\n"
            "    def step_queue(self):",
        ),
    ],
    "overtake_1023": [("QUEUE_PASSES = 1024", "QUEUE_PASSES = 1023")],
    "overtake_1025": [("QUEUE_PASSES = 1024", "QUEUE_PASSES = 1025")],
    "overtake_never_limits": [("QUEUE_PASSES = 1024", "QUEUE_PASSES = 10**9")],
    "overtake_counted_per_pass": [
        (
            "            for j in skipped:\n                j.passed += 1\n",
            "            for j in skipped:\n"
            "                if j.last_pass != self.now:\n"
            "                    j.passed += 1\n"
            "                    j.last_pass = self.now\n",
        ),
        ("    passed: int = 0\n", "    passed: int = 0\n    last_pass: int = -1\n"),
    ],
    "overtake_limit_exclusive": [
        ("j.passed >= QUEUE_PASSES", "j.passed > QUEUE_PASSES")
    ],
    "queue_capacity_63": [("QUEUE_JOBS = 64", "QUEUE_JOBS = 63")],
    "queue_capacity_65": [("QUEUE_JOBS = 64", "QUEUE_JOBS = 65")],
    # allocator and slots
    "alloc_last_fit": [
        (
            "    for first in range(n_cores - count + 1):",
            "    for first in range(n_cores - count, -1, -1):",
        )
    ],
    "slot_highest_free": [
        (
            "return next((i for i, b in enumerate(self.table) if b is None), None)",
            "return next((i for i in range(15, -1, -1) if self.table[i] is None), None)",
        )
    ],
    "sweep_highest_slot_first": [
        (
            SLOT_LOOP + "            if busy and busy.done_pass",
            "        for slot, busy in reversed(list(enumerate(self.table))):\n"
            "            if busy and busy.done_pass",
        )
    ],
    "only_15_slots": [("RING_SLOTS = 16", "RING_SLOTS = 15")],
    "alloc_ignores_touching": [
        ("first + count <= f or f + c <= first", "first + count < f or f + c < first")
    ],
    # issuer turns
    "turns_stream_first": [("start = p % len(turns)", "start = 0")],
    "turns_queue_first": [("start = p % len(turns)", "start = 1")],
    "turns_offset_by_one": [("start = p % len(turns)", "start = (p + 1) % len(turns)")],
    # micro-ops per pass and the step limit
    "step_ops_63": [("STEP_OPS = 64", "STEP_OPS = 63")],
    "step_ops_65": [("STEP_OPS = 64", "STEP_OPS = 65")],
    "step_ops_128": [("STEP_OPS = 64", "STEP_OPS = 128")],
    "stall_counts_a_step": [
        (
            '                if state != "ok":\n                    return state\n',
            '                if state != "ok":\n'
            '                    self.steps += state == "stall"\n'
            "                    return state\n",
        )
    ],
    "step_limit_one_late": [
        ('self.steps >= self.h["max_steps"]', 'self.steps > self.h["max_steps"]')
    ],
    "step_limit_before_end_of_program": [
        (
            "            if self.pc >= len(self.ops):\n"
            "                return self.fail(ERR_JUMP, self.pc)\n"
            '            if self.steps >= self.h["max_steps"]:\n'
            "                return self.fail(ERR_STEPS, self.pc)\n",
            '            if self.steps >= self.h["max_steps"]:\n'
            "                return self.fail(ERR_STEPS, self.pc)\n"
            "            if self.pc >= len(self.ops):\n"
            "                return self.fail(ERR_JUMP, self.pc)\n",
        )
    ],
    # deadlines, cancel, orphans
    "deadline_one_pass_late": [
        (LATE, "                and self.now > busy.deadline_pass\n")
    ],
    "deadline_before_sweep": [
        (
            "        self.sweep()\n        self.check_deadlines()\n        if cancel",
            "        self.check_deadlines()\n        self.sweep()\n        if cancel",
        )
    ],
    "deadline_names_highest_slot": [
        (
            SLOT_LOOP
            + "            if (\n                busy\n                and busy.deadline_pass",
            "        for slot, busy in reversed(list(enumerate(self.table))):\n"
            "            if (\n                busy\n                and busy.deadline_pass",
        )
    ],
    "cancel_before_deadline": [
        (
            "        self.check_deadlines()\n        if cancel and self.error is None:\n"
            "            self.fail(ERR_CANCELLED, self.pc)\n",
            "        if cancel and self.error is None:\n"
            "            self.fail(ERR_CANCELLED, self.pc)\n"
            "        self.check_deadlines()\n",
        )
    ],
    "cancel_pc_zero": [
        ("self.fail(ERR_CANCELLED, self.pc)", "self.fail(ERR_CANCELLED, 0)")
    ],
    "orphans_not_drained": [
        ("                sim.drain(p + 1, max_passes)\n", "                pass\n")
    ],
    "orphans_have_no_deadline": [
        (
            "                else:\n                    self.post(EV_TIMEOUT, slot)\n",
            "                else:\n                    self.stopped = False\n",
        )
    ],
    "orphan_timeout_names_pc": [
        ("self.post(EV_TIMEOUT, slot)", "self.post(EV_TIMEOUT, busy.record.pc)")
    ],
    "orphan_event_has_no_req": [
        (
            "                    self.post(EV_TIMEOUT, slot)\n",
            "                    self.events.append(Event(EV_TIMEOUT, 0, slot, self.now))\n",
        )
    ],
    "deadline_does_not_stop_the_grid": [("                self.stopped = True\n", "")],
    "default_deadline_ignored": [
        (
            'if deadline_passes is None and h["deadline_us"] == 0:',
            'if False and deadline_passes is None and h["deadline_us"] == 0:',
        )
    ],
    "default_deadline_for_every_header": [
        (
            'if deadline_passes is None and h["deadline_us"] == 0:',
            "if deadline_passes is None:",
        )
    ],
    # failure checks: which first
    "team_check_after_slot_wait": [
        (
            TEAM + "            return self.fail(ERR_TEAM, self.pc)\n"
            "        slot = self.free_slot()\n"
            '        if slot is None:\n            return "stall"\n',
            "        slot = self.free_slot()\n"
            '        if slot is None:\n            return "stall"\n'
            + TEAM
            + "            return self.fail(ERR_TEAM, self.pc)\n",
        )
    ],
    "count_check_after_queue_full": [
        (
            SUBMIT_COUNT
            + "            return self.fail(ERR_NEVER_FITS, self.pc)\n"
            + QUEUE_FULL,
            QUEUE_FULL
            + SUBMIT_COUNT
            + "            return self.fail(ERR_NEVER_FITS, self.pc)\n",
        )
    ],
    "team_sum_wraps": [
        (
            "    return count == 0 or first > n_cores or count > n_cores - first",
            "    return count == 0 or (first + count) & MASK > n_cores",
        )
    ],
    "team_off_by_one_count": [("count > n_cores - first", "count >= n_cores - first")],
    "team_empty_allowed": [("return count == 0 or first", "return first")],
    "job_of_zero_allowed": [(SUBMIT_COUNT, "        if count > self.n_cores:\n")],
    "job_of_whole_grid_refused": [
        (SUBMIT_COUNT, "        if count == 0 or count >= self.n_cores:\n")
    ],
    # micro-op semantics
    "jlt_signed": [
        (
            "nxt = o.imm if regs[o.a] < regs[o.b] else nxt",
            "nxt = o.imm if regs[o.a] ^ 0x80000000 < regs[o.b] ^ 0x80000000 else nxt",
        )
    ],
    "mul_not_wrapped": [
        (
            "regs[o.a] = regs[o.b] * regs[o.c] & MASK",
            "regs[o.a] = regs[o.b] * regs[o.c]",
        )
    ],
    "shr_amount_unmasked": [("regs[o.b] >> (o.imm & 31)", "regs[o.b] >> o.imm")],
    "shr_arithmetic": [
        (
            "regs[o.a] = regs[o.b] >> (o.imm & 31)",
            "regs[o.a] = ((regs[o.b] ^ 0x80000000) - 0x80000000 >> (o.imm & 31)) & MASK",
        )
    ],
    "mod_signed": [
        (
            "regs[o.a] = regs[o.b] % regs[o.c]",
            "regs[o.a] = ((regs[o.b] ^ 0x80000000) - 0x80000000) % regs[o.c] & MASK",
        )
    ],
    "sub_not_wrapped": [
        (
            "regs[o.a] = regs[o.b] - regs[o.c] & MASK",
            "regs[o.a] = regs[o.b] - regs[o.c]",
        )
    ],
    "addi_not_wrapped": [
        ("regs[o.a] = regs[o.b] + o.imm & MASK", "regs[o.a] = regs[o.b] + o.imm")
    ],
    "add_not_wrapped": [
        (
            "regs[o.a] = regs[o.b] + regs[o.c] & MASK",
            "regs[o.a] = regs[o.b] + regs[o.c]",
        )
    ],
    "and_is_or": [
        ("regs[o.a] = regs[o.b] & regs[o.c]", "regs[o.a] = regs[o.b] | regs[o.c]")
    ],
    "jeq_is_jne": [
        (
            "nxt = o.imm if regs[o.a] == regs[o.b] else nxt",
            "nxt = o.imm if regs[o.a] != regs[o.b] else nxt",
        )
    ],
    "jnz_is_jz": [
        (
            "nxt = o.imm if regs[o.a] != 0 else nxt",
            "nxt = o.imm if regs[o.a] == 0 else nxt",
        )
    ],
    "rand_leaves_seed": [
        (
            "                regs[SEED_REG] = xorshift32(regs[SEED_REG])\n"
            "                regs[o.a] = regs[SEED_REG]\n",
            "                regs[o.a] = xorshift32(regs[SEED_REG])\n",
        )
    ],
    "rand_wrong_shift": [("x ^= x << 5 & MASK", "x ^= x << 7 & MASK")],
    "mod_zero_gives_zero": [
        (
            "                if regs[o.c] == 0:\n                    return self.fail(ERR_MOD_ZERO, self.pc)\n"
            "                regs[o.a] = regs[o.b] % regs[o.c]\n",
            "                regs[o.a] = regs[o.b] % regs[o.c] if regs[o.c] else 0\n",
        )
    ],
    "event_aux_not_masked": [
        (
            "self.post(EV_STREAM, regs[o.a] & 0xFFFFFF)",
            "self.post(EV_STREAM, regs[o.a])",
        )
    ],
    "end_aux_not_masked": [
        (
            "self.post(EV_STREAM_END, self.halt & 0xFFFFFF)",
            "self.post(EV_STREAM_END, self.halt)",
        )
    ],
    # waiting, results, templates
    "pending_forgets_queued_jobs": [
        (
            "        self.queue.append(job)\n        self.pending += 1\n",
            "        self.queue.append(job)\n",
        )
    ],
    "halt_ignores_queued_jobs": [
        (
            '            if self.pending:\n                return "stall"\n            self.halt',
            '            if self.pending - len(self.queue):\n                return "stall"\n            self.halt',
        )
    ],
    "halt_does_not_wait": [
        (
            '            if self.pending:\n                return "stall"\n            self.halt',
            "            self.halt",
        )
    ],
    "wait_is_off_by_one": [("self.pending > o.imm", "self.pending >= o.imm")],
    "ack_in_the_same_pass": [
        (
            "done = None if delay is None else self.now + 1 + delay",
            "done = None if delay is None else self.now + delay",
        )
    ],
    "result_starts_nonzero": [
        (
            "self.pc = self.steps = self.seq = self.pending = self.last_result = 0",
            "self.pc = self.steps = self.seq = self.pending = 0\n        self.last_result = 7",
        )
    ],
    "result_of_unflagged_records": [
        (
            "                if busy.record.flags & F_RESULT:\n                    self.last_result = busy.result\n",
            "                self.last_result = busy.result\n",
        )
    ],
    "excl_ignored": [
        ("        if t.flags & F_EXCL and any(", "        if False and any(")
    ],
    "excl_touching_teams_wait": [
        (
            "first < f + c and f < first + count for f, c in self.busy_teams()",
            "first <= f + c and f <= first + count for f, c in self.busy_teams()",
        )
    ],
    "queue_flag_kept": [
        (
            "flags = words[1] >> 8 & 0xFF & ~F_QUEUE",
            "flags = words[1] >> 8 & 0xFF | (F_QUEUE if queued else 0)",
        )
    ],
    "earlier_patch_wins": [
        (
            "        for i in range(t.n_patch):\n            words[ARG_WORD[t.arg[i]]]",
            "        for i in reversed(range(t.n_patch)):\n            words[ARG_WORD[t.arg[i]]]",
        )
    ],
    "patch_list_ignores_n_patch": [
        (
            "        for i in range(t.n_patch):\n            words[ARG_WORD[t.arg[i]]]",
            "        for i in range(PATCHES):\n            words[ARG_WORD[t.arg[i]]]",
        )
    ],
    "queued_job_reads_registers_at_issue": [
        (
            "job = _Job(self.patched(self.templates[o.imm]), count, self.pc)",
            "job = _Job(self.templates[o.imm], count, self.pc)",
        ),
        (
            "self.put(slot, job.words, first",
            "self.put(slot, self.patched(job.words), first",
        ),
    ],
    # the check
    "check_ignores_version": [('or h["version"] != VERSION', "")],
    "check_ignores_magic": [('h["magic"] != MAGIC or ', "")],
    "check_accepts_zero_seed": [(' or h["seed"] == 0', "")],
    "check_accepts_no_micro_ops": [
        ('1 <= h["n_ops"] <= MAX_OPS', 'h["n_ops"] <= MAX_OPS')
    ],
    "check_ignores_n_ops_limit": [('1 <= h["n_ops"] <= MAX_OPS', '1 <= h["n_ops"]')],
    "check_ignores_n_records_limit": [(' or h["n_records"] > MAX_OPS', "")],
    "check_accepts_zero_max_steps": [('h["max_steps"] == 0 or ', "")],
    "check_accepts_longer_uploads": [
        (" or len(data) != _size(h)", " or len(data) < _size(h)")
    ],
    "check_ignores_the_region": [("or len(data) > region_size ", "")],
    "check_region_off_by_one": [
        ("len(data) > region_size", "len(data) >= region_size")
    ],
    "check_ignores_unknown_opcodes": [
        (
            "        if o.op not in OPS:\n            return StreamError(ERR_OPCODE, pc)\n",
            "        if o.op not in OPS:\n            continue\n",
        )
    ],
    "check_looks_at_unused_registers": [
        (
            "        if any(getattr(o, f) >= REGS for f in fields):",
            "        if any(getattr(o, f) >= REGS for f in 'abc'):",
        )
    ],
    "check_ignores_register_c": [
        ("for f in fields", "for f in fields.replace('c', '')")
    ],
    "check_jump_target_equal_allowed": [('o.imm >= h["n_ops"]', 'o.imm > h["n_ops"]')],
    "check_template_index_equal_allowed": [
        ('o.imm >= h["n_records"]', 'o.imm > h["n_records"]')
    ],
    "check_templates_before_micro_ops": [
        (
            "    first_use = {}\n    for pc, o in enumerate(ops):",
            "    first_use = {}\n    for index, t in enumerate(templates):\n"
            "        if _bad_template(t):\n            return StreamError(ERR_TEMPLATE, 0)\n"
            "    for pc, o in enumerate(ops):",
        )
    ],
    "check_template_pc_zero": [("first_use.get(index, 0)", "0")],
    "check_template_pc_is_last_user": [
        ("first_use.setdefault(o.imm, pc)", "first_use[o.imm] = pc")
    ],
    "check_ignores_segment_zero": [
        ("not 1 <= t.seg <= SEG_USER_MAX", "t.seg > SEG_USER_MAX")
    ],
    "check_allows_host_segments": [
        ("not 1 <= t.seg <= SEG_USER_MAX", "not 1 <= t.seg <= 255")
    ],
    "check_ignores_flag_bits": [(" or t.flags & ~(F_EXCL | F_RESULT)", "")],
    "check_allows_queue_flag": [
        ("~(F_EXCL | F_RESULT)", "~(F_EXCL | F_RESULT | F_QUEUE)")
    ],
    "check_five_patches_allowed": [("t.n_patch > PATCHES", "t.n_patch > PATCHES + 1")],
    "check_ignores_patch_argument": [("t.arg[i] >= ARGS or ", "")],
    "check_ignores_patch_register": [(" or t.reg[i] >= REGS", "")],
    "check_checks_unused_patches": [
        (
            "any(t.arg[i] >= ARGS or t.reg[i] >= REGS for i in range(t.n_patch))",
            "any(t.arg[i] >= ARGS or t.reg[i] >= REGS for i in range(PATCHES))",
        )
    ],
    "check_ignores_first_reg": [
        ("for r in (t.first_reg, t.count_reg)", "for r in (t.count_reg,)")
    ],
    "check_ignores_count_reg": [
        ("for r in (t.first_reg, t.count_reg)", "for r in (t.first_reg,)")
    ],
    "check_demands_zero_padding": [
        ('    if h["max_steps"] == 0', '    if any(data[32:64]) or h["max_steps"] == 0')
    ],
    "bad_upload_req_always_zero": [
        (
            'req = struct.unpack_from("<I", data, 24)[0] if len(data) >= 28 else 0',
            "req = 0",
        )
    ],
    "bad_upload_event_has_pass_one": [
        ("bad.reason | bad.pc << 8, 0)]", "bad.reason | bad.pc << 8, 1)]")
    ],
}


def load(patches, name):
    source = SOURCE
    for old, new in patches:
        assert source.count(old) == 1, (name, old, source.count(old))
        source = source.replace(old, new)
    module = types.ModuleType(f"mutant_{name}")
    sys.modules[module.__name__] = module
    try:
        exec(compile(source, f"<{name}>", "exec"), module.__dict__)
    except BaseException:
        del sys.modules[module.__name__]
        raise
    return module


def too_big_uploads_are_refused(module):
    """True if the check refuses 65,536 micro-ops and 65,536 templates.

    Uploads that size are too big to save as files, so these two limits are probed here.
    """
    big = 1 << 23
    for n_ops, n_records in ((65536, 0), (1, 65536)):
        head = struct.pack("<8I32x", st.MAGIC, st.VERSION, n_ops, n_records, 9, 0, 1, 1)
        ops = struct.pack("<4B3I", st.OP_JMP, 0, 0, 0, 0, 0, 0) * n_ops
        tpl = struct.pack("<16I", 0, 1, 0, *[0] * 13) + bytes(16)
        data = head + ops + bytes(-len(ops) % 64) + tpl * n_records
        data += bytes(-len(data) % 64)
        if module.check(data, big) != (st.ERR_HEADER, 0):
            return False
    return True


BIG_UPLOAD_MUTANTS = {"check_ignores_n_ops_limit", "check_ignores_n_records_limit"}


def killed_by(module):
    """The first reference whose saved answer the module no longer reproduces."""
    for name in ref.REFERENCES:
        upload = (ref.DATA / f"{name}.bin").read_bytes()
        doc = json.loads((ref.DATA / f"{name}.json").read_text())
        try:
            got = ref.expected(upload, doc["params"], module)
        except Exception:  # noqa: BLE001  (any crash of a mutant kills it)
            return name
        if got != doc["expect"]:
            return name
    return None


def survives(name):
    """True if no reference notices the mutant `name`."""
    module = load(MUTANTS[name], name)
    try:
        if name in BIG_UPLOAD_MUTANTS:
            return too_big_uploads_are_refused(module)
        return killed_by(module) is None
    finally:
        del sys.modules[module.__name__]


def main():
    module = load([], "none")
    try:
        clean = killed_by(module) is None and too_big_uploads_are_refused(module)
    finally:
        del sys.modules[module.__name__]
    if not clean:
        print("the unmutated source does not reproduce every reference")
        return 1
    alive = [name for name in sorted(MUTANTS) if survives(name)]
    print(f"{len(MUTANTS) - len(alive)} of {len(MUTANTS)} mutants killed")
    for name in alive:
        print(f"survives: {name}")
    return bool(alive)


if __name__ == "__main__":
    sys.exit(main())
