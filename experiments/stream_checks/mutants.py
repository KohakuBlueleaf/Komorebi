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
GROUP = "        if group_outside(first, count, self.core_count):\n"
SUBMIT_COUNT = "        if count == 0 or count > self.core_count:\n"
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
            "    for first in range(core_count - count + 1):",
            "    for first in range(core_count - count, -1, -1):",
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
    # submitter turns
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
            "                return self.fail(ERROR_JUMP, self.pc)\n"
            '            if self.steps >= self.h["max_steps"]:\n'
            "                return self.fail(ERROR_STEPS, self.pc)\n",
            '            if self.steps >= self.h["max_steps"]:\n'
            "                return self.fail(ERROR_STEPS, self.pc)\n"
            "            if self.pc >= len(self.ops):\n"
            "                return self.fail(ERROR_JUMP, self.pc)\n",
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
            "            self.fail(ERROR_CANCELLED, self.pc)\n",
            "        if cancel and self.error is None:\n"
            "            self.fail(ERROR_CANCELLED, self.pc)\n"
            "        self.check_deadlines()\n",
        )
    ],
    "cancel_pc_zero": [
        ("self.fail(ERROR_CANCELLED, self.pc)", "self.fail(ERROR_CANCELLED, 0)")
    ],
    "orphans_not_drained": [
        ("                sim.drain(p + 1, max_passes)\n", "                pass\n")
    ],
    "orphans_have_no_deadline": [
        (
            "                else:\n                    self.post(EVENT_TIMEOUT, slot)\n",
            "                else:\n                    self.stopped = False\n",
        )
    ],
    "orphan_timeout_names_pc": [
        ("self.post(EVENT_TIMEOUT, slot)", "self.post(EVENT_TIMEOUT, busy.command.pc)")
    ],
    "orphan_event_has_no_req": [
        (
            "                    self.post(EVENT_TIMEOUT, slot)\n",
            "                    self.events.append(Event(EVENT_TIMEOUT, 0, slot, self.now))\n",
        )
    ],
    "deadline_does_not_stop_the_grid": [("                self.stopped = True\n", "")],
    "default_deadline_ignored": [
        (
            'if deadline_passes is None and h["deadline_microseconds"] == 0:',
            'if False and deadline_passes is None and h["deadline_microseconds"] == 0:',
        )
    ],
    "default_deadline_for_every_header": [
        (
            'if deadline_passes is None and h["deadline_microseconds"] == 0:',
            "if deadline_passes is None:",
        )
    ],
    # failure checks: which first
    "group_check_after_slot_wait": [
        (
            GROUP + "            return self.fail(ERROR_GROUP, self.pc)\n"
            "        slot = self.free_slot()\n"
            '        if slot is None:\n            return "stall"\n',
            "        slot = self.free_slot()\n"
            '        if slot is None:\n            return "stall"\n'
            + GROUP
            + "            return self.fail(ERROR_GROUP, self.pc)\n",
        )
    ],
    "count_check_after_queue_full": [
        (
            SUBMIT_COUNT
            + "            return self.fail(ERROR_NEVER_FITS, self.pc)\n"
            + QUEUE_FULL,
            QUEUE_FULL
            + SUBMIT_COUNT
            + "            return self.fail(ERROR_NEVER_FITS, self.pc)\n",
        )
    ],
    "group_sum_wraps": [
        (
            "    return count == 0 or first > core_count or count > core_count - first",
            "    return count == 0 or (first + count) & MASK > core_count",
        )
    ],
    "group_off_by_one_count": [
        ("count > core_count - first", "count >= core_count - first")
    ],
    "group_empty_allowed": [("return count == 0 or first", "return first")],
    "job_of_zero_allowed": [(SUBMIT_COUNT, "        if count > self.core_count:\n")],
    "job_of_whole_grid_rejected": [
        (SUBMIT_COUNT, "        if count == 0 or count >= self.core_count:\n")
    ],
    # micro-op semantics
    "jlt_signed": [
        (
            "nxt = o.immediate if registers[o.a] < registers[o.b] else nxt",
            "nxt = o.immediate if registers[o.a] ^ 0x80000000 < registers[o.b] ^ 0x80000000 else nxt",
        )
    ],
    "mul_not_wrapped": [
        (
            "registers[o.a] = registers[o.b] * registers[o.c] & MASK",
            "registers[o.a] = registers[o.b] * registers[o.c]",
        )
    ],
    "shr_amount_unmasked": [
        ("registers[o.b] >> (o.immediate & 31)", "registers[o.b] >> o.immediate")
    ],
    "shr_arithmetic": [
        (
            "registers[o.a] = registers[o.b] >> (o.immediate & 31)",
            "registers[o.a] = ((registers[o.b] ^ 0x80000000) - 0x80000000 >> (o.immediate & 31)) & MASK",
        )
    ],
    "mod_signed": [
        (
            "registers[o.a] = registers[o.b] % registers[o.c]",
            "registers[o.a] = ((registers[o.b] ^ 0x80000000) - 0x80000000) % registers[o.c] & MASK",
        )
    ],
    "sub_not_wrapped": [
        (
            "registers[o.a] = registers[o.b] - registers[o.c] & MASK",
            "registers[o.a] = registers[o.b] - registers[o.c]",
        )
    ],
    "addi_not_wrapped": [
        (
            "registers[o.a] = registers[o.b] + o.immediate & MASK",
            "registers[o.a] = registers[o.b] + o.immediate",
        )
    ],
    "add_not_wrapped": [
        (
            "registers[o.a] = registers[o.b] + registers[o.c] & MASK",
            "registers[o.a] = registers[o.b] + registers[o.c]",
        )
    ],
    "and_is_or": [
        (
            "registers[o.a] = registers[o.b] & registers[o.c]",
            "registers[o.a] = registers[o.b] | registers[o.c]",
        )
    ],
    "jeq_is_jne": [
        (
            "nxt = o.immediate if registers[o.a] == registers[o.b] else nxt",
            "nxt = o.immediate if registers[o.a] != registers[o.b] else nxt",
        )
    ],
    "jnz_is_jz": [
        (
            "nxt = o.immediate if registers[o.a] != 0 else nxt",
            "nxt = o.immediate if registers[o.a] == 0 else nxt",
        )
    ],
    "rand_leaves_seed": [
        (
            "                registers[SEED_REGISTER] = xorshift32(registers[SEED_REGISTER])\n"
            "                registers[o.a] = registers[SEED_REGISTER]\n",
            "                registers[o.a] = xorshift32(registers[SEED_REGISTER])\n",
        )
    ],
    "rand_wrong_shift": [("x ^= x << 5 & MASK", "x ^= x << 7 & MASK")],
    "mod_zero_gives_zero": [
        (
            "                if registers[o.c] == 0:\n                    return self.fail(ERROR_MOD_ZERO, self.pc)\n"
            "                registers[o.a] = registers[o.b] % registers[o.c]\n",
            "                registers[o.a] = registers[o.b] % registers[o.c] if registers[o.c] else 0\n",
        )
    ],
    "event_auxiliary_not_masked": [
        (
            "self.post(EVENT_STREAM, registers[o.a] & 0xFFFFFF)",
            "self.post(EVENT_STREAM, registers[o.a])",
        )
    ],
    "end_auxiliary_not_masked": [
        (
            "self.post(EVENT_STREAM_END, self.halt & 0xFFFFFF)",
            "self.post(EVENT_STREAM_END, self.halt)",
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
    "wait_is_off_by_one": [
        ("self.pending > o.immediate", "self.pending >= o.immediate")
    ],
    "ack_in_the_same_pass": [
        (
            "done = None if delay is None else self.now + 1 + delay",
            "done = None if delay is None else self.now + delay",
        )
    ],
    "result_starts_nonzero": [
        (
            "self.pc = self.steps = self.sequence = self.pending = self.last_result = 0",
            "self.pc = self.steps = self.sequence = self.pending = 0\n        self.last_result = 7",
        )
    ],
    "result_of_unflagged_commands": [
        (
            "                if busy.command.flags & FLAG_RESULT:\n                    self.last_result = busy.result\n",
            "                self.last_result = busy.result\n",
        )
    ],
    "excl_ignored": [
        ("        if t.flags & FLAG_EXCLUSIVE and any(", "        if False and any(")
    ],
    "excl_touching_groups_wait": [
        (
            "first < f + c and f < first + count for f, c in self.busy_groups()",
            "first <= f + c and f <= first + count for f, c in self.busy_groups()",
        )
    ],
    "queue_flag_kept": [
        (
            "flags = words[1] >> 8 & 0xFF & ~FLAG_QUEUE",
            "flags = words[1] >> 8 & 0xFF | (FLAG_QUEUE if queued else 0)",
        )
    ],
    "earlier_patch_wins": [
        (
            "        for i in range(t.patch_count):\n            words[ARG_WORD[t.arg[i]]]",
            "        for i in reversed(range(t.patch_count)):\n            words[ARG_WORD[t.arg[i]]]",
        )
    ],
    "patch_list_ignores_patch_count": [
        (
            "        for i in range(t.patch_count):\n            words[ARG_WORD[t.arg[i]]]",
            "        for i in range(PATCHES):\n            words[ARG_WORD[t.arg[i]]]",
        )
    ],
    "queued_job_reads_registers_at_submit": [
        (
            "job = _Job(self.patched(self.templates[o.immediate]), count, self.pc)",
            "job = _Job(self.templates[o.immediate], count, self.pc)",
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
        ('1 <= h["op_count"] <= MAX_OPS', 'h["op_count"] <= MAX_OPS')
    ],
    "check_ignores_op_count_limit": [
        ('1 <= h["op_count"] <= MAX_OPS', '1 <= h["op_count"]')
    ],
    "check_ignores_command_count_limit": [(' or h["command_count"] > MAX_OPS', "")],
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
            "        if o.op not in OPS:\n            return StreamError(ERROR_OPCODE, pc)\n",
            "        if o.op not in OPS:\n            continue\n",
        )
    ],
    "check_looks_at_unused_registers": [
        (
            "        if any(getattr(o, f) >= REGISTERS for f in fields):",
            "        if any(getattr(o, f) >= REGISTERS for f in 'abc'):",
        )
    ],
    "check_ignores_register_c": [
        ("for f in fields", "for f in fields.replace('c', '')")
    ],
    "check_jump_target_equal_allowed": [
        ('o.immediate >= h["op_count"]', 'o.immediate > h["op_count"]')
    ],
    "check_template_index_equal_allowed": [
        ('o.immediate >= h["command_count"]', 'o.immediate > h["command_count"]')
    ],
    "check_templates_before_micro_ops": [
        (
            "    first_use = {}\n    for pc, o in enumerate(ops):",
            "    first_use = {}\n    for index, t in enumerate(templates):\n"
            "        if _bad_template(t):\n            return StreamError(ERROR_TEMPLATE, 0)\n"
            "    for pc, o in enumerate(ops):",
        )
    ],
    "check_template_pc_zero": [("first_use.get(index, 0)", "0")],
    "check_template_pc_is_last_user": [
        ("first_use.setdefault(o.immediate, pc)", "first_use[o.immediate] = pc")
    ],
    "check_ignores_handler_zero": [
        ("not 1 <= t.handler <= HANDLER_USER_MAX", "t.handler > HANDLER_USER_MAX")
    ],
    "check_allows_host_handlers": [
        ("not 1 <= t.handler <= HANDLER_USER_MAX", "not 1 <= t.handler <= 255")
    ],
    "check_ignores_flag_bits": [(" or t.flags & ~(FLAG_EXCLUSIVE | FLAG_RESULT)", "")],
    "check_allows_queue_flag": [
        (
            "~(FLAG_EXCLUSIVE | FLAG_RESULT)",
            "~(FLAG_EXCLUSIVE | FLAG_RESULT | FLAG_QUEUE)",
        )
    ],
    "check_five_patches_allowed": [
        ("t.patch_count > PATCHES", "t.patch_count > PATCHES + 1")
    ],
    "check_ignores_patch_argument": [("t.arg[i] >= ARGS or ", "")],
    "check_ignores_patch_register": [(" or t.register[i] >= REGISTERS", "")],
    "check_checks_unused_patches": [
        (
            "any(t.arg[i] >= ARGS or t.register[i] >= REGISTERS for i in range(t.patch_count))",
            "any(t.arg[i] >= ARGS or t.register[i] >= REGISTERS for i in range(PATCHES))",
        )
    ],
    "check_ignores_first_register": [
        (
            "for r in (t.first_register, t.count_register)",
            "for r in (t.count_register,)",
        )
    ],
    "check_ignores_count_register": [
        (
            "for r in (t.first_register, t.count_register)",
            "for r in (t.first_register,)",
        )
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


def too_big_uploads_are_rejected(module):
    """True if the check rejects 65,536 micro-ops and 65,536 templates.

    Uploads that size are too big to save as files, so these two limits are probed here.
    """
    big = 1 << 23
    for op_count, command_count in ((65536, 0), (1, 65536)):
        head = struct.pack(
            "<8I32x", st.MAGIC, st.VERSION, op_count, command_count, 9, 0, 1, 1
        )
        ops = struct.pack("<4B3I", st.OP_JMP, 0, 0, 0, 0, 0, 0) * op_count
        template = struct.pack("<16I", 0, 1, 0, *[0] * 13) + bytes(16)
        data = head + ops + bytes(-len(ops) % 64) + template * command_count
        data += bytes(-len(data) % 64)
        if module.check(data, big) != (st.ERROR_HEADER, 0):
            return False
    return True


BIG_UPLOAD_MUTANTS = {
    "check_ignores_op_count_limit",
    "check_ignores_command_count_limit",
}


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
            return too_big_uploads_are_rejected(module)
        return killed_by(module) is None
    finally:
        del sys.modules[module.__name__]


def main():
    module = load([], "none")
    try:
        clean = killed_by(module) is None and too_big_uploads_are_rejected(module)
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
