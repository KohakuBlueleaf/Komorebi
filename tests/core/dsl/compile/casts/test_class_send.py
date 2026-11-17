"""send() from a class: a bare `self.send(rank, message)` gives the run up by name with
no quota left (never a message dropped unseen); `went = self.send(rank, message)` is
the runtime's send as a word; the timeout is km.MessagesGaveUp naming the core and what
to write (no card)."""

import numpy as np

import komorebi as km
from komorebi import abi
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.front.nodes import prune, render
from komorebi.session import messages_gave


class Ray(km.Message):
    origin: int
    depth: int


class Fan(km.Program):
    @km.handler
    def trace(self, depth: int):
        if self.seeding():
            seed = Ray(origin=self.rank, depth=depth)
            self.send(self.rank, seed)
        for message in self.arrive():
            if message.depth != 0:
                for i in range(3):
                    child = Ray(origin=message.origin * 3 + i, depth=message.depth - 1)
                    self.send((message.origin + i) % self.size, child)
            self.consume()


class Tried(km.Program):
    @km.handler
    def trace(self, depth: int):
        for message in self.arrive():
            if message.depth != 0:
                child = Ray(origin=message.origin, depth=message.depth - 1)
                went = self.send(self.rank, child)
                if went == 0:
                    self.forward(self.rank, message)
                else:
                    self.consume()
            else:
                self.consume()


def compute_of(cls):
    compiled = compile_class(cls, {}, {})
    return "\n".join(render(prune(compiled.compute["trace"]), 0))


def test_a_bare_send_gives_the_run_up_with_no_quota_left():
    compute = compute_of(Fan)
    assert compute.count("km_messages.send_or_give_up(") == 2  # the seed, a child
    assert "km_messages.send(" not in compute


def test_a_named_send_is_the_runtimes_send_as_a_word():
    compute = compute_of(Tried)
    assert "const uint32_t went = (km_messages.send(" in compute
    assert "? 1u : 0u);" in compute
    assert "send_or_give_up" not in compute


def test_the_timeout_names_a_quota_give_up_and_what_to_write():
    h = abi.headers()
    blocks = np.zeros((2, h["KM_CONTROL_BYTES"] // 4), np.uint32)
    gave = (h["KM_CONTROL_MESSAGES"] + h["KM_MESSAGES_GAVE"]) // 4
    words = [0] * h["KMD_H_WORDS"]
    assert messages_gave.named({"x280_cores": words}, blocks, h) is None
    blocks[1, gave : gave + 4] = [h["kWhereMessagesQuota"], 5, 1, 0]
    words[h["KMD_H_MESSAGES_GAVE"]] = 0x30000 | 2  # 1 + core 1; the high bits masked
    named = messages_gave.named({"x280_cores": words}, blocks, h)
    assert (named["core"], named["where"], named["position"]) == (1, "quota", 5)
    error = km.MessagesGaveUp(7, 0, 1, (0, 2), [], gave=named)
    assert isinstance(error, km.CommandTimeout)
    assert "the run was given up by core 1: quota (self.send to rank 5" in str(error)
    assert "raise max_live, or write went = self.send(rank, m)" in str(error)
    error.gave |= {"rank": 113, "island": 1}  # as IslandRun names it
    error.name_gave()
    assert "given up by rank 113 (island 1, core 1): quota" in str(error)
