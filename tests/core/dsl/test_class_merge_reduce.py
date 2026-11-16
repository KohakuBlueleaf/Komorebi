"""A state merged up a group's tree, not added: `km.reduce(self.state, form="tree",
group=S, combine=MERGE)` runs the MERGE body on each arrival and leaves the state in
place at the root; a sender's state goes on from mover B. Its refusals (no card)."""

import pytest
from chain_fixture import quiet, tensor

import komorebi as km
from komorebi.errors import ProgramError

MERGE = "merge_state(arrival_buffer, arrival_index, state);"


class Attend(km.Program):
    """Each core's state, merged over groups of 4, the root's taken after."""

    state: km.L1[km.fp32, 3, 32, 32] = km.own(push=None)

    @km.handler
    def attend(self, q: km.Tensor):
        km.cxx("fill(state);", takes=[self.state])
        km.reduce(self.state, form="tree", group=4, combine=MERGE)
        if self.rank == self.leader:
            km.cxx("finish(state);", takes=[self.state])


def made(cls):
    return cls(l1=1 << 20, log=quiet, q=tensor(8, 0x10000))


def test_each_arrival_is_merged_by_the_body_and_a_sender_pushes_its_state():
    program = made(Attend)
    compute = (program.built.folder / "compute.cpp").read_text()
    movement = (program.built.folder / "movement.cpp").read_text()
    merge = compute[compute.index("merged with the group's") :]
    assert "km::reduce(compute, kArrival0, km::digit_group(" in merge
    assert "const uint32_t arrival_buffer = kArrival0.cb;" in merge
    assert "const uint32_t arrival_index = km::slot_page(kArrival0, km_slot);" in merge
    assert "const uint32_t state = kFieldState;" in merge
    assert merge.index(MERGE) < merge.index(
        "cb_push_back(kFieldState, kFieldStatePages)"
    )
    sent = merge[: merge.index("cb_push_back(kFieldState")]
    assert ".root) {  // to mover B's send" in sent  # a sender's state alone
    assert "km_room_state = compute.reserve_back(kFieldState" in sent
    assert "kFieldState, kFieldStatePages)) {" in movement  # mover B's emit
    assert "reduce_arrive(movement, command, kArrival0" in movement
    header = (program.built.folder / "attend.h").read_text()
    arrival = next(line for line in header.splitlines() if "Arrival kArrival0" in line)
    assert arrival.split("{")[1].split("}")[0].split(", ")[1:] == ["6", "3"]  # 2 x 3


class Plain(Attend):
    stored: km.L1[km.fp32, 3, 32, 32]

    @km.handler
    def attend(self, q: km.Tensor):
        km.reduce(self.stored, form="tree", group=4, combine=MERGE)


class Pushed(Attend):
    token: km.L1[km.fp32, 3, 32, 32] = km.own(push="pack")

    @km.handler
    def attend(self, q: km.Tensor):
        km.reduce(self.token, form="tree", group=4, combine=MERGE)


class Half(Attend):
    small: km.L1[km.bf16, 3, 32, 32] = km.own(push=None)

    @km.handler
    def attend(self, q: km.Tensor):
        km.reduce(self.small, form="tree", group=4, combine=MERGE)


class Named(Attend):
    @km.handler
    def attend(self, q: km.Tensor):
        merged = km.reduce(self.state, form="tree", group=4, combine=MERGE)
        if self.rank == self.leader:
            km.cxx("(void)0;", takes=[merged])


class Owner(Attend):
    @km.handler
    def attend(self, q: km.Tensor):
        km.reduce(self.state, form="owner", combine=MERGE)


class Unassigned(Attend):
    @km.handler
    def attend(self, q: km.Tensor):
        km.reduce(self.state, form="tree", group=4)


@pytest.mark.parametrize(
    "cls, says",
    [
        (Plain, "combine= merges a state written in place: an own field of fp32"),
        (Pushed, "combine= merges a state written in place"),
        (Half, "self.small is bfloat16: a merged state is fp32 tiles"),
        (Named, "leaves the merged state in place"),
        (Owner, "a merge by owner holds a slot"),
        (Unassigned, "a reduce's sum is a value: it is assigned"),
    ],
)
def test_a_merge_of_anything_but_an_own_fp32_state_is_refused(cls, says):
    with pytest.raises(ProgramError) as error:
        made(cls)
    assert says in str(error.value) and "line " in str(error.value)
