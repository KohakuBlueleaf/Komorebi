"""The direct ring's ring pattern written as a class: `went = self.direct(rank,
message)` and a forward when it is 0, with the results words of kernels/ring.h, so
ring.py's replay checks it (ring.py --as-class)."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2] / "src")]

import komorebi as km  # noqa: E402
from komorebi import abi  # noqa: E402

RING = abi.constants(HERE / "kernels" / "ring.h")
MODE_DIRECT = RING["kModeDirect"]
RES_RECORDS = RING["kResRecords"]
RECORD_WORDS = RING["kRecordWords"]


class Token(km.Message):
    """A token: its id, the hops it has left, its generator, the sum of its ranks."""

    id: int
    hops: int
    state: int
    sum: int


class DirectRing(km.Program):
    """ring: `tokens` tokens a rank of the first `ranks`, each `hops` hops from r to
    r + 1, by a direct message (mode direct, a forward when it is busy) or forwards."""

    @km.handler
    def ring(
        self,
        tokens: int,
        hops: int,
        mode: int,
        pattern: int,
        ranks: int,
        records: int,
        results: int,
    ):
        visits = 0
        consumed = 0
        direct = 0
        busy = 0
        if self.seeding():  # noqa: SIM102 (the seeding if stands alone)
            if self.rank < ranks:
                for index in range(tokens):
                    state = (self.rank * tokens + index + 1) * 0x9E3779B1
                    if state == 0:
                        state = 1
                    seed = Token(id=self.rank * tokens + index, hops=hops, state=state)
                    self.send(self.rank, seed)
        for message in self.arrive():
            visits += 1
            message.sum += self.rank
            message.state ^= message.state << 13
            message.state ^= message.state >> 17
            message.state ^= message.state << 5
            if message.hops == 0:
                if consumed < records:
                    token = message.id + 1
                    state = message.state
                    total = message.sum
                    at = RES_RECORDS + consumed * RECORD_WORDS
                    km.cxx(
                        """
                        UNPACK(({
                            volatile tt_l1_ptr uint32_t* record = km::l1(results) + at;
                            record[0] = token;
                            record[1] = state;
                            record[2] = total;
                            record[3] = 0;
                        }));
                        """,
                        takes=[results, at, token, state, total],
                    )
                consumed += 1
                self.consume()
            else:
                message.hops -= 1
                to = (self.rank + 1) % ranks
                if mode == MODE_DIRECT:
                    went = self.direct(to, message)
                    if went == 0:
                        busy += 1
                        self.forward(to, message)
                    else:
                        direct += 1
                else:
                    self.forward(to, message)
        km.cxx(
            """
            UNPACK(({
                volatile tt_l1_ptr uint32_t* words = km::l1(results);
                words[0] = visits;
                words[1] = consumed;
                words[2] = direct;
                words[3] = busy;
                words[4] = command.sequence;
            }));
            """,
            takes=[results, visits, consumed, direct, busy],
        )
