"""Classes of c2_check.py: one put each (size, at=, source item)."""

import komorebi as km

CASES = []


class Put0(km.Program):  # size 4, at=self.rank, source self.st
    st: km.L1[km.u32, 1, 32, 32] = km.own(push=None)

    @km.handler
    def go(self, x: km.Tensor):
        x.items(size=4).put(self.st, at=self.rank)


CASES.append((Put0, 4, "self.rank", "self.st"))


class Put1(km.Program):  # size 8, at=self.rank, source self.st
    st: km.L1[km.u32, 1, 32, 32] = km.own(push=None)

    @km.handler
    def go(self, x: km.Tensor):
        x.items(size=8).put(self.st, at=self.rank)


CASES.append((Put1, 8, "self.rank", "self.st"))


class Put2(km.Program):  # size 12, at=self.rank, source self.st
    st: km.L1[km.u32, 1, 32, 32] = km.own(push=None)

    @km.handler
    def go(self, x: km.Tensor):
        x.items(size=12).put(self.st, at=self.rank)


CASES.append((Put2, 12, "self.rank", "self.st"))


class Put3(km.Program):  # size 20, at=self.rank, source self.st
    st: km.L1[km.u32, 1, 32, 32] = km.own(push=None)

    @km.handler
    def go(self, x: km.Tensor):
        x.items(size=20).put(self.st, at=self.rank)


CASES.append((Put3, 20, "self.rank", "self.st"))


class Put4(km.Program):  # size 16, at=self.rank, source self.st
    st: km.L1[km.u32, 1, 32, 32] = km.own(push=None)

    @km.handler
    def go(self, x: km.Tensor):
        x.items(size=16).put(self.st, at=self.rank)


CASES.append((Put4, 16, "self.rank", "self.st"))


class Put5(km.Program):  # size 64, at=self.rank * 4 + 1, source self.st[1]
    st: km.L1[km.u32, 1, 32, 32] = km.own(push=None)

    @km.handler
    def go(self, x: km.Tensor):
        x.items(size=64).put(self.st[1], at=self.rank * 4 + 1)


CASES.append((Put5, 64, "self.rank * 4 + 1", "self.st[1]"))


class Put6(km.Program):  # size 4, at=4, source self.st
    st: km.L1[km.u32, 1, 32, 32] = km.own(push=None)

    @km.handler
    def go(self, x: km.Tensor):
        x.items(size=4).put(self.st, at=4)


CASES.append((Put6, 4, "4", "self.st"))


class Put7(km.Program):  # size 4, at=3, source self.st
    st: km.L1[km.u32, 1, 32, 32] = km.own(push=None)

    @km.handler
    def go(self, x: km.Tensor):
        x.items(size=4).put(self.st, at=3)


CASES.append((Put7, 4, "3", "self.st"))


class Put8(km.Program):  # size 8, at=2, source self.st
    st: km.L1[km.u32, 1, 32, 32] = km.own(push=None)

    @km.handler
    def go(self, x: km.Tensor):
        x.items(size=8).put(self.st, at=2)


CASES.append((Put8, 8, "2", "self.st"))


class Put9(km.Program):  # size 8, at=3, source self.st[1]
    st: km.L1[km.u32, 1, 32, 32] = km.own(push=None)

    @km.handler
    def go(self, x: km.Tensor):
        x.items(size=8).put(self.st[1], at=3)


CASES.append((Put9, 8, "3", "self.st[1]"))


class Put10(km.Program):  # size 12, at=4, source self.st
    st: km.L1[km.u32, 1, 32, 32] = km.own(push=None)

    @km.handler
    def go(self, x: km.Tensor):
        x.items(size=12).put(self.st, at=4)


CASES.append((Put10, 12, "4", "self.st"))


class Put11(km.Program):  # size 12, at=5, source self.st
    st: km.L1[km.u32, 1, 32, 32] = km.own(push=None)

    @km.handler
    def go(self, x: km.Tensor):
        x.items(size=12).put(self.st, at=5)


CASES.append((Put11, 12, "5", "self.st"))
