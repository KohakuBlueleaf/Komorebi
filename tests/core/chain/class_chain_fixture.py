"""A chain's run on the host (no card): a fake backend whose blocks keep their bytes,
and a tensor's pages, for the class chain tests."""

from fake_port import FakeBackend


class Pages:
    def __init__(self, pages, address):
        self.page_size, self.num_pages, self.address = 4096, pages, address

    def accessor_args(self):
        return [2, self.page_size]


class Block:
    def __init__(self, data):
        self.address, self.data = 0x10_0000, bytes(data)

    def free(self):
        pass


class BlockBackend(FakeBackend):
    def __init__(self, port):
        super().__init__(port)
        self.blocks = []

    def block(self, data, place=None):
        self.blocks.append(Block(data if place is None else place(0x10_0000)))
        return self.blocks[-1]

    def fence(self):
        pass
