"""A DRAM buffer as a host-only build needs it (page size, pages, address): a Tensor
of it compiles a class with no session."""


class Pages:
    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 64, 0x1000

    def accessor_args(self):
        return [2, self.page_size]
