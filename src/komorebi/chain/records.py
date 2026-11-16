"""Chain records at one bank address on every island: a region taken on each island's
DRAM when the sessions open (their allocators alike then), handed out in whole lines,
so one chain command (its a7) names its records on every island it crosses to."""

from komorebi.device import dram

REGION_BYTES = 256 * 1024  # each bank's: 4,096 records of 64 bytes
PAGE_BYTES = 2048  # the region's pages: interleaved, each bank's run at one address


class Region:
    """`bytes_` of every bank at one address on every island (`address`), or none
    (`address` None) when the islands' first free places differ."""

    def __init__(self, sessions, bytes_=REGION_BYTES):
        self.size = bytes_
        self._free = [(0, bytes_)]  # (offset, bytes), by offset
        # small pages (page p on bank p % banks at the address plus p // banks pages):
        # the session's own, written by bank address, so no transfer chunk follows it
        page = min(PAGE_BYTES, bytes_)
        shape = (dram.DRAM_BANKS * bytes_ // page, page // 4)
        self._tensors = [
            session.empty(shape, "uint32", own=True) for session in sessions
        ]
        addresses = {tensor.address for tensor in self._tensors}
        self.address = addresses.pop() if len(addresses) == 1 else None
        if self.address is None:
            self.free()

    def take(self, bytes_):
        """The bank address of `bytes_` (whole lines) at the same place on every
        island; None when the region has no such room."""
        bytes_ = -(-bytes_ // 64) * 64
        for index, (offset, size) in enumerate(self._free if self.address else []):
            if size >= bytes_:
                self._free[index] = (offset + bytes_, size - bytes_)
                if size == bytes_:
                    del self._free[index]
                return self.address + offset
        return None

    def give(self, address, bytes_):
        """Gives back what take() gave, joined to its free neighbours."""
        offset, bytes_ = address - self.address, -(-bytes_ // 64) * 64
        self._free.append((offset, bytes_))
        self._free.sort()
        joined = []
        for start, size in self._free:
            if joined and joined[-1][0] + joined[-1][1] == start:
                joined[-1] = (joined[-1][0], joined[-1][1] + size)
            else:
                joined.append((start, size))
        self._free = joined

    def free(self):
        for tensor in self._tensors:
            tensor.free()
        self._tensors, self.address = [], None
