"""A card session: opens the device, loads a program as persistent kernels on every
Tensix core and the dispatcher on X280 cluster 0, and hands out a `Run`.

Needs a tt-metal tree or the runtime package; `import komorebi` does not import it.
"""

from komorebi.device import channel as channel_module
from komorebi.device import remembered
from komorebi.session import sizes
from komorebi.session.parts import load
from komorebi.session.parts.load import LoadMixin
from komorebi.session.parts.memory import MemoryMixin
from komorebi.session.parts.open import OpenMixin
from komorebi.session.parts.transfer import TransferMixin


class Session(OpenMixin, MemoryMixin, TransferMixin, LoadMixin):
    """One card: `card` selects it, `device` uses an open one; `grid_timeout` bounds
    every grid wait. As a context manager it stops a loaded run at the exit (through
    the runtime on an exception) and closes the device it opened; so do the process's
    exit and SIGTERM. km.Session (komorebi.sessions) gives islands an IslandSession."""

    def __init__(
        self,
        card=None,
        device=None,
        pll_mhz=None,
        tree=None,
        log=None,
        grid_timeout=10.0,
        kernel_bytes=None,
        channel=channel_module.REGION_DEFAULT,
        slow_code=False,
        expect=None,
    ):
        self.card = card
        self.device = device
        self._own_device = device is None
        self.pll_mhz = load.X280_MHZ if pll_mhz is None else pll_mhz
        self._default_clock = pll_mhz is None  # may fall back to X280_FALLBACK_MHZ
        self.tree = tree
        self.log = log or (lambda message: None)
        self.log_given = log is not None  # without one, a reopen is printed to stderr
        self.grid_timeout = grid_timeout
        self.kernel_bytes = kernel_bytes  # None: the runtime's, then the kernels'
        # kernel_bytes is a size to open at, not a pin: a load may reopen the card at
        # another (a split card's remembered size, komorebi.islands.session)
        self.kernel_followed = False
        self.slow_code = slow_code  # slots may end above L1_FAST_CODE
        self._init_memory()
        self._worker_l1 = sizes.worker_l1_size(kernel_bytes)
        if device is not None and kernel_bytes is not None:
            raise ValueError("kernel_bytes is set when the session opens the device")
        self._memory = None  # the remembered kernel buffer (komorebi.device.remembered)
        self._init_transfer()
        if kernel_bytes is None and device is None:
            self._memory = remembered.Memory(self.log, expect)
            if self._memory.size and self._memory.size <= sizes.KERNEL_BUFFER_MAX:
                self._worker_l1 = sizes.worker_l1_size(self._memory.size)
        self._init_load()
        # the host channel: the region asked for, its bytes once mapped (else None)
        self._region = channel_module.check_region(channel) if channel else None
        # a given device's channel is mapped at the first transfer (open() maps its own)
        self._channel_mapped = device is None
        self._init_open()

    def __enter__(self):
        if self.device is None:
            self.open()
        return self

    def __exit__(self, kind, error, traceback):
        try:
            if self.run is not None and not self.run.stopped:
                if kind is None:
                    self.run.stop()
                else:
                    self.run.abort()
        finally:
            self.close()
