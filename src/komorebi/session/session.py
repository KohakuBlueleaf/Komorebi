"""A card session: opens the device, loads a program as persistent kernels on every
Tensix core and the dispatcher on X280 cluster 0, and hands out a `Run`.

Needs a tt-metal tree or the runtime package; `import komorebi` does not import it.
"""

from komorebi.device import channel as channel_module
from komorebi.device import native, remembered
from komorebi.errors import KomorebiError
from komorebi.session import dispatch as dispatch_module
from komorebi.session import process, sizes
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
        dispatcher=None,
    ):
        if dispatcher is not None:
            dispatch_module.choose(dispatcher, True)  # a name it knows
        self._dispatcher = dispatcher  # x280, tensix; None: chosen at the first load
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
        self._arch = None  # its chip's architecture, when first asked (arch)
        if device is not None and kernel_bytes is not None:
            raise ValueError("kernel_bytes is set when the session opens the device")
        self._worker_l1 = None
        if kernel_bytes is not None:
            self._worker_l1 = sizes.worker_l1_size(kernel_bytes, self.arch)
        self._memory = None  # the remembered kernel buffer (komorebi.device.remembered)
        self._init_transfer()
        if kernel_bytes is None and device is None:
            arch = self._arch_known()
            shape = remembered.chip_shape("card", arch)
            self._memory = remembered.Memory(self.log, expect, shape)
            size = self._memory.size
            if size and arch and size <= sizes.kernel_buffer_max(arch):
                self._worker_l1 = sizes.worker_l1_size(size, arch)
        self._init_load()
        # the host channel: the region asked for, its bytes once mapped (else None)
        self._region = channel_module.check_region(channel) if channel else None
        # a given device's channel is mapped at the first transfer (open() maps its own)
        self._channel_mapped = device is None
        self._init_open()

    @property
    def arch(self):
        """Its chip's architecture (Device.arch's name): the given device's, else its
        card's, read before the open (process.card_arch) and checked at it."""
        if self._arch is None and self.device is not None:
            self._arch = self.device.arch()
        elif self._arch is None:
            card = self.card if self.card is not None else process.visible_card()
            self._arch = process.card_arch(card)
        return self._arch

    @property
    def ranks(self):
        """The cores a load's program runs on: the grid's, less a Tensix dispatcher's
        (the island session's: its flat ranks)."""
        return dispatch_module.workers(self.grid, self.dispatcher)

    @property
    def dispatcher(self):
        """The dispatcher of its loads: "x280" or "tensix" (Session(dispatcher=...),
        else KMR_DISPATCHER, else the X280 where the build and the chip have one)."""
        return self._dispatch_kind(self._module or native.load_module(self.tree))

    def _arch_known(self):
        """Its chip's architecture, or None when its card's type cannot be read: the
        open's preflight then refuses the card by name (card_ready)."""
        try:
            return self.arch
        except KomorebiError:
            return None

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
