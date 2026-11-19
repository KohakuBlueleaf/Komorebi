"""What differs between chip architectures that the host needs, in one table keyed by
the open device's name for its architecture (Device.arch): the place for every constant
that depends on the chip."""

from collections.abc import Callable
from typing import NamedTuple

from komorebi.errors import KomorebiError

# The tile compiler's words checked against Wormhole's (tt-isa-documentation, tt-llk's
# ckernel_ops.h and ckernel_instr_params.h): its instructions Blackhole alone has,
WORMHOLE_ABSENT = (
    "SFPGT",
    "SFPLE",
    "SFPARECIP",
    "SFPMUL24",
    "CFGSHIFTMASK",
    "MOVDBGB2D",
    "RESOURCEDECL",
    "STREAMWAIT",
    "STREAMWRCFG",
)
MAD_NEGATE = 3  # SFPMAD, SFPADD, SFPMUL instr_mod1 bits 0, 1: NEGATE_VA, NEGATE_VC
SHFT_BLACKHOLE = 6  # SFPSHFT instr_mod1 bits 1, 2: ARITHMETIC, ARG_IMM_USE_VC
CAST_WORMHOLE = 2  # SFPCAST instr_mod1 0, 1 (rounding); 2, 3 are Blackhole's int casts
ADDR_MODS = 4  # Wormhole's AddrMod field is 2 bits (4 to 7 only with ADDR_MOD_SET_Base)
# vector words the tile compiler writes on Wormhole as their mode less 4, the base set
BASED = ("SFPLOAD", "SFPSTORE", "SFPLOADMACRO")
WAIT_SAME = 7  # STALLWAIT condition bits C0 to C2 mean the same; C3 on are renumbered
# STALLWAIT fields the tile compiler writes by tt-llk name (ckernel::p_stall):
# Blackhole's number to the name, which each chip's header numbers its own way
STALL_NAMES = {
    "stall_res": {0x100: "ckernel::p_stall::STALL_SFPU"},
    "wait_res": {0x10: "ckernel::p_stall::MATH"},
}
SHFT2_LANES = (
    2,
    3,
    4,
)  # SFPSHFT2's lane modes: their result is not held the next cycle
HALF_WRITES = {"SFPLOADI": (8, 10), "SFPLOAD": (14, 15)}  # instr_mod0: half an LReg
TEMPLATE_VD = 12  # a word whose VD is 12 + k stores load macro template k (back door)
REPLAY_MOST = 32  # Wormhole's replay count: 0 is 32 (64 on Blackhole), above it wraps


# Blackhole's vector-unit opcodes Wormhole lacks: in a Wormhole kernel's listing a raw
# word (`.insn 4, w`: the Tensix word rotated left by 2, low bits 2)
ABSENT_OPCODES = {0x96: "SFPLE", 0x97: "SFPGT", 0x98: "SFPMUL24", 0x99: "SFPARECIP"}
MAD_OPCODES = {0x84: "SFPMAD", 0x85: "SFPADD", 0x86: "SFPMUL"}
WORMHOLE_HAZARD = (
    "docs/reference/hazards.md, 'A Blackhole vector-unit word in a kernel'"
)


def no_insn_refusal(mnemonic, operands):
    """Every instruction a kernel's build holds is the chip's own: none refused."""
    return None


def wormhole_insn_refusal(mnemonic, operands):
    """Why an instruction of a Wormhole kernel's linked listing (objdump's mnemonic and
    operands) computes otherwise than the Blackhole word it was written as, or None: a
    MAD word's NEGATE bits (Wormhole clears them silently), a word Wormhole lacks."""
    mod1, name = None, mnemonic.upper()
    if mnemonic in ("sfpmad", "sfpadd", "sfpmul"):
        mod1 = int(operands.split(",")[-1], 0)
    elif mnemonic == ".insn":
        raw = int(operands.split(",")[-1], 0)
        if raw & 3 != 2:
            return None
        word = (raw >> 2) | (raw & 3) << 30
        opcode, mod1 = word >> 24, word & 15
        if opcode in ABSENT_OPCODES:
            return f"{ABSENT_OPCODES[opcode]} is an instruction Wormhole does not have"
        if opcode not in MAD_OPCODES:
            return None
        name = MAD_OPCODES[opcode]
    if mod1 is not None and mod1 & MAD_NEGATE:
        return (
            f"{name} instr_mod1 {mod1}: NEGATE_VA, NEGATE_VC are Blackhole's; Wormhole "
            "clears them and computes the result unnegated"
        )
    return None


def no_tile_refusal(before, word):
    """Every word the tile compiler emits is Blackhole's own: none refused."""
    return None


def wormhole_tile_refusal(before, word):
    """Why `word` (unit.chip_words: op, args, note, kind, reads, writes) after `before`
    (None at a run's start) is not what it is on Blackhole on a Wormhole core, or None.
    """
    op, args, _, _, reads, _ = word
    mod1 = args.get("instr_mod1", 0)
    if op in WORMHOLE_ABSENT:
        return f"{op} is an instruction Wormhole does not have"
    if op in ("SFPMAD", "SFPADD", "SFPMUL") and mod1 & MAD_NEGATE:
        return (
            f"{op} instr_mod1 {mod1}: NEGATE_VA, NEGATE_VC are Blackhole's (bits 0, 1)"
        )
    if op == "SFPSHFT" and mod1 & SHFT_BLACKHOLE:
        return f"SFPSHFT instr_mod1 {mod1}: ARITHMETIC, ARG_IMM_USE_VC are Blackhole's"
    if op == "SFPCAST" and mod1 >= CAST_WORMHOLE:
        return f"SFPCAST instr_mod1 {mod1}: Blackhole's integer cast"
    mode = args.get("sfpu_addr_mode", args.get("addr_mode", 0))
    if op not in BASED and mode >= ADDR_MODS:
        return (
            f"{op} address mode {mode}: Wormhole's field takes 0 to 3, and the unit "
            "sets ADDR_MOD_SET_Base only around its vector words"
        )
    if op == "REPLAY" and not 0 < args["len"] <= REPLAY_MOST:
        return (
            f"REPLAY of {args['len']} words: Wormhole replays 1 to 32 as given (its "
            "count holds the SFPNOPs Wormhole needs between the recorded words)"
        )
    wait = args.get("wait_res", 0)
    if op == "STALLWAIT" and wait & ~WAIT_SAME and wait not in STALL_NAMES["wait_res"]:
        return (
            f"STALLWAIT condition {wait:#x}: Wormhole numbers bits C3 on otherwise, "
            "and the unit writes only p_stall::MATH by its name"
        )
    return None if before is None else wormhole_hold(before, word)


def wormhole_hold(before, word):
    """Why `word` right after `before` (unit.chip_words' words) runs otherwise on a
    Wormhole core, which an SFPNOP between them mends (the tile compiler writes one
    there on Wormhole), or None."""
    op, args, _, _, reads, _ = word
    b_op, b_args, _, b_kind, _, b_writes = before
    held_not = set(reads) & set(b_writes)
    if b_kind == "mad" and held_not:
        return (
            f"{op} reads LReg {min(held_not)}, {b_op}'s result, the next cycle: "
            "Wormhole does not hold it back (an SFPNOP between)"
        )
    if b_op == "SFPSHFT2" and b_args["instr_mod1"] in SHFT2_LANES and held_not:
        return f"{op} reads LReg {min(held_not)} the cycle after SFPSHFT2's lane mode"
    half = args.get("instr_mod0") in HALF_WRITES.get(op, ())
    if b_op == "SFPSWAP" and (op in ("SFP_STOCH_RND", "SFPSHFT2") or half):
        return f"{op} right after SFPSWAP: their writes collide on Wormhole"
    if (
        b_op == "SFPSWAP"
        and b_args.get("lreg_dest", 0) >= TEMPLATE_VD
        and op != "SFPNOP"
    ):
        return f"{op} right after SFPSWAP stored as a template: Wormhole runs it twice"
    return None


class Chip(NamedTuple):
    """One architecture's facts."""

    busy_mhz: int  # the Tensix clock the driver raises a chip to while it is held
    x280: bool  # an X280 cluster beside the grid: the dispatcher a load needs
    # one multicast rectangle covers the whole worker grid, its NoC0 gaps included
    # (measured); else a rectangle stays in a panel
    whole_grid: bool
    # a multicast whose rectangle holds its sender is sent with loopback_src, the sender
    # counted (measured); else the plain form, the sender counted out
    multicast_loopback: bool
    # tt-metal places a core's circular buffers on the DRAM alignment (program.cpp)
    cb_align: int
    full_grid: tuple | None  # the unharvested worker grid whole_grid was measured on
    tensix_noc0_x: tuple | None  # Tensix columns' NoC0 x by harvest bit (None: rows)
    # a Tensix core's L1 and where the runtime's kernel buffer starts (MEM_L1_SIZE,
    # MEM_MAP_END): the open's sizes before the device says, checked at the open
    l1_bytes: int
    kernel_buffer_base: int
    # wall-clock ticks a cycle of the clock the chip reports (tt_aiclk), measured:
    # ticks to time and time limits to ticks use tick_mhz (experiments/core/wall_clock)
    tick_ratio: float
    # tt-metal's fabric's largest payload (erisc_datamover_builder.hpp): an island
    # session's packet is at most this (islands.session.largest_packet)
    fabric_payload_bytes: int
    # the link cores' own buffer index (the put queue): km_abi.h's KM_LINK_CB past
    # every other, or None: the highest a load leaves free below the runtime's 29
    link_cb: int | None
    # the links a link core holds, a CPU each (KM_LINKS_A_CORE): two, or one where the
    # writer's 16 KiB holds no link service (Wormhole's NCRISC; its reader the BRISC)
    links_a_core: int
    # an island session's links toward each neighbour when it names none (links=None):
    # None every link found (4 a Blackhole pair), or so many (a link core each)
    default_links: int | None
    # whether a link packet may target the far chip's DRAM (hazards.md: Wormhole's not)
    link_far_dram: bool
    # why a tile compiler's word (after the one before it) is wrong on the chip, or
    # None: a predicate, not a set, as modifiers and the next cycle's holds differ too
    tile_refusal: Callable
    # why an instruction of a kernel's linked listing (objdump's mnemonic, operands)
    # computes otherwise on the chip than the Blackhole word it was written as, or None
    insn_refusal: Callable
    # the pack core (TRISC2) has the RISC-V vector extension (Zve32f; Wormhole's ISA
    # string, rv32im_zmmul_xtttensixwh, has none): compute option enable_trisc2_rvv
    vector_extension: bool
    # the pack core reads and writes Dst through a RISC-V window (Wormhole has none:
    # the unpacker and the packer move its words): KM_DST_WINDOW of km_dst.hpp
    dst_window: bool
    # the writer's optimisation level when the program sets none (None: the runtime's)
    # and its defines, for the code its CPU holds (Wormhole's NCRISC: 16 KiB)
    writer_opt: str | None
    writer_defines: tuple


CHIPS = {
    "blackhole": Chip(
        busy_mhz=1350,
        x280=True,
        whole_grid=False,
        multicast_loopback=False,
        cb_align=64,
        full_grid=None,
        tensix_noc0_x=(1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16),
        l1_bytes=1536 * 1024,
        kernel_buffer_base=41088,
        tick_ratio=1.0,  # 1,349.99 Mticks/s at a reported 1,350 MHz
        fabric_payload_bytes=15232,
        link_cb=33,
        links_a_core=2,
        default_links=None,
        link_far_dram=True,
        tile_refusal=no_tile_refusal,
        insn_refusal=no_insn_refusal,
        vector_extension=True,
        dst_window=True,
        writer_opt=None,
        writer_defines=(),
    ),
    # a multicast rectangle from any core, loopback_src, NoC 1 given high corner to
    # low: the 8 x 10 grid in one (round 3, a Galaxy's chip in slow dispatch)
    "wormhole_b0": Chip(
        busy_mhz=1000,
        x280=False,
        whole_grid=True,
        multicast_loopback=True,
        cb_align=32,  # a slot buffer measured at 0xf7e0
        full_grid=(8, 10),
        tensix_noc0_x=None,
        l1_bytes=1464 * 1024,
        kernel_buffer_base=35168,
        tick_ratio=0.9855,  # 985.5 Mticks/s at a reported 1,000 MHz (two workers)
        fabric_payload_bytes=7616,
        link_cb=None,  # 32 indices: KM_LINK_CB (33) is past them
        links_a_core=1,
        default_links=1,  # 16 links a chip, 4 a neighbour: one (a link core) each
        link_far_dram=False,
        tile_refusal=wormhole_tile_refusal,
        insn_refusal=wormhole_insn_refusal,
        vector_extension=False,
        dst_window=False,
        writer_opt="Os",
        writer_defines=(("KM_WAIT_SHARED", "1"),),  # the waits' loop once (km_core.hpp)
    ),
}
# The driver's card types (tt_card_type) by architecture: a card's before its open.
CARD_TYPES = {
    "p100": "blackhole",
    "p150": "blackhole",
    "p300": "blackhole",
    "n150": "wormhole_b0",
    "n300": "wormhole_b0",
    "galaxy-wormhole": "wormhole_b0",
}


def card_arch(card_type):
    """The architecture of the driver's card type `card_type` ("p150a": its prefix);
    KomorebiError for a type not listed."""
    for prefix, arch in CARD_TYPES.items():
        if card_type and card_type.startswith(prefix):
            return arch
    raise KomorebiError(
        f"card type {card_type!r} is not known (known: {', '.join(CARD_TYPES)})"
    )


# A card's remedy when it is not ready, by the driver's card type (tt_card_type): a
# Galaxy's chips are reset together; any other card is reset alone.
RESETS = {
    "galaxy-wormhole": (
        "reset the whole box with tt-smi -glx_reset (a Galaxy's chips are reset "
        "together: every process holding one of them exits first), then open it again"
    ),
}
RESET = (
    "reset it with tt-smi -r {card} (the recovery guide, when a card reset is still "
    "the only way), then open it again"
)


def reset(card, card_type):
    """What to do for card `card` (tt-metal's number) of the driver's card type
    `card_type` (None: not known) when it is not ready."""
    return RESETS.get(card_type, RESET).format(card=card)


def noc0_columns(arch, harvest_mask):
    """NoC0 x of each logical column of a chip of `arch` whose harvest mask is
    `harvest_mask`: the Tensix columns whose mask bit is clear."""
    columns = chip(arch).tensix_noc0_x
    if columns is None:
        raise KomorebiError(f"a {arch} chip harvests rows, not columns")
    return [x for i, x in enumerate(columns) if not harvest_mask >> i & 1]


def grid_panels(arch, grid):
    """Whether a multicast rectangle may cover the whole `grid` of a chip of `arch`
    (whole_grid); KomorebiError for a grid other than the one that was measured."""
    facts = chip(arch)
    if facts.whole_grid and facts.full_grid and tuple(grid) != facts.full_grid:
        raise KomorebiError(
            f"a {arch} chip with a {grid[0]} x {grid[1]} worker grid (harvested; "
            f"{facts.full_grid[0]} x {facts.full_grid[1]} unharvested) is not handled: "
            "its multicast rectangles were measured on the whole grid only"
        )
    return facts.whole_grid


def tick_mhz(arch, aiclk_mhz):
    """The wall clock's ticks a microsecond of a chip of `arch` whose reported clock
    is `aiclk_mhz` (None: not known): what ticks become time by."""
    return None if aiclk_mhz is None else aiclk_mhz * chip(arch).tick_ratio


def chip(arch):
    """The facts of architecture `arch` (Device.arch's name); KomorebiError names the
    ones known for any other."""
    try:
        return CHIPS[arch]
    except KeyError:
        raise KomorebiError(
            f"no chip facts for architecture {arch!r} (known: {', '.join(CHIPS)})"
        ) from None
