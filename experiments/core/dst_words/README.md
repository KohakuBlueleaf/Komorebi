# Dst words through `km::dst`

Question: do 32-bit words a pack thread writes into Dst and reads out of it through
`km::dst` (`komorebi/device/km_dst.hpp`) come back bit for bit, in the order of
Blackhole's Dst window, on a chip without the window, and what does a tile cost each
way against the window?

`km::dst` takes its form from the chips table's `dst_window` (`KM_DST_WINDOW`):

- Blackhole: the pack core's RISC-V window into Dst, word by word;
- Wormhole (no window): the unpacker writes a tile into Dst (tt-llk's unpack-to-dest
  path for 32-bit formats, the unpack thread serving the pack thread's request) and
  the packer reads it out.

`probe.py` runs one program on core (0, 0) per buffer format (float32, int32, uint32):
five input tiles, four of random words (4,096) and one of special patterns (NaN payloads
of both signs, infinities, subnormals, both zeros, all ones, every exponent with its
mantissa 0 and all ones, both signs). Each tile goes into Dst tile 0 and is read back;
the vector unit then copies tile 0's 32 vectors to tile 1 in another order (vector v to
5 v + 3 mod 32, `SFPLOAD` / `SFPSTORE` of 32 bits as they are), which is read back
against the host's model of the window's order (Dst word w: row w / 16, column w % 16).
A line a format: the wrong words each way, and the spans of the write (into Dst, done)
and the read (out of Dst, done) of each tile.

    python probe.py                       # the session's chip
    python probe.py --formats float32 --save out.npz
    kmr prebuild --cards 0 probe.py
    kmr prebuild --arch wormhole_b0 --cards 0 probe.py

## Measured

2026-11-19: Blackhole card 0 (prebuilt) and Wormhole chips 0 and 1 of the Galaxy box.

| | Blackhole (window) | Wormhole (unpacker and packer) |
|---|---|---|
| Wrong words, back (float32, int32) | 0 of 5,120, 0 of 5,120 | 0, 0 (both chips) |
| Wrong words, the vector unit's reordering | 0, 0 | 0, 0 (both chips) |
| A tile into Dst | 2.26 to 2.27 us | 0.16 to 0.22 us |
| A tile out of Dst | 2.51 us | 0.15 to 0.17 us |

Every special pattern came back as it went in: neither the unpacker nor the packer
changes a NaN payload, a subnormal, a negative zero or all ones in either format, and
the vector unit's 32-bit loads and stores keep them too. Wormhole's output words are
card 0's, every one (20,480 words compared). uint32 (the raster project's output
format), Wormhole chip 0: 0 wrong each way, a tile 0.17 to 0.23 us in and 0.16 to 0.17
us out. The window moves a word per RISC-V load or
store; the unpacker and the packer move a tile at a time, 13 times faster into Dst and
16 times out of it. A tile is `km::dst`'s unit there; the window takes any word range.
