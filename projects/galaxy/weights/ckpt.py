"""The released checkpoint read on the host with numpy only: a safetensors shard's
header, a tensor by memory map, and the released values of its stored formats
(BF16, F32, FP8 E4M3 with an E8M0 scale a 32 x 32 block, FP4 E2M1 two a byte with an
E8M0 scale a 32 along K).
"""

import json
import os
import struct

import numpy as np

DTYPES = {
    "BF16": (np.uint16, 2),
    "F32": (np.float32, 4),
    "F8_E4M3": (np.uint8, 1),
    "F8_E8M0": (np.uint8, 1),
    "I8": (np.uint8, 1),  # FP4 E2M1, two a byte
}
# FP4 E2M1 codes 0..15: sign, 2 exponent bits, 1 mantissa bit
FP4_VALUES = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32
)


def _e4m3_table():
    """FP8 E4M3 (fn: no infinities, 0x7f and 0xff NaN) codes 0..255 as float32."""
    codes = np.arange(256)
    sign = np.where(codes & 0x80, -1.0, 1.0)
    exp = (codes >> 3) & 0xF
    man = codes & 0x7
    normal = np.ldexp(1.0 + man / 8.0, exp - 7)
    subnormal = np.ldexp(man / 8.0, -6)
    values = sign * np.where(exp == 0, subnormal, normal)
    values[(codes & 0x7F) == 0x7F] = np.nan
    return values.astype(np.float32)


E4M3_VALUES = _e4m3_table()


class Checkpoint:
    """A checkpoint folder's shards, by tensor name (model.safetensors.index.json)."""

    def __init__(self, root):
        self.root = root
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            self.where = json.load(f)["weight_map"]
        self._headers = {}

    def header(self, shard):
        if shard not in self._headers:
            with open(os.path.join(self.root, shard), "rb") as f:
                (size,) = struct.unpack("<Q", f.read(8))
                head = json.loads(f.read(size))
            head.pop("__metadata__", None)
            self._headers[shard] = (8 + size, head)
        return self._headers[shard]

    def info(self, name):
        """(dtype, shape, shard, byte offset in the shard, bytes)."""
        shard = self.where[name]
        base, head = self.header(shard)
        entry = head[name]
        start, end = entry["data_offsets"]
        return entry["dtype"], tuple(entry["shape"]), shard, base + start, end - start

    def raw(self, name):
        """The stored array, memory mapped (BF16 as uint16 bits, FP8/FP4 as uint8)."""
        dtype, shape, shard, offset, size = self.info(name)
        np_type, item = DTYPES[dtype]
        return np.memmap(
            os.path.join(self.root, shard), np_type, "r", offset, shape=shape
        )

    def values(self, name):
        """The released values as float32: a quantized weight times its scale."""
        dtype = self.info(name)[0]
        raw = self.raw(name)
        if dtype == "BF16":
            return bf16_to_f32(raw)
        if dtype == "F32":
            return np.array(raw)
        if dtype == "F8_E4M3":
            return fp8_values(raw, self.raw(name.replace(".weight", ".scale")))
        if dtype == "I8":
            return fp4_values(raw, self.raw(name.replace(".weight", ".scale")))
        raise ValueError(f"{name}: {dtype} has no values of its own")


def bf16_to_f32(bits):
    return (np.asarray(bits, np.uint16).astype(np.uint32) << 16).view(np.float32)


def e8m0(scale):
    """E8M0 bytes as float32 powers of two (255 is NaN)."""
    s = np.asarray(scale, np.uint8).astype(np.int32)
    out = np.ldexp(np.float32(1.0), s - 127).astype(np.float32)
    return np.where(s == 255, np.float32(np.nan), out)


def fp8_values(weight, scale):
    """FP8 E4M3 [out, in] times its E8M0 scale a 32 x 32 block ([out/32, in/32])."""
    w = E4M3_VALUES[np.asarray(weight)]
    s = e8m0(scale)
    s = np.repeat(np.repeat(s, 32, axis=0), 32, axis=1)[: w.shape[0], : w.shape[1]]
    return w * s


def fp4_codes(packed):
    """[out, in/2] bytes as [out, in] FP4 codes, the low nibble first."""
    p = np.asarray(packed, np.uint8)
    return np.stack([p & 0xF, p >> 4], axis=-1).reshape(p.shape[0], -1)


def fp4_values(packed, scale):
    """FP4 E2M1 [out, in/2] packed times its E8M0 scale a 32 along in ([out, in/32])."""
    w = FP4_VALUES[fp4_codes(packed)]
    return w * np.repeat(e8m0(scale), 32, axis=1)
