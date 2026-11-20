"""The rotary positions and the window state of a DeepSeek-V4.1-Flash layer on one
Wormhole chip: rope_swap then rope rotate the last 64 of 512 columns in place;
insert_move then insert put kv's rows into the window (exact). Tensors are seen 512
wide. README.md, "The rotary positions and the insert"."""

import numpy as np

import komorebi as km

TILE = 32
HEAD_DIM, ROPE_DIM, HEADS = 512, 64, 64
ROPE_FIRST = (HEAD_DIM - ROPE_DIM) // TILE  # the rotated tiles: 14 and 15
ROPE_TILES = ROPE_DIM // TILE
WINDOW = 128  # the ring's slots
WINDOW_TILES = 1 + WINDOW // TILE + 1  # the sink's zero tile, the ring, the block's
RING_FIRST, BLOCK_ROW = 1, WINDOW_TILES - 1
WHOLE = TILE  # the insert's variant of the whole tile
VARIANTS = TILE + 1
# The config's rotary numbers (config.json): a layer of compress ratio 0 rotates at
# rope_theta without YaRN, the others at compress_rope_theta with it.
ROPE_ARGS = {
    "rope_theta": 10000.0,
    "compress_rope_theta": 160000.0,
    "original_seq_len": 65536,
    "rope_factor": 16.0,
    "beta_fast": 32,
    "beta_slow": 1,
}


@km.tile_function
def mix(x, y, c, s):
    return x * c + y * s


@km.tile_function
def keep_put(old, m, y):
    return old * m + y


class Window(km.Program):
    """Two commands each, a product then the element-wise rest. rope_swap: y's tile row
    h = t's tiles 14, 15 of row h times the swap; rope: t = t c + y s in place.
    insert_move: y's tile c = e kv[0:1, c]; insert: the window's tile (row, c) = old m +
    y."""

    pair_sums: km.L1[km.fp32, 1, ROPE_TILES, 32, 32]
    one_sum: km.L1[km.fp32, 1, 1, 32, 32]

    @km.tile_function(fidelity="HiFi3")
    def project(x, w):
        return x @ w

    @km.handler
    def rope_swap(self, t: km.Tensor, swap: km.Tensor, y: km.Tensor):
        h = self.rank
        for xb, wb in zip(
            km.Stream(
                t.tiles[h : h + 1, ROPE_FIRST : ROPE_FIRST + ROPE_TILES],
                block=ROPE_TILES,
            ),
            km.Stream(
                swap.tiles[0:ROPE_TILES, 0:ROPE_TILES], block=ROPE_TILES * ROPE_TILES
            ),
        ):
            self.pair_sums += self.project(xb, wb)
        whole = km.reduce(self.pair_sums, form="tree", group=1)
        if self.rank == self.leader:
            y.pages(h * ROPE_TILES, ROPE_TILES).write(whole)

    @km.handler
    def rope(self, t: km.Tensor, y: km.Tensor, cos: km.Tensor, sin: km.Tensor, at: int):
        h = self.rank
        out = km.Stream(
            t.tiles[h : h + 1, ROPE_FIRST : ROPE_FIRST + ROPE_TILES],
            write=True,
            block=ROPE_TILES,
        )
        for x, ys, c, s in zip(
            km.Stream(
                t.tiles[h : h + 1, ROPE_FIRST : ROPE_FIRST + ROPE_TILES],
                block=ROPE_TILES,
            ),
            km.Stream(y.tiles[h : h + 1, 0:ROPE_TILES], block=ROPE_TILES),
            km.Stream(cos.tiles[at : at + 1, 0:ROPE_TILES], block=ROPE_TILES),
            km.Stream(sin.tiles[at : at + 1, 0:ROPE_TILES], block=ROPE_TILES),
        ):
            out.push(mix(x, ys, c, s))

    @km.handler
    def insert_move(
        self, kv: km.Tensor, move: km.Tensor, moved: km.Tensor, variant: int
    ):
        c = self.rank
        for xb, wb in zip(
            km.Stream(move.tiles[variant : variant + 1, 0:1], block=1),
            km.Stream(kv.tiles[0:1, c : c + 1], block=1),
        ):
            self.one_sum += self.project(xb, wb)
        whole = km.reduce(self.one_sum, form="tree", group=1)
        if self.rank == self.leader:
            moved.pages(c, 1).write(whole)

    @km.handler
    def insert(
        self,
        window: km.Tensor,
        moved: km.Tensor,
        keep: km.Tensor,
        row: int,
        variant: int,
    ):
        c = self.rank
        out = km.Stream(window.tiles[row : row + 1, c : c + 1], write=True, block=1)
        for old, m, ys in zip(
            km.Stream(window.tiles[row : row + 1, c : c + 1], block=1),
            km.Stream(keep.tiles[variant : variant + 1, 0:1], block=1),
            km.Stream(moved.pages(c, 1), block=1),
        ):
            out.push(keep_put(old, m, ys))


def swap_tiles():
    """The swap [64, 64]: P on the diagonal (x P swaps each adjacent pair of x's
    columns), zero off it."""
    p = np.zeros((ROPE_DIM, ROPE_DIM), np.float32)
    p[np.arange(0, ROPE_DIM, 2), np.arange(1, ROPE_DIM, 2)] = 1.0
    p[np.arange(1, ROPE_DIM, 2), np.arange(0, ROPE_DIM, 2)] = 1.0
    return p


def frequencies(layer_compressed, args=ROPE_ARGS):
    """model.py precompute_freqs_cis's angles' frequencies [32] of a layer: theta
    rope_theta and no YaRN for a layer of compress ratio 0, else compress_rope_theta
    with YaRN (original_seq_len, rope_factor, beta_fast, beta_slow); float32 as torch.
    """
    dim = ROPE_DIM
    if layer_compressed:
        base, original = float(args["compress_rope_theta"]), int(
            args["original_seq_len"]
        )
    else:
        base, original = float(args["rope_theta"]), 0
    exponent = np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim)
    freqs = (np.float32(1.0) / (np.float32(base) ** exponent)).astype(np.float32)
    if original > 0:
        factor = float(args["rope_factor"])

        def corrected(rotations):
            return dim * np.log(original / (rotations * 2 * np.pi)) / (2 * np.log(base))

        low = max(int(np.floor(corrected(float(args["beta_fast"])))), 0)
        high = min(int(np.ceil(corrected(float(args["beta_slow"])))), dim - 1)
        ramp = np.clip(
            (np.arange(dim // 2, dtype=np.float32) - low) / max(high - low, 1e-3), 0, 1
        ).astype(np.float32)
        smooth = np.float32(1) - ramp
        freqs = (freqs / np.float32(factor) * (1 - smooth) + freqs * smooth).astype(
            np.float32
        )
    return freqs


def angles(freqs, positions):
    """The angles [positions, 32] (torch.outer in float32)."""
    return np.outer(np.arange(positions, dtype=np.float32), freqs).astype(np.float32)


def tables(freqs, positions):
    """The rope tables' values, each [positions x 32, 64] (tile row t: positions t ..
    t + 31): cos, the forward rotation's signed sin, the inverse's."""
    theta = angles(freqs, positions + TILE).astype(np.float64)
    cos = np.repeat(np.cos(theta), 2, axis=1)
    sin = np.repeat(np.sin(theta), 2, axis=1)
    sin[:, 0::2] *= -1.0  # x0 c - x1 s, x1 c + x0 s
    index = np.arange(positions)[:, None] + np.arange(TILE)[None, :]
    slide = lambda values: values[index].reshape(-1, ROPE_DIM)  # noqa: E731
    return slide(cos), slide(sin), slide(-sin)


def insert_tables():
    """The insert's tables [33 x 32, 32] each: keep (m) and move (e) of each variant."""
    keep = np.ones((VARIANTS, TILE, TILE), np.float32)
    move = np.zeros((VARIANTS, TILE, TILE), np.float32)
    for r in range(TILE):
        keep[r, r, :] = 0.0
        move[r, r, 0] = 1.0
    keep[WHOLE] = 0.0
    move[WHOLE] = np.eye(TILE, dtype=np.float32)
    return keep.reshape(-1, TILE), move.reshape(-1, TILE)


def rotate(x, freqs, start, inverse=False):
    """float64 apply_rotary_emb of x [..., tokens, 512]'s last 64 at positions start ..
    start + tokens - 1."""
    x = np.array(x, np.float64)
    tokens = x.shape[-2]
    theta = angles(freqs, start + tokens)[start:].astype(np.float64)
    if inverse:
        theta = -theta
    pairs = x[..., -ROPE_DIM:].reshape(*x.shape[:-1], ROPE_DIM // 2, 2)
    re, im = pairs[..., 0], pairs[..., 1]
    c, s = np.cos(theta), np.sin(theta)
    out = np.stack([re * c - im * s, re * s + im * c], axis=-1)
    x[..., -ROPE_DIM:] = out.reshape(*x.shape[:-1], ROPE_DIM)
    return x


def ring_row(pos):
    """The window's tile row and the row in it of position pos."""
    slot = pos % WINDOW
    return RING_FIRST + slot // TILE, slot % TILE
