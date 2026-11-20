"""Host (CPU) stand-ins for the GPU kernels of the model's reference code: the same six
functions, the same arithmetic (README.md, `reference/cpu_kernels.py`)."""

import torch

FP8_MAX = 448.0
FP4_MAX = 6.0
# E2M1 magnitudes by code 0..7 (code 8..15 are the negatives)
FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
FP4_LUT = torch.cat([FP4_GRID, -FP4_GRID])


def _pow2_ceil(t: torch.Tensor) -> torch.Tensor:
    """2 ** ceil(log2(t)) of positive normal float32 values, by their bits (the kernel's
    fast_log2_ceil and fast_pow2)."""
    bits = t.contiguous().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return ((exp + 127) << 23).view(torch.float32)


def _round_scale(amax: torch.Tensor, max_value: float) -> torch.Tensor:
    inv = torch.tensor(1.0 / max_value, dtype=torch.float32)
    return _pow2_ceil(amax * inv)


def fp4_round(x: torch.Tensor) -> torch.Tensor:
    """Round float32 values in [-6, 6] to the nearest E2M1 value, ties to even code."""
    a = x.abs()
    grid = FP4_GRID
    mids = (grid[1:] + grid[:-1]) / 2  # 7 midpoints
    # bucketize (right=False): a value equal to midpoint k gets index k, its lower
    # neighbour
    idx = torch.bucketize(a, mids)
    # a value exactly on a midpoint goes to the even code of its two neighbours k and k
    # + 1
    eq = torch.isin(a, mids)
    if eq.any():
        k = idx[eq]
        idx[eq] = torch.where(k % 2 == 0, k, k + 1)
    return torch.copysign(grid[idx], x)


def act_quant(
    x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False
):
    """Block-wise FP8 (E4M3) quantization along the last axis. inplace=True writes the
    dequantized values back into x (in x's dtype) and returns x; otherwise returns the
    E4M3 values and the scales (shape [..., N // block_size])."""
    N = x.size(-1)
    assert N % block_size == 0
    xf = x.float().unflatten(-1, (N // block_size, block_size))
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    if scale_fmt is not None:
        s = _round_scale(amax, FP8_MAX)
    else:
        s = amax * torch.tensor(1.0 / FP8_MAX, dtype=torch.float32)
    y = (xf / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((y.float() * s).flatten(-2).to(x.dtype))
        return x
    return y.flatten(-2), s.squeeze(-1).to(scale_dtype)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    """Block-wise FP4 (E2M1) quantization with power-of-two (E8M0) or E4M3 scales. Only
    the in-place form (quantize, then dequantize back into x) is used by the model."""
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    if not inplace:
        raise NotImplementedError(
            "the model only uses fp4_act_quant(..., inplace=True)"
        )
    N = x.size(-1)
    assert N % block_size == 0
    xf = x.float().unflatten(-1, (N // block_size, block_size))
    amax = xf.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(6 * 2.0**-9)
        s = (amax / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(6 * 2.0**-126)
        s = _round_scale(amax, FP4_MAX)
    y = fp4_round((xf / s).clamp(-FP4_MAX, FP4_MAX))
    x.copy_((y * s).flatten(-2).to(x.dtype))
    return x


def dequant_fp8_weight(
    b: torch.Tensor, b_s: torch.Tensor, block_size: int = 32
) -> torch.Tensor:
    """E4M3 weight [N, K] with one E8M0 scale per block_size x block_size block ->
    bfloat16 (every product of an E4M3 value and a power of two is exact in bfloat16).
    """
    N, K = b.shape
    nb, kb = -(-N // block_size), -(-K // block_size)
    w = b.to(torch.bfloat16)
    s = b_s.float().to(torch.bfloat16)
    if N % block_size == 0 and K % block_size == 0:
        return (w.view(nb, block_size, kb, block_size) * s[:, None, :, None]).view(N, K)
    s = s.repeat_interleave(block_size, 0)[:N].repeat_interleave(block_size, 1)[:, :K]
    return w * s


def _e2m1_bf16_bits(c: int) -> int:
    """The bfloat16 bit pattern of E2M1 code c: sign bit 3, exponent field bits 2..1,
    mantissa bit 0; exponent field 0 is 0 or 0.5."""
    s, e, m = (c >> 3) & 1, (c >> 1) & 3, c & 1
    mag = ((e + 126) << 7) | (m << 6) if e > 0 else m * (126 << 7)
    return mag | (s << 15)


# a byte of two E2M1 codes -> the two bfloat16 patterns in one int32 (little-endian: low
# nibble first)
_FP4_PAIR_LUT = torch.tensor(
    [_e2m1_bf16_bits(v & 15) | (_e2m1_bf16_bits(v >> 4) << 16) for v in range(256)],
    dtype=torch.int64,
).to(torch.int32)


def dequant_fp4_weight(
    b: torch.Tensor, b_s: torch.Tensor, block_size: int = 32
) -> torch.Tensor:
    """E2M1 weight stored [N, K // 2] (two codes a byte, the low nibble first along K)
    with one E8M0 scale per block_size along K -> bfloat16 [N, K] (exact)."""
    u = b.view(torch.uint8)
    N = u.size(0)
    vals = (
        _FP4_PAIR_LUT.index_select(0, u.flatten().to(torch.int32))
        .view(torch.bfloat16)
        .view(N, -1)
    )
    K = vals.size(1)
    s = b_s.float().to(torch.bfloat16)
    return (vals.view(N, K // block_size, block_size) * s.unsqueeze(-1)).view(N, K)


def _dequant_act(a: torch.Tensor, a_s: torch.Tensor, block: int) -> torch.Tensor:
    """E4M3 activations with a power-of-two scale per block along the last axis ->
    bfloat16 (exact)."""
    K = a.size(-1)
    A = a.float().unflatten(-1, (K // block, block)) * a_s.float().unsqueeze(-1)
    return A.flatten(-2).to(torch.bfloat16)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    """C[M, N] = A[M, K] @ B[N, K]^T, A and B in E4M3 with per-block scales. Both
    operands are exact in bfloat16; the product of two bfloat16 matrices on the CPU sums
    in float32.
    """
    A = _dequant_act(a, a_s, block_size)
    B = dequant_fp8_weight(b, b_s, block_size)
    return (A @ B.t()).to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    """C[M, N] = A_fp8[M, K] @ B_fp4[N, K]^T (exact bfloat16 operands, float32 sums)."""
    A = _dequant_act(a, a_s, act_block_size)
    B = dequant_fp4_weight(b, b_s, 32)
    return (A @ B.t()).to(torch.get_default_dtype())


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """o[b, m, h, :] = softmax over the entries topk_idxs[b, m, :] of kv[b] (with a sink
    per head) of (q . kv) * scale, times kv. q [b, m, h, d] bf16, kv [b, n, d] bf16,
    topk_idxs int32 with -1 for no entry. Same block order (64 entries a block) and
    roundings as the GPU kernel.
    """
    b, m, h, d = q.shape
    topk = topk_idxs.size(-1)
    block = 64
    out = torch.empty_like(q)
    sink = attn_sink.float()
    for bi in range(b):
        idx = topk_idxs[bi].long()  # [m, topk]
        valid = idx != -1
        g = kv[bi][idx.clamp(min=0)].float() * valid.unsqueeze(
            -1
        )  # [m, topk, d]; invalid rows 0
        qf = q[bi].float()  # [m, h, d]
        s = torch.einsum("mhd,mtd->mht", qf, g) * softmax_scale
        s = s.masked_fill(~valid.unsqueeze(1), float("-inf"))
        run_max = torch.full((m, h), -1e30, dtype=torch.float32)
        sum_exp = torch.zeros((m, h), dtype=torch.float32)
        acc = torch.zeros((m, h, d), dtype=torch.float32)
        for t0 in range(0, topk, block):
            sb = s[:, :, t0 : t0 + block]
            vb = g[:, t0 : t0 + block]
            prev = run_max
            run_max = torch.maximum(prev, sb.amax(-1))
            corr = torch.exp(prev - run_max)
            p = torch.exp(sb - run_max.unsqueeze(-1))
            sum_exp = sum_exp * corr + p.sum(-1)
            acc = acc * corr.unsqueeze(-1) + torch.einsum(
                "mht,mtd->mhd", p.to(torch.bfloat16).float(), vb
            )
        sum_exp = sum_exp + torch.exp(sink.unsqueeze(0) - run_max)
        out[bi] = (acc / sum_exp.unsqueeze(-1)).to(q.dtype)
    return out


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """Split the 24 mixes of a token into pre (4), post (4) and comb (4 x 4,
    Sinkhorn-normalised)."""
    hc = hc_mult
    mixes = mixes.float()
    scale = hc_scale.float()
    base = hc_base.float()
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * scale[2] + base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = torch.exp(comb - comb.amax(-1, keepdim=True))
    comb = comb / comb.sum(-1, keepdim=True) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
