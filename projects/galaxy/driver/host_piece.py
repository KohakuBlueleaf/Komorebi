"""The reference as a box piece (README.md, `driver/host_piece.py`)."""

from ..reference.model_loader import Checkpoint, load_reference
from .path import TokenPath


class HostPiece:
    sites = (
        "wq_a",
        "wq_b",
        "wkv",
        "wo_a",
        "wo_b",
        "q_norm",
        "kv_norm",
        "attn_norm",
        "ffn_norm",
        "rope_q",
        "rope_kv",
        "rope_o",
        "sparse_attn",
        "gate",
        "shared",
        "routed",
        "hc_attn_mixes",
        "hc_ffn_mixes",
        "hc_attn_pre",
        "hc_ffn_pre",
        "hc_attn_post",
        "hc_ffn_post",
        "engram_mix",
        "hc_head_pre",
        "final_norm",
        "head",
        "choose",
    )

    def __init__(self, ckpt_dir):
        self.ref = load_reference(ckpt_dir, max_seq_len=4096)
        self.path = TokenPath(self.ref)

    def __call__(self, site, layer, *x):
        m, mod = self.ref.model, self.ref.mod
        if layer is None:
            last = m.layers[-1]
            return {
                "hc_head_pre": last.hc_pre,
                "final_norm": m.norm,
                "head": m.head,
                "choose": lambda lg: lg.argmax(dim=-1),
            }[site](*x)
        blk = m.layers[layer]
        attn, ffn = blk.attn, blk.ffn
        if site in ("wq_a", "wq_b", "wkv", "wo_b", "q_norm", "kv_norm"):
            return getattr(attn, site)(*x)
        if site in ("attn_norm", "ffn_norm"):
            return getattr(blk, site)(*x)
        if site in ("rope_q", "rope_kv", "rope_o"):
            return mod.apply_rotary_emb(x[0].clone(), x[1], site == "rope_o")
        if site == "wo_a":
            return self.path.wo_a_ref(attn, *x)
        if site == "sparse_attn":
            return mod.sparse_attn(*x)
        if site == "gate":
            return ffn.gate(*x)
        if site == "shared":
            return ffn.shared_experts(*x)
        if site == "routed":
            return TokenPath.routed_ref(ffn, *x)
        if site.endswith("_mixes"):
            k = site.split("_")[1]
            return blk.hc_mixes(
                x[0],
                getattr(blk, f"hc_{k}_fn"),
                getattr(blk, f"hc_{k}_scale"),
                getattr(blk, f"hc_{k}_base"),
            )
        if site.endswith("_pre"):
            return blk.hc_pre(*x)
        if site.endswith("_post"):
            return blk.hc_post(*x)
        if site == "engram_mix":
            return TokenPath.engram_mix_ref(blk.engram, *x)
        raise KeyError(site)


def make(session, weights, layers):
    assert isinstance(
        weights, Checkpoint
    ), "host_piece needs W = the Checkpoint (verify --ckpt)"
    return HostPiece(weights.dir)
