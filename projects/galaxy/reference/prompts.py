"""The fixed prompts `en`, `zh` (32 tokens each) and `long` (README.md,
`reference/prompts.py`)."""

import os
import sys

PROMPTS = {
    "en": "Explain in two short sentences why the sky looks blue during the day but "
    "turns red and orange at sunset, close to the far horizon line.",
    "zh": "中国的首都是哪里？请用两三句话，简"
    "单介绍一下这座古老城市的悠久历史、传统"
    "文化、地方美食以及它著名的名胜古迹。",
}


def encode(ref, ckpt_dir: str, text: str) -> list[int]:
    """The token ids of one user turn in the model's chat format (as generate.py)."""
    enc = os.path.join(ckpt_dir, "encoding")
    if enc not in sys.path:
        sys.path.insert(0, enc)
    from encoding import encode_messages  # noqa: PLC0415 (the checkpoint's encoding/)

    return ref.tokenizer.encode(
        encode_messages([{"role": "user", "content": text}], thinking_mode="chat")
    )


LONG_TOKENS = 1100
LONG_HEAD = "Summarize the following text in three sentences.\n\n"


def long_prompt(ref, ckpt_dir: str) -> str:
    """The `long` prompt's text: the longest prefix of README.md whose prompt is at most
    LONG_TOKENS tokens."""
    with open(os.path.join(ckpt_dir, "README.md"), encoding="utf-8") as f:
        body = f.read()
    lo, hi = 0, len(body)
    while lo < hi:  # the largest cut with at most LONG_TOKENS tokens
        mid = (lo + hi + 1) // 2
        if len(encode(ref, ckpt_dir, LONG_HEAD + body[:mid])) <= LONG_TOKENS:
            lo = mid
        else:
            hi = mid - 1
    text = LONG_HEAD + body[:lo]
    assert len(encode(ref, ckpt_dir, text)) > 1024
    return text


def prompt_text(ref, ckpt_dir: str, key: str) -> str:
    return long_prompt(ref, ckpt_dir) if key == "long" else PROMPTS[key]
