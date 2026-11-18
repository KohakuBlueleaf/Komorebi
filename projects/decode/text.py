"""Detokenizing on the card: each token id's bytes, as the thread's tables hold them
(the byte-level BPE vocabulary of tokenizer.json, the added tokens as their text)."""

import json
from pathlib import Path

import numpy as np


def byte_of_char():
    """The byte-level alphabet's inverse: {character: byte} (GPT-2's mapping)."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars, extra = {}, 0
    for byte in range(256):
        if byte in printable:
            chars[chr(byte)] = byte
        else:
            chars[chr(256 + extra)] = byte
            extra += 1
    return chars


def pieces(snapshot, vocab):
    """Every id's bytes for `vocab` ids (empty where the tokenizer has none):
    (offsets uint32 [vocab + 1], bytes uint8), id i's in bytes[offsets[i]:offsets[i+1]].
    """
    tokenizer = json.loads((Path(snapshot) / "tokenizer.json").read_text())
    chars = byte_of_char()
    table = [b""] * vocab
    for piece, index in tokenizer["model"]["vocab"].items():
        if index < vocab:
            table[index] = bytes(chars[c] for c in piece)
    for added in tokenizer.get("added_tokens", ()):
        if added["id"] < vocab:
            table[added["id"]] = added["content"].encode()
    offsets = np.zeros(vocab + 1, np.uint32)
    offsets[1:] = np.cumsum([len(piece) for piece in table])
    return offsets, np.frombuffer(b"".join(table), np.uint8).copy()


def joined(offsets, data, ids):
    """The bytes of `ids` from the tables, as the thread writes them."""
    return b"".join(data[offsets[i] : offsets[i + 1]].tobytes() for i in ids)
