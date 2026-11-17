"""A chain call's description from its handler's lines (chain_calls describes each
call; gather describes km.all_gather's record)."""

from dataclasses import dataclass, field


@dataclass
class Described:
    """A call's description from its handler's lines (tensors by their object)."""

    handler: object
    words: list
    group: tuple
    size: int
    width: int
    output: object = None
    reads: dict = field(default_factory=dict)  # tensor -> (k, ranges, lines)
    sends: bool = False
    prefetch: dict = field(default_factory=dict)  # tensor -> [(line, True/False)]
    marks: dict = field(default_factory=dict)  # tensor -> "past" or "direct"
    name: str = None
    tensor_bits: list = field(default_factory=list)  # (tensor, its parameter's bit)
    mask_word: int = None
    outputs: list = field(default_factory=list)  # its other outputs: (tensor, width)
    body_reads: dict = field(default_factory=dict)  # tensor id -> (tensor, lines)
    body_writes: dict = field(default_factory=dict)  # tensor id -> (tensor, lines)
    unranged: dict = field(default_factory=dict)  # tensor -> why it is no range
    program: object = None  # the program whose handler it calls
    streamed: int = None  # the line of a write stream of its output, if any
    covering: dict = field(default_factory=dict)  # tensor id -> line of a cover range
    rows: dict = field(default_factory=dict)  # output id -> (r0, r1) its roots write
    row_reads: dict = field(default_factory=dict)  # tensor id -> RowRead (columns)
    in_place: bool = False  # its handler's direct input lands in its stream's halves
    submitted: tuple = None  # (name, its word's place 0 to 9): km.at_submit's word
    word_tensors: dict = field(default_factory=dict)  # a tensor word's place -> tensor
    gather: tuple = None  # km.all_gather's: each island's first rank
    by_island: tuple = None  # a write by island: each island's first rank


def writes_key(one, key):
    """Whether described call `one` writes the tensor of id `key` (an output)."""
    outputs = [one.output, *(tensor for tensor, _ in one.outputs)]
    return any(tensor is not None and id(tensor) == key for tensor in outputs)
