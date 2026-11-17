"""What test_pieces_meet.py and other tests share: made."""

from class_handler_constants_fixture import TENSORS, quiet


def made(cls, **given):
    return cls(l1=1 << 20, log=quiet, **TENSORS, **given)
