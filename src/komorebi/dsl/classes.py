"""km.Program: the program description that a class deriving it with handler methods
makes a program written as a class (komorebi.dsl.program), or, deriving two such
classes, their merge (komorebi.dsl.merged)."""

from komorebi.dsl import first_use, merged
from komorebi.dsl import program as made
from komorebi.program import description


class Program(description.Program):
    """A program: given its kernels, an ordinary program; derived by a class with
    handler methods, a program written as a class, compiled when it is made."""

    def __init_subclass__(cls, compute=None, defines=None, startup=None, **kwargs):
        """`compute` its compute kernel's options by the host API's names, `defines`
        its kernels' defines, `startup` compute's start-up (km.Startup)."""
        super().__init_subclass__(**kwargs)
        parents = [
            base for base in cls.__bases__ if getattr(base, "built_handlers", None)
        ]
        if len(parents) > 1:  # class Layer(Attention, Linear): the classes merged
            merged.prepare(cls, parents, compute, defines, startup)
        else:
            made.prepare(cls, compute, defines, startup)

    @classmethod
    def merge_made(cls, sources, name, overlay, resident, session, l1, log):
        """km.merge of made classes in the kernel form: one program of them."""
        return merged.merge_made(sources, name, overlay, resident, session, l1, log)

    def first_use_ahead(self, handler, wanted):
        """(name, parts, run.addon's keywords) of a handler's set built ahead."""
        return first_use.ahead(self, handler, wanted)
