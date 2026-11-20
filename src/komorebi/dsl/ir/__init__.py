"""The DSL as a frontend of the Komorebi IR: a class's handlers as an IR module's text,
its origin beside it (docs/ir/frontends.md), for `komorebi_ir.compile`, which it does
not import (`built.program` does: the class's Program through the IR). Today's make
calls neither: the class compiler's path to C++ is unchanged."""

from komorebi.dsl.ir.walk import Made, module

__all__ = ["Made", "module"]
