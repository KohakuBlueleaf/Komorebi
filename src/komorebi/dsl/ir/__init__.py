"""The DSL as a frontend of the Komorebi IR: a class's handlers as an IR module's text,
its origin beside it (docs/ir/frontends.md). It does not import komorebi_ir, which
compiles what it gives: `komorebi_ir.compile(made.text, made.origin)`. Today's make
does not call it: the class compiler's path to C++ is unchanged."""

from komorebi.dsl.ir.walk import Made, module

__all__ = ["Made", "module"]
