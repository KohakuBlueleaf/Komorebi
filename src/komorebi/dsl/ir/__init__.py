"""The DSL as a frontend of the Komorebi IR: a class's handlers as an IR module's text,
its origin beside it (docs/ir/frontends.md). The walk does not import `komorebi_ir`;
`built.module` lays its text out and `built.program` compiles it. Today's make calls
neither: the class compiler's path to C++ is unchanged."""

from komorebi.dsl.ir.walk import Made, written

__all__ = ["Made", "written"]
