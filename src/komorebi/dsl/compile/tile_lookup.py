"""Which tile function a call is, and a compute tile line that is not a tile
function's call (refused in a loop that calls one)."""

import ast
import inspect

from komorebi.dsl.compile.records import Value
from komorebi.dsl.compile.source import refuse
from komorebi.dsl.front import front


def other_tile_line(frame, node):
    """A tile line of compute that is not a tile function's call: refused in a
    loop that calls one (its setup holds the vector unit's registers)."""
    if not frame.loop_states:
        return
    state = frame.loop_states[-1]
    if state["function"] is not None:
        refuse(
            frame.where,
            node,
            f"this loop calls the tile function {state['function']}, whose setup "
            "holds the vector unit for the loop",
            "write the loop's tile arithmetic in that tile function",
        )
    state["other"] = node


def tile_function_of(frame, node):
    """The tile function a call is (`self.f(...)` of a @km.tile_function of the
    class, or `f(...)` of the module's), or None."""
    if not isinstance(node, ast.Call):
        return None
    if id(node) in frame.expression_calls:
        return frame.expression_calls[id(node)][1]
    func = node.func
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "self"
    ):
        for owner in inspect.getmro(frame.method.owner):
            found = vars(owner).get(func.attr)
            if found is not None:
                return found if isinstance(found, front.TileFunction) else None
        return None
    if isinstance(func, ast.Name) and func.id not in frame.scope:
        found = frame.names.get(func.id)
        if isinstance(found, front.TileFunction):
            return found
        if (
            inspect.isfunction(found)
            and node.args
            and all(
                isinstance(arg, ast.Name)
                and frame.scope.get(arg.id, Value("")).kind == "streamed"
                for arg in node.args
            )
        ):  # a plain function of a stream's tiles
            return front.TileFunction(found)
    return None
