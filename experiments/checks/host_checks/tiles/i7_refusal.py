import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from stand_in import Pages

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.front.nodes import prune, render

C = 0.7978845608028654


@km.tile_function
def gelu_tanh(x):
    return 0.5 * x * (1.0 + km.tanh(C * (x + 0.044715 * x * x * x)))


@km.tile_function
def gelu_sigmoid(x):
    return x * km.sigmoid(2.0 * C * (x + 0.044715 * x * x * x))


@km.tile_function
def down(x):
    return x * 0.0 + km.reduce_sum(x, dim=-2, exact=True).col()


def cls_for(fn):
    class G(km.Program):
        @km.handler
        def run(self, x: km.Tensor, out: km.Tensor):
            results = km.Stream(out.pages(0, 2), write=True)
            for t in km.Stream(x.pages(0, 2)):
                results.push(fn(t))

    return G


def tensors():
    return {n: Tensor(Pages(4096), (64, 32), "float32", "tile") for n in ("x", "out")}


for name, fn in (("tanh", gelu_tanh), ("sigmoid", gelu_sigmoid), ("dim=-2", down)):
    try:
        compile_class(cls_for(fn), {}, tensors())
        print(f"{name}: BUILT")
    except Exception as e:  # noqa: BLE001
        text = str(e)
        print(f"{name}: {type(e).__name__}: ...{text[text.index('tile compiler'):]}")


class Lines(km.Program):
    @km.handler
    def run(self, n: int):
        km.cxx("one_line();")  # LINE_ONE
        km.cxx("""
            three_quoted();
            """)  # the text is on the line after the opening quotes


compiled = compile_class(Lines, {}, {})
src = Path(__file__).read_text().split("\n")
one = next(i for i, line in enumerate(src, 1) if "LINE_ONE" in line)
three = next(i for i, line in enumerate(src, 1) if "three_quoted();" in line)
code = "\n".join(render(prune(compiled.compute["run"]), 0))

for want, call in ((one, "one_line();"), (three, "three_quoted();")):
    at = code.index("\n" + call)
    line = code[:at].rsplit("#line ", 1)[1].split(" ")[0]
    print(
        f"{call} at source line {want}: #line {line}",
        "OK" if int(line) == want else "WRONG",
    )
