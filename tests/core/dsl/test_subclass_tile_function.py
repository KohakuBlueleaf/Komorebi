"""A tile function of the class's body (`self.f(...)`) redefined in a subclass: the
subclass's handlers, its parent's among them, call its own; a grandchild's, a merged
class's (each handler its own class's, two of one name built apart) and a chain
build's the same way (no card)."""

from class_constants_fixture import grid

import komorebi as km

TENSORS = {"x": grid(1, 2), "y": grid(1, 2)}


def quiet(line):
    pass


class Twice(km.Program):
    @km.tile_function
    def act(a):
        return a + a

    @km.handler
    def apply(self, x: km.Tensor, y: km.Tensor):
        out = km.Stream(y.tiles[0:1, 0:2], write=True)
        for t in km.Stream(x.tiles[0:1, 0:2]):
            out.push(self.act(t))


class Squared(Twice):
    @km.tile_function
    def act(a):
        return a * a


class Kept(Squared):  # a grandchild redefining nothing: Squared's
    pass


class Cubed(Squared):  # a grandchild redefining it again
    @km.tile_function
    def act(a):
        return a * a * a


class Square(km.Program):  # Squared's tile function declared with its handler
    @km.tile_function
    def act(a):
        return a * a

    @km.handler
    def apply(self, x: km.Tensor, y: km.Tensor):
        out = km.Stream(y.tiles[0:1, 0:2], write=True)
        for t in km.Stream(x.tiles[0:1, 0:2]):
            out.push(self.act(t))


class Cube(km.Program):
    @km.tile_function
    def act(a):
        return a * a * a

    @km.handler
    def apply(self, x: km.Tensor, y: km.Tensor):
        out = km.Stream(y.tiles[0:1, 0:2], write=True)
        for t in km.Stream(x.tiles[0:1, 0:2]):
            out.push(self.act(t))


class Halves(km.Program):  # another class whose tile function is named act
    @km.tile_function
    def act(a):
        return a * 0.5

    @km.handler
    def halve(self, x: km.Tensor, y: km.Tensor):
        out = km.Stream(y.tiles[0:1, 0:2], write=True)
        for t in km.Stream(x.tiles[0:1, 0:2]):
            out.push(self.act(t))


class Plain(Twice):  # Twice's handler and act: beside Squared, a common base
    pass


class Halving(km.Program):  # a tile function of another name
    @km.tile_function
    def half(a):
        return a * 0.5

    @km.handler
    def halve(self, x: km.Tensor, y: km.Tensor):
        out = km.Stream(y.tiles[0:1, 0:2], write=True)
        for t in km.Stream(x.tiles[0:1, 0:2]):
            out.push(self.half(t))


class Chained(Squared):
    @km.chain
    def twice(self, x, y):
        self.apply(x, y)
        self.apply(y, x)


def built(program):
    """The tile programs' text of a made program, by file name."""
    folder = program.built.folder
    return {f.name: f.read_text() for f in sorted(folder.glob("tile_*.txt"))}


def made(cls, **options):
    return cls(l1=1 << 20, log=quiet, **TENSORS, **options)


def test_a_subclass_s_tile_function_is_the_one_its_parent_s_handler_calls():
    twice, squared = built(made(Twice)), built(made(Squared))
    assert squared != twice
    assert squared == built(made(Square))


def test_a_grandchild_calls_the_nearest_tile_function():
    assert built(made(Kept)) == built(made(Square))
    assert built(made(Cubed)) == built(made(Cube))


def test_a_merged_class_calls_each_parent_s_own_tile_function():
    class Both(Squared, Halving):
        pass

    texts = set(built(made(Both)).values())
    assert set(built(made(Square)).values()) <= texts
    assert set(built(made(Halving)).values()) <= texts


def unit(texts, name):
    """The tile program of `texts` that is `name`'s text under its own name."""
    return {text.replace(name, "act", 1) for text in texts}


def test_two_classes_tile_functions_of_one_name_are_built_apart():
    class Both(Squared, Halves):
        pass

    square = set(built(made(Square)).values())
    halves = set(built(made(Halves)).values())
    for program in (made(Both), km.merge(made(Squared), made(Halves), log=quiet)):
        texts = built(program)
        assert sorted(texts) == ["tile_act.txt", "tile_act_.txt"]
        assert {texts["tile_act.txt"]} == square
        assert unit([texts["tile_act_.txt"]], "act_") == halves


def test_a_common_base_s_handler_runs_each_parent_s_act():
    class Both(Squared, Plain):  # apply: Twice's, in both; act Squared's and Twice's
        pass

    program = made(Both)
    assert sorted(program.handlers) == ["plain_apply", "squared_apply"]
    texts = built(program)
    assert {texts["tile_act.txt"]} == set(built(made(Square)).values())
    assert unit([texts["tile_act_.txt"]], "act_") == set(built(made(Twice)).values())


def test_a_chain_build_calls_the_subclass_s_tile_function():
    assert built(made(Chained)) == built(made(Square, chain=True))
