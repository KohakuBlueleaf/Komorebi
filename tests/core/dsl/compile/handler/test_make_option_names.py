"""A tensor parameter named like one of the class's make options (no card): refused at
the parameter, naming the options; the options are Program.make's keywords; an int
word of such a name is a command word and builds."""

import inspect

import pytest
from class_lines_fixture import Pages

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl import program
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compile.source import MAKE_OPTIONS
from komorebi.dsl.compiler import compile_class


class Deep(km.Program):
    @km.handler
    def run(self, depth: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 2), write=True)
        for tile in km.Stream(depth.pages(0, 2)):
            results.push(tile)


class Counted(km.Program):
    @km.handler
    def run(self, x: km.Tensor, out: km.Tensor, depth: int):
        results = km.Stream(out.pages(0, depth), write=True)
        for tile in km.Stream(x.pages(0, depth)):
            results.push(tile)


def tensors(*names):
    return {n: Tensor(Pages(4096), (64, 32), "float32", "tile") for n in names}


def test_the_options_are_make_s_keywords():
    parameters = inspect.signature(program.make).parameters.values()
    keywords = [
        p.name
        for p in parameters
        if p.name != "self" and p.kind is not inspect.Parameter.VAR_KEYWORD
    ]
    assert tuple(keywords) == MAKE_OPTIONS


def test_a_tensor_named_like_an_option_is_refused_by_name():
    with pytest.raises(Refused, match="`depth` is one of the class's make options"):
        compile_class(Deep, {}, tensors("depth", "out"))


def test_an_int_word_of_that_name_builds():
    compiled = compile_class(Counted, {}, tensors("x", "out"))
    assert "run" in compiled.handlers
