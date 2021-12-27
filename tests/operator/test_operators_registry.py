import pytest

from ezcv.operator import get_available_operators, Operator, register_operator
from ezcv.pipeline import PipelineContext
from ezcv.test_utils import assert_terms_in_exception
from ezcv.typing import Image


@register_operator
class Operator1(Operator):
    def run(self, img: Image, ctx: PipelineContext) -> Image:
        pass


@register_operator
class Operator2(Operator):
    def run(self, img: Image, ctx: PipelineContext) -> Image:
        pass


class TestGetAvailableOperators:
    def test_return_type(self):
        r = get_available_operators()
        assert iter(r) is not None

    def test_length(self):
        assert len(get_available_operators()) == 2

    def test_operators(self):
        r = get_available_operators()
        assert set(r) == {Operator1, Operator2}


class TestRegisterOperator:
    def test_invalid_class(self):
        class NotAnOperator(object):
            pass

        with pytest.raises(ValueError) as e:
            register_operator(NotAnOperator)
        assert_terms_in_exception(e, ['not', 'operator'])
