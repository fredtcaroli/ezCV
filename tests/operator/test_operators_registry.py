import pytest

from ezcv.operator import get_available_operators, Operator, register_operator
from ezcv.test_utils import assert_terms_in_exception


def test_get_available_operators_return():
    r = get_available_operators()
    assert iter(r) is not None
    for i in r:
        assert issubclass(i, Operator)


def test_get_available_operators_returns_something():
    assert len(get_available_operators()) > 0


def test_register_operator_returns_class():
    class FooOperator(Operator):
        pass
    r = register_operator(FooOperator)
    assert r is FooOperator


def test_register_operator_invalid_class():
    class NotAnOperator(object):
        pass
    with pytest.raises(ValueError) as e:
        register_operator(NotAnOperator)
    assert_terms_in_exception(e, ['not', 'operator'])


def test_register_operator_registers():
    class SomeOperator(Operator):
        pass
    register_operator(SomeOperator)
    operators = get_available_operators()
    assert SomeOperator in operators
