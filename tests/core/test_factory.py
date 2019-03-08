import pytest
from pytest import fixture

from ezcv.operator.core import create_operator, Operator
from ezcv.operator.core.parameter import IntegerParameter, NumberParameter


class UniqueObjectClass(object):
    pass


unique_object = object()


class TestOperator(Operator):
    param1 = IntegerParameter()
    param2 = NumberParameter()
    non_param = unique_object


@fixture
def config(shared_datadir):
    config = {
        'implementation': __name__ + '.TestOperator',
        'params': {
            'param1': 5,
            'param2': 2.5,
        }
    }
    return config


def test_create_operator_return(config):
    operator = create_operator(config)
    assert isinstance(operator, TestOperator)


def test_create_operator_params(config):
    operator = create_operator(config)
    assert operator.param1 == 5
    assert operator.param2 == 2.5


def test_create_operator_ignore_non_parameters(config):
    operator = create_operator(config)
    assert operator.non_param is unique_object


def test_create_operator_fail_invalid_param_name(config):
    config['params']['invalid_param'] = 10

    with pytest.raises(ValueError) as e:
        create_operator(config)

    assert 'invalid' in str(e).lower()


def test_create_operator_fail_invalid_param_type(config):
    config['params']['non_param'] = 10

    with pytest.raises(ValueError) as e:
        create_operator(config)

    assert 'invalid' in str(e).lower()


def test_create_operator_fail_missing_params(config):
    del config['params']['param2']

    with pytest.raises(ValueError) as e:
        create_operator(config)

    assert 'missing' in str(e).lower()


def test_create_operator_invalid_implementation(config):
    config['implementation'] = 'not.an.operator'

    with pytest.raises(ValueError) as e:
        create_operator(config)

    msg = str(e).lower()
    assert 'invalid' in msg and 'implementation' in msg


def test_create_operator_invalid_implementation_no_dots(config):
    config['implementation'] = 'thisisnotanoperator'

    with pytest.raises(ValueError) as e:
        create_operator(config)

    msg = str(e).lower()
    assert 'invalid' in msg and 'implementation' in msg


class NotAnOperator(object):
    pass


def test_create_operator_not_an_operator(config):
    config['implementation'] = __name__ + '.NotAnOperator'

    with pytest.raises(ValueError) as e:
        create_operator(config)

    msg = str(e).lower()
    assert 'not' in msg and 'operator' in msg
