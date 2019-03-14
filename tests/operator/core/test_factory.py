from typing import Any

import numpy as np
import pytest
from pytest import fixture

from ezcv.operator.core.factory import create_operator
from ezcv.operator.core.operator import Operator
from ezcv.operator.core.parameter import IntegerParameter, NumberParameter, Parameter
from ezcv.pipeline import PipelineContext
from tests.utils import assert_terms_in_exception

unique_object = object()


class TestOperator(Operator):
    param1 = IntegerParameter(default_value=10)
    param2 = NumberParameter(default_value=5.3)
    non_param = unique_object

    def run(self, img: np.ndarray, ctx: PipelineContext) -> np.ndarray:
        raise NotImplementedError()


@fixture
def config():
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

    assert_terms_in_exception(e, ['invalid'])


def test_create_operator_fail_invalid_param_type(config):
    config['params']['non_param'] = 10

    with pytest.raises(ValueError) as e:
        create_operator(config)

    assert_terms_in_exception(e, ['invalid'])


def test_create_operator_missing_params_default(config):
    del config['params']['param2']

    operator = create_operator(config)
    assert operator.param2 == 5.3


def test_create_operator_invalid_implementation(config):
    config['implementation'] = 'not.an.operator'

    with pytest.raises(ValueError) as e:
        create_operator(config)

    assert_terms_in_exception(e, ['invalid', 'implementation'])


def test_create_operator_invalid_implementation_no_dots(config):
    config['implementation'] = 'thisisnotanoperator'

    with pytest.raises(ValueError) as e:
        create_operator(config)

    assert_terms_in_exception(e, ['invalid', 'implementation'])


class NotAnOperator(object):
    pass


def test_create_operator_not_an_operator(config):
    config['implementation'] = __name__ + '.NotAnOperator'

    with pytest.raises(ValueError) as e:
        create_operator(config)

    assert_terms_in_exception(e, ['not', 'operator'])


def test_create_operator_invalid_parameter_value(config):

    class FailingParameter(Parameter):

        def to_config(self, value: Any) -> Any:
            raise NotImplementedError()

        def from_config(self, config: Any) -> None:
            assert False

    bak = TestOperator.param1
    TestOperator.param1 = FailingParameter(default_value=None)

    with pytest.raises(ValueError) as e:
        create_operator(config)

    TestOperator.param1 = bak

    assert_terms_in_exception(e, ['fail', 'pars'])
