from typing import Any, Tuple

import numpy as np
import pytest
from pytest import fixture

from ezcv.operator.core.config import create_operator, get_operator_config
from ezcv.operator.core.operator import Operator
from ezcv.operator.core.parameter import IntegerParameter, DoubleParameter, Parameter
from ezcv.pipeline import PipelineContext
from ezcv.test_utils import assert_terms_in_exception

unique_object = object()


class TestOperator(Operator):
    param1 = IntegerParameter(default_value=10, lower=0, upper=15)
    param2 = DoubleParameter(default_value=5.3, lower=0, upper=15)
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


def test_get_operator_config_return_type():
    config = get_operator_config(TestOperator())
    assert isinstance(config, dict)


def test_get_operator_config_implementation_exists():
    op = TestOperator()
    config = get_operator_config(op)
    assert 'implementation' in config


def test_get_operator_config_implementation_value():
    op = TestOperator()
    config = get_operator_config(op)
    assert config['implementation'] == __name__ + '.TestOperator'


def test_get_operator_config_params_exists():
    op = TestOperator()
    config = get_operator_config(op)
    assert 'params' in config


def test_get_operator_config_params_type():
    op = TestOperator()
    config = get_operator_config(op)
    assert isinstance(config['params'], dict)


def test_get_operator_config_params_keys():
    op = TestOperator()
    config = get_operator_config(op)
    assert set(config['params'].keys()) == {'param1', 'param2'}


def test_get_operator_config_params_default_values():
    """ Tests get_operator_config params when the operator has default values
    """
    op = TestOperator()
    config = get_operator_config(op)
    assert config['params'] == {'param1': 10, 'param2': 5.3}


def test_get_operator_config_params_values():
    op = TestOperator()
    op.param1 = 5
    op.param2 = 3.5
    config = get_operator_config(op)
    assert config['params'] == {'param1': 5, 'param2': 3.5}


class ComplexTestParameter(Parameter[Tuple[int, int]]):
    def from_config(self, config: Any) -> Tuple[int, int]:
        return config['val1'], config['val2']

    def to_config(self, value: Tuple[int, int]) -> Any:
        return {'val1': value[0], 'val2': value[1]}


class ComplexTestOperator(Operator):
    some_param = ComplexTestParameter((1, 2))


def test_complex_parameter_config():
    op = ComplexTestOperator()
    config = get_operator_config(op)
    assert config['params'] == {'some_param': {'val1': 1, 'val2': 2}}


def test_complex_parameter_parsing():
    op = ComplexTestOperator()
    config = get_operator_config(op)
    config['params']['some_param']['val1'] = 10
    config['params']['some_param']['val2'] = 20

    parsed_op = create_operator(config)
    assert parsed_op.some_param == (10, 20)
