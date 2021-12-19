from typing import Any, Tuple

import pytest

from ezcv.config import create_operator, get_operator_config, get_parameters_specs, ConfigParsingError, create_pipeline, \
    Config, get_pipeline_config
from ezcv.operator.core.operator import Operator
from ezcv.operator.core.parameter import IntegerParameter, DoubleParameter, ParameterSpec
from ezcv.pipeline import PipelineContext
from ezcv.test_utils import assert_terms_in_exception
from ezcv.typing import Image

unique_object = object()


param1_default_value = 10
param2_default_value = 5.3


class OperatorForTesting(Operator):
    param1 = IntegerParameter(default_value=param1_default_value, lower=0, upper=15)
    param2 = DoubleParameter(default_value=param2_default_value, lower=0, upper=15)
    non_param = unique_object

    def run(self, img: Image, ctx: PipelineContext) -> Image:
        raise NotImplementedError()


class NotAnOperator(object):
    pass


param1_value = 5
param2_value = 2.5


@pytest.fixture
def operator_config() -> Config:
    operator_config = {
        'implementation': __name__ + '.OperatorForTesting',
        'params': {
            'param1': param1_value,
            'param2': param2_value,
        }
    }
    return operator_config


class TestCreateOperator:
    def test_return_type(self, operator_config):
        operator = create_operator(operator_config)
        assert isinstance(operator, OperatorForTesting)

    def test_params(self, operator_config):
        operator = create_operator(operator_config)
        assert operator.param1 == param1_value
        assert operator.param2 == param2_value

    def test_ignore_non_parameters(self, operator_config):
        operator = create_operator(operator_config)
        assert operator.non_param is unique_object

    def test_fail_invalid_param_name(self, operator_config):
        operator_config['params']['invalid_param'] = 10

        with pytest.raises(ConfigParsingError) as e:
            create_operator(operator_config)

        assert_terms_in_exception(e, ['invalid'])

    def test_fail_non_param(self, operator_config):
        operator_config['params']['non_param'] = 10

        with pytest.raises(ConfigParsingError) as e:
            create_operator(operator_config)

        assert_terms_in_exception(e, ['invalid'])

    def test_missing_params(self, operator_config):
        del operator_config['params']['param2']

        operator = create_operator(operator_config)
        assert operator.param2 == param2_default_value

    def test_invalid_implementation(self, operator_config):
        operator_config['implementation'] = 'not.an.operator'

        with pytest.raises(ConfigParsingError) as e:
            create_operator(operator_config)

        assert_terms_in_exception(e, ['invalid', 'implementation'])

    def test_invalid_implementation_no_dots(self, operator_config):
        operator_config['implementation'] = 'thisisnotanoperator'

        with pytest.raises(ConfigParsingError) as e:
            create_operator(operator_config)

        assert_terms_in_exception(e, ['invalid', 'implementation'])

    def test_not_an_operator(self, operator_config):
        operator_config['implementation'] = __name__ + '.NotAnOperator'

        with pytest.raises(ConfigParsingError) as e:
            create_operator(operator_config)

        assert_terms_in_exception(e, ['not', 'operator'])

    def test_invalid_parameter_value(self, operator_config):

        class FailingParameter(ParameterSpec):

            def to_config(self, value: Any) -> Any:
                raise NotImplementedError()

            def from_config(self, operator_config: Any) -> None:
                assert False

        bak = OperatorForTesting.param1
        OperatorForTesting.param1 = FailingParameter(default_value=None)

        with pytest.raises(ConfigParsingError) as e:
            create_operator(operator_config)

        OperatorForTesting.param1 = bak

        assert_terms_in_exception(e, ['fail', 'pars'])

    def test_fail_extra_config_attributes(self, operator_config):
        operator_config['unknown_attribute'] = 42
        with pytest.raises(ConfigParsingError):
            create_operator(operator_config)

    def test_fail_missing_implementation(self, operator_config):
        del operator_config['implementation']
        with pytest.raises(ConfigParsingError):
            create_operator(operator_config)


class TestGetOperatorConfig:
    def test_return_type(self):
        operator_config = get_operator_config(OperatorForTesting())
        assert isinstance(operator_config, dict)

    def test_return_value(self, operator_config):
        operator = create_operator(operator_config)
        actual_operator_config = get_operator_config(operator)
        assert actual_operator_config == operator_config


class NestedConfigParameter(ParameterSpec[Tuple[int, int]]):
    def from_config(self, operator_config: Any) -> Tuple[int, int]:
        return operator_config['val1'], operator_config['val2']

    def to_config(self, value: Tuple[int, int]) -> Any:
        return {'val1': value[0], 'val2': value[1]}


class NestedConfigParamTestOperator(Operator):
    some_param = NestedConfigParameter((1, 2))

    def run(self, img: Image, ctx: PipelineContext) -> Image:
        pass


def test_nested_param_config():
    param_value = (3, 4)
    operator = NestedConfigParamTestOperator()
    operator.some_param = param_value
    operator_config = get_operator_config(operator)
    actual_operator = create_operator(operator_config)
    assert actual_operator.some_param == (3, 4)


class TestGetParametersSpecs:
    def test_happy_path(self):
        params = get_parameters_specs(OperatorForTesting)
        assert len(params) == 2
        assert 'param1' in params and params['param1'] is OperatorForTesting.param1
        assert 'param2' in params and params['param2'] is OperatorForTesting.param2

    def test_no_params(self):
        class NoParamsOperator(Operator):
            def run(self, img: Image, ctx: PipelineContext) -> Image:
                pass

        params = get_parameters_specs(NoParamsOperator)
        assert len(params) == 0

    def test_parameters_order(self):
        class OperatorForTestingParamsOrder(Operator):
            foo = IntegerParameter(default_value=0, lower=0, upper=1)
            bar = IntegerParameter(default_value=0, lower=0, upper=1)
            baz = IntegerParameter(default_value=0, lower=0, upper=1)
            something = IntegerParameter(default_value=0, lower=0, upper=1)

            def run(self, img: Image, ctx: PipelineContext) -> Image:
                raise NotImplementedError()

        params = get_parameters_specs(OperatorForTestingParamsOrder)
        assert list(params.keys()) == ['foo', 'bar', 'baz', 'something']


@pytest.fixture
def pipeline_config(operator_config: Config) -> Config:
    config = {
        'version': '0.0',
        'pipeline': [
            {
                'name': 'op1',
                'config': operator_config
            },
            {
                'name': 'op2',
                'config': operator_config
            }
        ]
    }
    return config


class TestCreatePipeline:
    def test_pipeline_operators_type(self, pipeline_config):
        pipeline = create_pipeline(pipeline_config)
        op1 = pipeline.operators['op1']
        assert isinstance(op1, OperatorForTesting)
        op2 = pipeline.operators['op2']
        assert isinstance(op2, OperatorForTesting)

    def test_pipeline_operators_params(self, pipeline_config):
        pipeline = create_pipeline(pipeline_config)
        op1 = pipeline.operators['op1']
        op2 = pipeline.operators['op2']
        assert op1.param1 == op2.param1 == param1_value and op1.param2 == op2.param2 == param2_value

    def test_unknown_pipeline_attribute(self, pipeline_config):
        pipeline_config['abc'] = 42
        with pytest.raises(ConfigParsingError) as e:
            _ = create_operator(pipeline_config)
        assert_terms_in_exception(e, ['unknown'])


def test_get_pipeline_config(pipeline_config):
    pipeline = create_pipeline(pipeline_config)
    actual_pipeline_config = get_pipeline_config(pipeline)
    assert actual_pipeline_config == pipeline_config
