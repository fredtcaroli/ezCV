import pytest

from ezcv.operator import Operator, IntegerParameter
from ezcv.pipeline import PipelineContext
from ezcv.typing import Image


def test_operator_not_implemented():
    operator = Operator()
    with pytest.raises(NotImplementedError):
        operator.run(None, None)


class OperatorForTesting(Operator):
    param2 = IntegerParameter(default_value=0, lower=0, upper=1)
    param1 = IntegerParameter(default_value=0, lower=0, upper=1)

    def run(self, img: Image, ctx: PipelineContext) -> Image:
        raise NotImplementedError()


class TestGetParametersSpecs:
    @pytest.mark.parametrize('obj', (OperatorForTesting, OperatorForTesting()))
    def test_happy_path(self, obj):
        params = obj.get_parameters_specs()
        assert len(params) == 2
        assert 'param1' in params and params['param1'] is OperatorForTesting.param1
        assert 'param2' in params and params['param2'] is OperatorForTesting.param2

    def test_no_params(self):
        class NoParamsOperator(Operator):
            def run(self, img: Image, ctx: PipelineContext) -> Image:
                pass

        params = NoParamsOperator.get_parameters_specs()
        assert len(params) == 0

    def test_parameters_order(self):
        params = OperatorForTesting.get_parameters_specs()
        assert list(params.keys()) == ['param2', 'param1']
