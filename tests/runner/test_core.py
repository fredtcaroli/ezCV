from io import StringIO
from unittest.mock import patch, ANY, call

import numpy as np
import pytest

from ezcv import Runner
from ezcv.operator import Operator, IntegerParameter, NumberParameter
from ezcv.runner.context import PipelineContext
from tests.utils import parametrize_img, build_img
from ezcv.utils import is_image


class TestOperator(Operator):
    def run(self, img: np.ndarray, ctx: PipelineContext) -> np.ndarray:
        return img + 1

    param1 = IntegerParameter()
    param2 = NumberParameter()


def get_config_stream():
    config = """
        version: 0.0

        pipeline:
          - name: op1
            config:
              implementation: {operator}
              params:
                param1: 3
                param2: 1.5
          - name: op2
            config:
              implementation: {operator}
              params:
                param1: 5
                param2: 1
        """.format(operator=__name__ + '.TestOperator')
    stream = StringIO(config)
    return stream


@pytest.fixture
def config_stream():
    return get_config_stream()


def test_runner_load_return(config_stream):
    r = Runner.load(config_stream)
    assert isinstance(r, Runner)


def test_runner_load_add_operator_calls(config_stream):
    with patch('ezcv.runner.core.Runner.add_operator') as mock:
        _ = Runner.load(config_stream)
        assert mock.mock_calls == [call('op1', ANY), call('op2', ANY)]


def test_runner_load_create_operator_calls(config_stream):
    with patch('ezcv.operator.create_operator') as mock:
        _ = Runner.load(config_stream)
        assert mock.call_count == 2


def test_runner_has_operators():
    runner = Runner()
    assert hasattr(runner, 'operators')


def test_runner_operators_starts_empty():
    runner = Runner()
    assert len(runner.operators) == 0


def test_runner_operators_indexable(config_stream):
    runner = Runner.load(config_stream)
    _ = runner.operators['op1']
    _ = runner.operators['op2']


def test_runner_operators_invalid_name(config_stream):
    runner = Runner.load(config_stream)
    with pytest.raises(Exception):
        _ = runner.operators['invalid']


def test_runner_add_operator_operators_count():
    runner = Runner()
    runner.add_operator('test_op', TestOperator())
    assert len(runner.operators) == 1


def test_runner_add_operator_operators_name():
    runner = Runner()
    runner.add_operator('test_op', TestOperator())
    assert 'test_op' in runner.operators


def test_runner_add_operator_duplicated_name():
    runner = Runner()
    runner.add_operator('test_op', TestOperator())
    with pytest.raises(ValueError) as e:
        runner.add_operator('test_op', TestOperator())

    msg = str(e).lower()
    assert 'duplicated' in msg


@parametrize_img
@pytest.mark.parametrize('runner', [Runner(), Runner.load(get_config_stream())])
def test_runner_run_return(img, runner):
    r = runner.run(img)
    assert isinstance(r, tuple)
    assert len(r) == 2
    out, ctx = r
    assert is_image(out)
    assert isinstance(ctx, PipelineContext)


@parametrize_img(include_valid=False, include_invalid=True)
def test_runner_run_invalid_img(img):
    runner = Runner()
    with pytest.raises(ValueError) as e:
        runner.run(img)

    msg = str(e).lower()
    assert 'invalid' in msg and 'image' in msg


@parametrize_img(kind='black')
def test_runner_run_result(img, config_stream):
    runner = Runner.load(config_stream)
    out, ctx = runner.run(img)
    assert np.all(out == 2)


@parametrize_img
def test_runner_run_all_ops(img, config_stream):
    with patch(__name__ + '.TestOperator.run') as mock:
        mock.side_effect = lambda img, ctx: img
        runner = Runner.load(config_stream)
        _ = runner.run(img)
        assert mock.call_count == 2



def test_runner_run_set_ctx_original_img():
    img_rgb = build_img((128, 128), rgb=True)
    img_gray = build_img((128, 128), rgb=False)

    class TestCtxOriginalImgOperator(Operator):
        def run(self, img: np.ndarray, ctx: PipelineContext) -> np.ndarray:
            assert np.all(ctx.original_img == img)
            return img

    runner = Runner()
    runner.add_operator('test_op', TestCtxOriginalImgOperator())
    runner.run(img_rgb)
    runner.run(img_gray)


@parametrize_img(kind='black')
def test_runner_integration(img):
    # this test doesn't work if we can't blur something
    if img.shape[:2] == (1, 1):
        return

    config = """
        version: 0.0
        
        pipeline:
          - name: Blur1
            config:
              implementation: ezcv.operator.implementations.blur.GaussianBlur
              params:
                kernel_size: 3
                sigma: 1.5
          - name: Blur2
            config:
              implementation: ezcv.operator.implementations.blur.GaussianBlur
              params:
                kernel_size: 5
                sigma: 1
    """
    stream = StringIO(config)
    runner = Runner.load(stream)

    mid_index = img.shape[0] // 2
    img[mid_index, ...] = 255
    out, ctx = runner.run(img)
    assert np.all(out[mid_index, ...] < 255)
