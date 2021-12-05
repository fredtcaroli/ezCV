import re
from io import StringIO
from typing import Any
from unittest.mock import patch, Mock

import numpy as np
import pytest
import yaml

from ezcv import CompVizPipeline
from ezcv.operator import Operator, IntegerParameter, DoubleParameter
from ezcv.pipeline.context import PipelineContext
from ezcv.pipeline.core import OperatorFailedError, BadImageError
from ezcv.test_utils import build_img, parametrize_img, assert_terms_in_exception
from ezcv.typing import Image
from ezcv.utils import is_image


PARAM1_DEFAULT_VALUE = 10
PARAM1_LOWER = 5
PARAM1_UPPER = 15

PARAM2_DEFAULT_VALUE = 3.3
PARAM2_LOWER = 1
PARAM2_UPPER = 11


class TestOperator(Operator):
    def run(self, img: Image, ctx: PipelineContext) -> Image:
        return img + 1

    param1 = IntegerParameter(default_value=PARAM1_DEFAULT_VALUE, lower=PARAM1_LOWER, upper=PARAM1_UPPER)
    param2 = DoubleParameter(default_value=PARAM2_DEFAULT_VALUE, lower=PARAM2_LOWER, upper=PARAM2_UPPER)


OP1_PARAM1 = 3
OP1_PARAM2 = 1.5

OP2_PARAM1 = 5
OP2_PARAM2 = 1


def get_config_stream():
    config = f"""
        version: '0.0'

        pipeline:
          - name: op1
            config:
              implementation: {__name__}.TestOperator
              params:
                param1: {OP1_PARAM1}
                param2: {OP1_PARAM2}
          - name: op2
            config:
              implementation: {__name__}.TestOperator
              params:
                param1: {OP2_PARAM1}
                param2: {OP2_PARAM2}
        """
    stream = StringIO(config)
    return stream


@pytest.fixture
def config_stream():
    return get_config_stream()


class TestLoad:
    def test_return_type(self, config_stream):
        r = CompVizPipeline.load(config_stream)
        assert isinstance(r, CompVizPipeline)

    def test_create_pipeline_call(self):
        unique_object = object()
        with patch('ezcv.config.create_pipeline') as mock:
            mock.return_value = unique_object
            return_value = CompVizPipeline.load(get_config_stream())
            mock.assert_called_with(yaml.safe_load(get_config_stream()))
            assert return_value is unique_object


class TestOperatorsProperty:
    def test_pipeline_has_operators(self):
        pipeline = CompVizPipeline()
        assert hasattr(pipeline, 'operators')

    def test_pipeline_operators_starts_empty(self):
        pipeline = CompVizPipeline()
        assert len(pipeline.operators) == 0

    def test_pipeline_operators_indexable(self, config_stream):
        pipeline = CompVizPipeline.load(config_stream)
        _ = pipeline.operators['op1']
        _ = pipeline.operators['op2']

    def test_pipeline_operators_invalid_name(self, config_stream):
        pipeline = CompVizPipeline.load(config_stream)
        with pytest.raises(KeyError):
            _ = pipeline.operators['invalid']


class TestAddOperator:
    def test_operators_count(self):
        pipeline = CompVizPipeline()
        pipeline.add_operator('test_op', TestOperator())
        assert len(pipeline.operators) == 1

    def test_operators_name(self):
        pipeline = CompVizPipeline()
        pipeline.add_operator('test_op', TestOperator())
        assert 'test_op' in pipeline.operators

    def test_duplicated_name(self):
        pipeline = CompVizPipeline()
        pipeline.add_operator('test_op', TestOperator())
        with pytest.raises(ValueError) as e:
            pipeline.add_operator('test_op', TestOperator())

        assert_terms_in_exception(e, ['duplicated'])


class TestRemoveOperator:
    @pytest.mark.parametrize('idx', ['op_to_delete', 1])
    def test_happy_path(self, idx):
        operator_to_delete = 'op_to_delete'
        operator_to_stay = 'op_to_stay'
        pipeline = CompVizPipeline()
        pipeline.add_operator(operator_to_stay, TestOperator())
        pipeline.add_operator(operator_to_delete, TestOperator())
        pipeline.remove_operator(idx)
        assert list(pipeline.operators.keys()) == [operator_to_stay]

    @pytest.mark.parametrize('idx', ['nonexistent_name', -1, 10])
    def test_invalid_idx(self, idx):
        pipeline = CompVizPipeline()
        pipeline.add_operator('test_op', TestOperator())
        with pytest.raises(ValueError) as e:
            pipeline.remove_operator(idx)
        assert_terms_in_exception(e, ['invalid', 'operator'])


class TestRun:
    @parametrize_img
    @pytest.mark.parametrize('pipeline', [CompVizPipeline(), CompVizPipeline.load(get_config_stream())])
    def test_return_type(self, img, pipeline):
        r = pipeline.run(img)
        assert isinstance(r, tuple)
        assert len(r) == 2
        out, ctx = r
        assert is_image(out)
        assert isinstance(ctx, PipelineContext)

    @parametrize_img(include_valid=False, include_invalid=True)
    def test_invalid_img(self, img):
        pipeline = CompVizPipeline()
        with pytest.raises(BadImageError) as e:
            pipeline.run(img)

        assert_terms_in_exception(e, ['invalid', 'image'])

    @parametrize_img(kind='black')
    def test_result(self, img, config_stream):
        pipeline = CompVizPipeline.load(config_stream)
        out, ctx = pipeline.run(img)
        assert np.all(out == 2)

    @parametrize_img
    def test_run_all_ops(self, img, config_stream):
        with patch(__name__ + '.TestOperator.run') as mock:
            mock.side_effect = lambda i, _: i
            pipeline = CompVizPipeline.load(config_stream)
            _ = pipeline.run(img)
            assert mock.call_count == 2

    @pytest.mark.parametrize('return_value', [
        np.random.randint(0, 256, size=(16, 16), dtype='int64'),
        np.random.randint(0, 256, size=(0, 16), dtype='uint8'),
        np.random.randint(0, 256, size=(16, 0), dtype='uint8'),
        np.random.randint(0, 256, size=(0, 0), dtype='uint8'),
        np.random.randint(0, 256, size=(16, 16, 2), dtype='uint8'),
        10,
        None,
        object(),
        np.random.randint(0, 256, size=(10,), dtype='uint8')
    ])
    def test_check_op_return(self, return_value):
        class TestWrongReturnOperator(Operator):
            def run(self, img: np.ndarray, ctx: PipelineContext) -> Any:
                return return_value

        pipeline = CompVizPipeline()
        pipeline.add_operator('test_op', TestWrongReturnOperator())

        with pytest.raises(BadImageError) as e:
            pipeline.run(build_img((16, 16)))

        assert_terms_in_exception(e, ['return', 'invalid'])

    def test_only_gray_flag(self):
        class TestOnlyGrayFlagOperator(Operator):
            only_gray = True

            def run(self, img: Image, ctx: PipelineContext) -> Image:
                return img

        pipeline = CompVizPipeline()
        pipeline.add_operator('test_op', TestOnlyGrayFlagOperator())

        with pytest.raises(OperatorFailedError) as e:
            pipeline.run(build_img((16, 16), rgb=True))

        assert_terms_in_exception(e, ["expect", "gray"])


def test_pipeline_run_set_ctx_original_img():
    img_rgb = build_img((128, 128), rgb=True)
    img_gray = build_img((128, 128), rgb=False)

    class TestCtxOriginalImgOperator(Operator):
        def run(self, img: Image, ctx: PipelineContext) -> Image:
            assert np.all(ctx.original_img == img)
            return img

    pipeline = CompVizPipeline()
    pipeline.add_operator('test_op', TestCtxOriginalImgOperator())
    pipeline.run(img_rgb)
    pipeline.run(img_gray)


def test_pipeline_run_cant_alter_original_img():
    class TestCtxOriginalImg(Operator):
        def run(self, img: Image, ctx: PipelineContext) -> Image:
            original_img = ctx.original_img
            original_img[10, ...] = 255
            return img

    pipeline = CompVizPipeline()
    pipeline.add_operator('test_op', TestCtxOriginalImg())

    test_img = build_img((128, 128), kind='black')
    with pytest.raises(OperatorFailedError) as e:
        pipeline.run(test_img)

    assert_terms_in_exception(e, ['read-only'])


def test_pipeline_run_add_info_works():
    info_name = 'info_name'
    info_value = object()

    class TestCtxAddInfo(Operator):
        def run(self, img: Image, ctx: PipelineContext) -> Image:
            ctx.add_info(info_name, info_value)
            return img

    pipeline = CompVizPipeline()
    op_name1 = 'test_op1'
    pipeline.add_operator(op_name1, TestCtxAddInfo())
    op_name2 = 'test_op2'
    pipeline.add_operator(op_name2, TestCtxAddInfo())

    _, returned_ctx = pipeline.run(build_img((16, 16)))

    expected_info = {
        op_name1: {
            info_name: info_value
        },
        op_name2: {
            info_name: info_value
        }
    }
    actual_info = returned_ctx.info

    assert actual_info == expected_info


def test_pipeline_run_empty_info(config_stream):
    pipeline = CompVizPipeline.load(config_stream)
    img = build_img((16, 16))
    out, ctx = pipeline.run(img)
    assert ctx.info == {
        'op1': {},
        'op2': {}
    }


@parametrize_img(kind='black')
def test_pipeline_integration(img):
    # this test doesn't work if we can't blur something
    if img.shape[:2] == (1, 1):
        return

    config = """
        version: '0.0'
        
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
    pipeline = CompVizPipeline.load(stream)

    mid_index = img.shape[0] // 2
    img[mid_index, ...] = 255
    out, ctx = pipeline.run(img)
    assert np.all(out[mid_index, ...] < 255)


def test_pipeline_save_runs():
    pipeline = CompVizPipeline()
    pipeline.save(StringIO())


def test_pipeline_save_writes():
    pipeline = CompVizPipeline()
    output_stream = Mock()
    output_stream.write = Mock()
    pipeline.save(output_stream)
    output_stream.write.assert_called()


@pytest.fixture
def order_asserter_operator():
    counter = 0

    class OrderAsserter(Operator):
        def __init__(self, expected_count: int):
            super().__init__()
            self.expected_count = expected_count

        def run(self, img: Image, ctx: PipelineContext) -> Image:
            nonlocal counter
            assert counter == self.expected_count
            counter += 1
            return img

    return OrderAsserter


def test_pipeline_execution_order(order_asserter_operator):
    pipeline = CompVizPipeline()
    pipeline.add_operator('op1', order_asserter_operator(0))
    pipeline.add_operator('op2', order_asserter_operator(1))
    pipeline.add_operator('op3', order_asserter_operator(2))

    pipeline.run(build_img((16, 16)))


def test_operators_property_order():
    pipeline = CompVizPipeline()
    pipeline.add_operator('op1', TestOperator())
    pipeline.add_operator('op2', TestOperator())
    pipeline.add_operator('op3', TestOperator())

    assert list(pipeline.operators.keys()) == ['op1', 'op2', 'op3']


class TestRenameOperator:
    def test_rename_operator(self):
        pipeline = CompVizPipeline()
        starting_name = 'some_name'
        new_name = 'another_name'
        op = TestOperator()
        pipeline.add_operator(starting_name, op)
        pipeline.add_operator('foo', TestOperator())
        pipeline.add_operator('bar', TestOperator())

        expected_len = len(pipeline.operators)

        pipeline.rename_operator(starting_name, new_name)

        assert len(pipeline.operators) == expected_len
        assert new_name in pipeline.operators
        assert pipeline.operators[new_name] is op

    def test_rename_keeps_operators_order(self, order_asserter_operator):
        pipeline = CompVizPipeline()
        pipeline.add_operator('op1', order_asserter_operator(0))
        pipeline.add_operator('op2', order_asserter_operator(1))
        pipeline.add_operator('op3', order_asserter_operator(2))

        pipeline.rename_operator('op2', 'new_name')

        pipeline.run(build_img((16, 16)))

    def test_rename_nonexistent_name(self):
        pipeline = CompVizPipeline()

        with pytest.raises(ValueError) as e:
            pipeline.rename_operator('nonexistent', 'foo')

        assert_terms_in_exception(e, ['operator', 'name'])

    def test_rename_to_existent_name(self):
        pipeline = CompVizPipeline()

        pipeline.add_operator('op1', TestOperator())
        pipeline.add_operator('op2', TestOperator())

        with pytest.raises(ValueError) as e:
            pipeline.rename_operator('op1', 'op2')

        assert_terms_in_exception(e, ['name'])

    def test_rename_keeps_order(self):
        pipeline = CompVizPipeline()

        pipeline.add_operator('op1', TestOperator())
        pipeline.add_operator('op2', TestOperator())
        pipeline.add_operator('op3', TestOperator())

        pipeline.rename_operator('op2', 'renamed')

        assert list(pipeline.operators.keys()) == ['op1', 'renamed', 'op3']

    def test_rename_keeps_saving_order(self):
        pipeline = CompVizPipeline()

        pipeline.add_operator('op1', TestOperator())
        pipeline.add_operator('op2', TestOperator())
        pipeline.add_operator('op3', TestOperator())

        pipeline.rename_operator('op2', 'renamed')

        stream = StringIO()
        pipeline.save(stream)
        stream.seek(0)
        config = yaml.safe_load(stream)

        operators_order = [op_config['name'] for op_config in config['pipeline']]
        assert operators_order == ['op1', 'renamed', 'op3']


def test_operator_fail():
    class FailingOperator(Operator):
        def run(self, img: Image, ctx: PipelineContext) -> Image:
            raise ValueError('Failed')

    pipeline = CompVizPipeline()
    pipeline.add_operator('op', FailingOperator())

    with pytest.raises(OperatorFailedError):
        pipeline.run(build_img((16, 16)))
