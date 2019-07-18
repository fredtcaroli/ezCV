import re
from io import StringIO
from typing import Any
from unittest.mock import patch, ANY, call, MagicMock

import numpy as np
import pytest
import yaml

from ezcv import CompVizPipeline
from ezcv.operator import Operator, IntegerParameter, NumberParameter
from ezcv.pipeline.context import PipelineContext
from ezcv.test_utils import build_img, parametrize_img, assert_terms_in_exception
from ezcv.utils import is_image


class TestOperator(Operator):
    def run(self, img: np.ndarray, ctx: PipelineContext) -> np.ndarray:
        return img + 1

    param1 = IntegerParameter(default_value=10, lower=5, upper=15)
    param2 = NumberParameter(default_value=3.3, lower=1, upper=10)


def get_config_stream():
    config = """
        version: '0.0'

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


def test_pipeline_load_return(config_stream):
    r = CompVizPipeline.load(config_stream)
    assert isinstance(r, CompVizPipeline)


def test_pipeline_load_loaded_values():
    config = """
            version: '0.0'

            pipeline:
              - name: op1
                config:
                  implementation: {operator}
                  params:
                    param1: 3
              - name: op2
                config:
                  implementation: {operator}
                  params:
                    param2: 1
            """.format(operator=__name__ + '.TestOperator')
    stream = StringIO(config)
    pipeline = CompVizPipeline.load(stream)
    assert pipeline.operators['op1'].param1 == 3
    assert pipeline.operators['op1'].param2 == 3.3
    assert pipeline.operators['op2'].param1 == 10
    assert pipeline.operators['op2'].param2 == 1


def test_pipeline_load_add_operator_calls(config_stream):
    with patch('ezcv.pipeline.core.CompVizPipeline.add_operator') as mock:
        _ = CompVizPipeline.load(config_stream)
        assert mock.mock_calls == [call('op1', ANY), call('op2', ANY)]


def test_pipeline_load_create_operator_calls(config_stream):
    with patch('ezcv.operator.create_operator') as mock:
        _ = CompVizPipeline.load(config_stream)
        assert mock.call_count == 2


def test_pipeline_has_operators():
    pipeline = CompVizPipeline()
    assert hasattr(pipeline, 'operators')


def test_pipeline_operators_starts_empty():
    pipeline = CompVizPipeline()
    assert len(pipeline.operators) == 0


def test_pipeline_operators_indexable(config_stream):
    pipeline = CompVizPipeline.load(config_stream)
    _ = pipeline.operators['op1']
    _ = pipeline.operators['op2']


def test_pipeline_operators_type(config_stream):
    pipeline = CompVizPipeline.load(config_stream)
    op1 = pipeline.operators['op1']
    assert isinstance(op1, TestOperator)
    op2 = pipeline.operators['op2']
    assert isinstance(op2, TestOperator)


def test_pipeline_operators_params(config_stream):
    pipeline = CompVizPipeline.load(config_stream)
    op1 = pipeline.operators['op1']
    assert op1.param1 == 3 and op1.param2 == 1.5
    op2 = pipeline.operators['op2']
    assert op2.param1 == 5 and op2.param2 == 1.0


def test_pipeline_operators_invalid_name(config_stream):
    pipeline = CompVizPipeline.load(config_stream)
    with pytest.raises(Exception):
        _ = pipeline.operators['invalid']


def test_pipeline_add_operator_operators_count():
    pipeline = CompVizPipeline()
    pipeline.add_operator('test_op', TestOperator())
    assert len(pipeline.operators) == 1


def test_pipeline_add_operator_operators_name():
    pipeline = CompVizPipeline()
    pipeline.add_operator('test_op', TestOperator())
    assert 'test_op' in pipeline.operators


def test_pipeline_add_operator_duplicated_name():
    pipeline = CompVizPipeline()
    pipeline.add_operator('test_op', TestOperator())
    with pytest.raises(ValueError) as e:
        pipeline.add_operator('test_op', TestOperator())

    assert_terms_in_exception(e, ['duplicated'])


@parametrize_img
@pytest.mark.parametrize('pipeline', [CompVizPipeline(), CompVizPipeline.load(get_config_stream())])
def test_pipeline_run_return(img, pipeline):
    r = pipeline.run(img)
    assert isinstance(r, tuple)
    assert len(r) == 2
    out, ctx = r
    assert is_image(out)
    assert isinstance(ctx, PipelineContext)


@parametrize_img(include_valid=False, include_invalid=True)
def test_pipeline_run_invalid_img(img):
    pipeline = CompVizPipeline()
    with pytest.raises(ValueError) as e:
        pipeline.run(img)

    assert_terms_in_exception(e, ['invalid', 'image'])


@parametrize_img(kind='black')
def test_pipeline_run_result(img, config_stream):
    pipeline = CompVizPipeline.load(config_stream)
    out, ctx = pipeline.run(img)
    assert np.all(out == 2)


@parametrize_img
def test_pipeline_run_all_ops(img, config_stream):
    with patch(__name__ + '.TestOperator.run') as mock:
        mock.side_effect = lambda img, ctx: img
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
def test_pipeline_run_check_op_return(return_value):
    class TestWrongReturnOperator(Operator):
        def run(self, img: np.ndarray, ctx: PipelineContext) -> Any:
            return return_value

    pipeline = CompVizPipeline()
    pipeline.add_operator('test_op', TestWrongReturnOperator())

    with pytest.raises(ValueError) as e:
        pipeline.run(build_img((16, 16)))

    assert_terms_in_exception(e, ['return', 'invalid'])


def test_pipeline_run_set_ctx_original_img():
    img_rgb = build_img((128, 128), rgb=True)
    img_gray = build_img((128, 128), rgb=False)

    class TestCtxOriginalImgOperator(Operator):
        def run(self, img: np.ndarray, ctx: PipelineContext) -> np.ndarray:
            assert np.all(ctx.original_img == img)
            return img

    pipeline = CompVizPipeline()
    pipeline.add_operator('test_op', TestCtxOriginalImgOperator())
    pipeline.run(img_rgb)
    pipeline.run(img_gray)


def test_pipeline_run_cant_alter_original_img():
    img = build_img((128, 128), kind='black')

    class TestCtxOriginalImg(Operator):
        def run(self, img: np.ndarray, ctx: PipelineContext) -> np.ndarray:
            original_img = ctx.original_img
            original_img[10, ...] = 255
            return img

    pipeline = CompVizPipeline()
    pipeline.add_operator('test_op', TestCtxOriginalImg())

    with pytest.raises(ValueError) as e:
        pipeline.run(img)

    assert_terms_in_exception(e, ['read-only'])


def test_pipeline_run_add_info_works():
    info_name = 'info_name'
    info_value = object()

    class TestCtxAddInfo(Operator):
        def run(self, img: np.ndarray, ctx: PipelineContext) -> np.ndarray:
            ctx.add_info(info_name, info_value)
            return img

    pipeline = CompVizPipeline()
    op_name1 = 'test_op1'
    pipeline.add_operator(op_name1, TestCtxAddInfo())
    op_name2 = 'test_op2'
    pipeline.add_operator(op_name2, TestCtxAddInfo())

    out, ctx = pipeline.run(build_img((16, 16)))

    assert ctx.info == {
        op_name1: {
            info_name: info_value
        },
        op_name2: {
            info_name: info_value
        }
    }


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
    output_stream = StringIO()
    output_stream.write = MagicMock()
    pipeline.save(output_stream)
    output_stream.write.assert_called()


@pytest.fixture
def simple_config_string():
    pipeline = CompVizPipeline()
    output_stream = StringIO()
    pipeline.save(output_stream)
    output_stream.seek(0)
    config_str = output_stream.read()
    return config_str


def test_pipeline_save_is_yaml(simple_config_string):
    yaml.load(StringIO(simple_config_string))


def test_pipeline_save_version_exists(simple_config):
    assert 'version' in simple_config


@pytest.fixture
def simple_config(simple_config_string):
    return yaml.load(StringIO(simple_config_string))


def test_pipeline_save_version_is_version(simple_config):
    assert re.match(r'\d+(\.\d+)*', simple_config['version'])


def test_pipeline_save_version_is_right(simple_config):
    assert simple_config['version'] == '0.0'


def test_pipeline_save_pipeline_exists(simple_config):
    assert 'pipeline' in simple_config


def test_pipeline_save_pipeline_type(simple_config):
    assert isinstance(simple_config['pipeline'], list)


def test_pipeline_save_pipeline_empty(simple_config):
    assert simple_config['pipeline'] == []


@pytest.fixture
def more_complex_config(config_stream):
    pipeline = CompVizPipeline.load(config_stream)
    output_stream = StringIO()
    pipeline.save(output_stream)
    output_stream.seek(0)
    return yaml.load(output_stream.read())


def test_pipeline_save_pipeline_nb_of_stages(more_complex_config):
    assert len(more_complex_config['pipeline']) == 2


def test_pipeline_save_pipeline_stages_type(more_complex_config):
    assert all(isinstance(stage, dict) for stage in more_complex_config['pipeline'])


def test_pipeline_save_pipeline_names(more_complex_config):
    names = [stage_config['name'] for stage_config in more_complex_config['pipeline']]
    assert names == ['op1', 'op2']


def test_pipeline_save_gets_ops_configs(config_stream):
    pipeline = CompVizPipeline.load(config_stream)
    with patch('ezcv.operator.get_operator_config') as mock:
        mock.return_value = 'some_value'
        pipeline.save(StringIO())
        call_args = [cal[0] for cal in mock.call_args_list]
        assert (pipeline.operators['op1'],) in call_args
        assert (pipeline.operators['op2'],) in call_args


def test_pipeline_save_op_config_exists(more_complex_config):
    assert all('config' in stage_config for stage_config in more_complex_config['pipeline'])


def test_pipeline_save_op_config_return_from_get_operator_config(config_stream):
    pipeline = CompVizPipeline.load(config_stream)
    unique_value = 'unique_value'
    with patch('ezcv.operator.get_operator_config') as mock:
        mock.return_value = unique_value
        output_stream = StringIO()
        pipeline.save(output_stream)
        output_stream.seek(0)
        config = yaml.load(output_stream)
        assert config['pipeline'][0]['config'] == 'unique_value'
        assert config['pipeline'][1]['config'] == 'unique_value'

