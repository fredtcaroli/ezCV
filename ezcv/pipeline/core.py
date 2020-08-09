import collections
from typing import TextIO, Tuple

import yaml

import ezcv.operator as op_lib
from ezcv import utils
from ezcv.pipeline.context import PipelineContext
from ezcv.typing import Image


class CompVizPipeline(object):
    def __init__(self):
        self.operators = dict()
        self._operators_order = list()

    def run(self, img: Image) -> Tuple[Image, PipelineContext]:
        self._raise_if_invalid_img(img)
        last = img
        ctx = PipelineContext(img)
        for name in self._operators_order:
            operator = self.operators[name]
            with ctx.scope(name):
                last = operator.run(last, ctx)
            if not utils.is_image(last):
                raise ValueError('Invalid return value from "%s": "%s"' % (name, str(last)))
        return last, ctx

    def _raise_if_invalid_img(self, img: Image):
        if not utils.is_image(img):
            raise ValueError('Invalid image provided')

    def add_operator(self, name: str, operator: op_lib.Operator):
        self._raise_if_name_is_unavailable(name)
        self.operators[name] = operator
        self._operators_order.append(name)

    def rename_operator(self, name: str, new_name: str):
        if name not in self.operators:
            raise ValueError(f'Unexistent operator name: "{name}"')
        if new_name in self.operators:
            raise ValueError(f'Operator name "{new_name}" already exists')
        self._operators_order[self._operators_order.index(name)] = new_name
        self.operators[new_name] = self.operators.pop(name)

    def _raise_if_name_is_unavailable(self, name: str):
        if name in self.operators:
            raise ValueError('Trying to add a duplicated name: %s' % name)

    @staticmethod
    def load(stream: TextIO) -> "CompVizPipeline":
        pipeline_config = yaml.safe_load(stream)
        runner = CompVizPipeline()
        for op_config in pipeline_config['pipeline']:
            operator = op_lib.create_operator(op_config['config'])
            runner.add_operator(op_config['name'], operator)
        return runner

    def save(self, stream: TextIO):
        config = dict()
        config['version'] = '0.0'
        pipeline_config = list()
        for name, op in self.operators.items():
            stage_config = dict()
            stage_config['name'] = name
            stage_config['config'] = op_lib.get_operator_config(op)
            pipeline_config.append(stage_config)
        config['pipeline'] = pipeline_config
        yaml.safe_dump(config, stream, sort_keys=False)