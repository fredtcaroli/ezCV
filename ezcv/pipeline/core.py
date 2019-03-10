import collections
import numpy as np
from typing import TextIO, Tuple

import yaml

import ezcv.operator as op_lib
from ezcv import utils
from ezcv.pipeline.context import PipelineContext


class CompVizPipeline(object):
    def __init__(self):
        self.operators = collections.OrderedDict()

    def run(self, img: np.ndarray) -> Tuple[np.ndarray, PipelineContext]:
        self._raise_if_invalid_img(img)
        last = img
        ctx = PipelineContext(img)
        for name, operator in self.operators.items():
            with ctx.scope(name):
                last = operator.run(last, ctx)
            if not utils.is_image(last):
                raise ValueError('Invalid return value from "%s": "%s"' % (name, str(last)))
        return last, ctx

    def _raise_if_invalid_img(self, img: np.ndarray):
        if not utils.is_image(img):
            raise ValueError('Invalid image provided')

    def add_operator(self, name: str, operator: op_lib.Operator):
        self._raise_if_name_is_unavailable(name)
        self.operators[name] = operator

    def _raise_if_name_is_unavailable(self, name: str):
        if name in self.operators:
            raise ValueError('Trying to add a duplicated name: %s' % name)

    @staticmethod
    def load(stream: TextIO) -> "CompVizPipeline":
        pipeline_config = yaml.load(stream)
        runner = CompVizPipeline()
        for op_config in pipeline_config['pipeline']:
            operator = op_lib.create_operator(op_config['config'])
            runner.add_operator(op_config['name'], operator)
        return runner
