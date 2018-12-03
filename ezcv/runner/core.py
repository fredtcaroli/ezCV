import collections
import numpy as np
from typing import TextIO

from ezcv.config.PipelineConfig_pb2 import PipelineConfig
from ezcv.operator.core import create_operator, Operator


class Runner(object):
    def __init__(self):
        self.operators = collections.OrderedDict()

    def run(self, img: np.ndarray):
        ...

    def add_operator(self, name: str, operator: Operator):
        self.operators[name] = operator

    @staticmethod
    def load(stream: TextIO) -> "Runner":
        pipeline_config = PipelineConfig.ParseFromString(stream)
        runner = Runner()
        for op_config in pipeline_config.operators:
            op = create_operator(op_config)
            runner.add_operator(op.name, op)
        return runner
