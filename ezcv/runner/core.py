import collections
import numpy as np
from typing import TextIO

import yaml

from ezcv.operator import create_operator, Operator


class Runner(object):
    def __init__(self):
        self.operators = collections.OrderedDict()

    def run(self, img: np.ndarray):
        last = img
        for name, operator in self.operators.items():
            last = operator.run(last)
        return last

    def add_operator(self, name: str, operator: Operator):
        self.operators[name] = operator

    @staticmethod
    def load(stream: TextIO) -> "Runner":
        pipeline_config = yaml.load(stream)
        runner = Runner()
        for op_config in pipeline_config['pipeline']:
            op = create_operator(op_config['config'])
            runner.add_operator(op_config['name'], op)
        return runner
