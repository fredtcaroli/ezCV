from typing import TextIO, Tuple, Dict, List, Optional, Any

import yaml

import ezcv.operator as op_lib
from ezcv import utils
from ezcv.pipeline.context import PipelineContext
from ezcv.typing import Image


class CompVizPipeline(object):
    def __init__(self):
        self._operators: Dict[str, op_lib.Operator] = dict()
        self._operators_order: List[str] = list()

    @property
    def operators(self) -> Dict[str, op_lib.Operator]:
        return {op_name: self._operators[op_name] for op_name in self._operators_order}

    def run(self, img: Image) -> Tuple[Image, PipelineContext]:
        _raise_if_invalid_img(img)
        last = img
        ctx = PipelineContext(img)
        for name, operator in self.operators.items():
            with ctx.scope(name):
                try:
                    last = operator.run(last, ctx)
                except Exception as e:
                    raise OperatorFailedError(f'Operator {name} failed to run with message "{e}"') from e
            _raise_if_invalid_img(last, returned_from=name)
        return last, ctx

    def add_operator(self, name: str, operator: op_lib.Operator):
        self._raise_if_name_is_unavailable(name)
        self._operators[name] = operator
        self._operators_order.append(name)

    def rename_operator(self, name: str, new_name: str):
        if name not in self._operators:
            raise ValueError(f'Unexistent operator name: "{name}"')
        self._raise_if_name_is_unavailable(new_name)
        self._operators_order[self._operators_order.index(name)] = new_name
        self._operators[new_name] = self._operators.pop(name)

    def _raise_if_name_is_unavailable(self, name: str):
        if name in self._operators:
            raise ValueError('Trying to add a duplicated name: %s' % name)

    @staticmethod
    def load(stream: TextIO) -> "CompVizPipeline":
        from ezcv.config import create_pipeline
        pipeline_config = yaml.safe_load(stream)
        return create_pipeline(pipeline_config)

    def save(self, stream: TextIO):
        from ezcv.config import get_pipeline_config
        config = get_pipeline_config(self)
        yaml.safe_dump(config, stream, sort_keys=False)


def _raise_if_invalid_img(img: Image, returned_from: Optional[str] = None):
    if not utils.is_image(img):
        message = 'Invalid image'
        if returned_from is not None:
            message += f' returned from "{returned_from}"'
        message += f': {img}'
        raise BadImageError(message)


class OperatorFailedError(Exception):
    pass


class BadImageError(Exception):
    pass
