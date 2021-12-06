from typing import TextIO, Tuple, Dict, List, Optional, Union

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
            if operator.only_gray and last.ndim > 2:
                raise OperatorFailedError(f'Operator {name} expects a gray image')
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

    def remove_operator(self, name_or_index: Union[int, str]):
        index, name = self._identify_operator(name_or_index)
        del self._operators_order[index]
        del self._operators[name]

    def rename_operator(self, name_or_index: Union[int, str], new_name: str):
        index, name = self._identify_operator(name_or_index)
        self._raise_if_name_is_unavailable(new_name)
        self._operators_order[index] = new_name
        self._operators[new_name] = self._operators.pop(name)

    def move_operator(self, name_or_index: Union[int, str], target: int):
        index, name = self._identify_operator(name_or_index)
        if not isinstance(target, int) or target < 0 or target >= len(self._operators_order):
            raise ValueError(f'Invalid move target: {target}')
        self._operators_order.insert(target, self._operators_order.pop(index))

    def get_operator_name(self, index: int) -> str:
        index, name = self._identify_operator(index)
        return name

    @staticmethod
    def load(stream: TextIO) -> "CompVizPipeline":
        from ezcv.config import create_pipeline
        pipeline_config = yaml.safe_load(stream)
        return create_pipeline(pipeline_config)

    def save(self, stream: TextIO):
        from ezcv.config import get_pipeline_config
        config = get_pipeline_config(self)
        yaml.safe_dump(config, stream, sort_keys=False)

    def _identify_operator(self, name_or_index: Union[int, str]) -> Tuple[int, str]:
        """ Returns both the index and name of an operator, given either its index or its name """
        if isinstance(name_or_index, int):  # it's an index
            self._raise_if_index_is_invalid(name_or_index)
            index = name_or_index
            name = self._operators_order[name_or_index]
        elif isinstance(name_or_index, str):  # it's a name
            self._raise_if_name_doesnt_exist(name_or_index)
            index = self._operators_order.index(name_or_index)
            name = name_or_index
        else:
            raise ValueError(f'Operator identifier should be either a string or an integer. Got "{name_or_index}"')
        return index, name

    def _raise_if_name_is_unavailable(self, name: str):
        if name in self._operators:
            raise ValueError('Trying to add a duplicated name: %s' % name)

    def _raise_if_name_doesnt_exist(self, name: str):
        if name not in self._operators:
            raise ValueError(
                f'Trying to select an invalid operator name: "{name}" (from operators {self._operators_order})'
            )

    def _raise_if_index_is_invalid(self, index: int):
        nb_operators = len(self._operators)
        if index < 0 or index >= nb_operators:
            raise ValueError(f'Trying to select an invalid operator index: {index} (from {nb_operators} operators)')


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
