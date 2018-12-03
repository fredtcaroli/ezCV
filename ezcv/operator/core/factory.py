import inspect

from ezcv.config.PipelineConfig_pb2 import OperatorConfig
from ezcv.classpath import class_from_fully_qualified_name
from ezcv.operator.core import Operator
from ezcv.operator.core.parameter import Parameter
from ezcv.operator.core.parameter_adapter import get_adapter


def create_operator(config: OperatorConfig) -> Operator:
    fqn = config.classpath
    cls = class_from_fully_qualified_name(fqn)
    if not issubclass(cls, Operator):
        raise ValueError("%s is not an Operator" % fqn)

    parameters = inspect.getmembers(cls, lambda attribute: isinstance(attribute, Parameter))
    adapters = {name: get_adapter(param) for name, param in parameters}

    op = cls()
    for param_config in config.parameters:
        name = param_config.name
        value = param_config.value
        parsed_value = adapters[name].parse(value)
        setattr(op, name, parsed_value)

    return op
