import inspect

from ezcv.classpath import class_from_fully_qualified_name
from . import Operator
from .parameter import Parameter


def create_operator(config: dict) -> Operator:
    fqn = config['implementation']
    cls = class_from_fully_qualified_name(fqn)
    if not issubclass(cls, Operator):
        raise ValueError("%s is not an Operator" % fqn)

    parameters = {name: value for name, value in inspect.getmembers(cls) if isinstance(value, Parameter)}

    print(parameters)
    op = cls()
    for name, param_config in config['params'].items():
        parsed_value = parameters[name].from_config(param_config)
        setattr(op, name, parsed_value)

    return op
