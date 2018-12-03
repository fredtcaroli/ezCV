from typing import Any


# This is subclassing `Any` so IDEs don't go crazy with the parameter injection
class Parameter(Any):
    pass


class IntegerParameter(Parameter):
    pass


class NumberParameter(Parameter):
    pass
