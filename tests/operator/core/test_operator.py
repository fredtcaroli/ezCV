import pytest

from ezcv.operator import Operator


def test_operator_not_implemented():
    operator = Operator()
    with pytest.raises(NotImplementedError):
        operator.run(None)
