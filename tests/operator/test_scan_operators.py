from ezcv.operator import get_available_operators, Operator


def test_get_available_operators_return():
    r = get_available_operators()
    assert iter(r) is not None
    for i in r:
        assert issubclass(i, Operator)


def test_get_available_operators_returns_something():
    assert len(get_available_operators()) > 0


def test_get_available_operators_cache():
    assert get_available_operators() is get_available_operators()
