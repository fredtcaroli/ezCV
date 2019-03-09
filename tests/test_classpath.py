from ezcv.classpath import fully_qualified_name


class TestClass(object):
    pass


def test_fully_qualified_name_return():
    fqn = fully_qualified_name(TestClass)
    assert isinstance(fqn, str)


def test_fully_qualified_name():
    fqn = fully_qualified_name(TestClass)
    assert fqn == __name__ + '.TestClass'
