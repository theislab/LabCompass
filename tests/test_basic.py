import pytest

import sc_exp_design


def test_package_has_version():
    assert sc_exp_design.__version__ is not None


@pytest.mark.skip(reason="This decorator should be removed when test passes.")
def test_example():
    assert 1 == 0  # This test is designed to fail.
