import pytest

from ml.titanic import Titanic
__author__ = 'Kshitij Mathur (kshitij@gmail.com)'

@pytest.mark.unit
def test_drop_values():
    test = [{'A': 1, 'B': 2, 'C': 3},
            {'A': 10, 'B': 20, 'C': 30}]
    assert Titanic.drop_values(test, ['A'])
