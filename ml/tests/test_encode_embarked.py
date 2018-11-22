import pytest

from ml.titanic import Titanic
__author__ = 'Kshitij Mathur (kshitij@gmail.com)'

@pytest.mark.unit
def test_encode_embarked():
    test = 'S'
    assert Titanic.encode_embarked(test) == [1, 0, 0]

    test = 'C'
    assert Titanic.encode_embarked(test) == [0, 1, 0]

    test = 'Q'
    assert Titanic.encode_embarked(test) == [0, 0, 1]
