import pytest

from ml.titanic import Titanic
__author__ = 'Kshitij Mathur (kshitij@gmail.com)'

@pytest.mark.unit
def test_encode_gender():
    test = 'male'
    assert Titanic.encode_gender(test) == [1, 0]

    test = 'female'
    assert Titanic.encode_gender(test) == [0, 1]

    test = 'pokemon'
    assert Titanic.encode_gender(test) == [1, 0]
