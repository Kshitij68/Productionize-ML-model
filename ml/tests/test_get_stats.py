import pytest

from ml.titanic import Titanic
__author__ = 'Kshitij Mathur (kshitij@gmail.com)'

@pytest.mark.unit
def test_get_stats():
    data = [{'A': 1, 'B': 2, 'C': 3},
            {'A': 10, 'B': 20, 'C': 30},
            {'A': 100, 'B': 200, 'C': 300},
            {'A': 1000, 'B': 2000, 'C': 3000}]
    test = 'male'
    assert Titanic.encode_gender(test) == [1, 0]

    test = 'female'
    assert Titanic.encode_gender(test) == [0, 1]

    test = 'pokemon'
    assert Titanic.encode_gender(test) == [1, 0]
