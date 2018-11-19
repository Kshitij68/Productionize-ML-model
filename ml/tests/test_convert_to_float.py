import pytest

from ml.titanic import Titanic
__author__ = 'Kshitij Mathur (kshitij@gmail.com)'

@pytest.mark.unit
def test_convert_to_float():
    convert_to_float = Titanic.convert_to_float
    test = [{'key1': 1, 'key2': 2}, {'key1': 12, 'key2': 22}, {'key1': 13, 'key2': 23},
            {'key1': 14, 'key2': 24}]
    assert convert_to_float(test) == [{'key1': 1.0, 'key2': 2.0}, {'key1': 12.0, 'key2': 22.0},
                                      {'key1': 13.0, 'key2': 23.0}, {'key1': 14.0, 'key2': 24.0}]
