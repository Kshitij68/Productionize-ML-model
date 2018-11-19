import pytest

from ml.titanic import Titanic
__author__ = 'Kshitij Mathur (kshitij@gmail.com)'


@pytest.mark.unit
def test_dict_to_array():
    dict_to_array = Titanic.dict_to_array

    test = {}
    assert dict_to_array(test) == []

    test = {'key1':'1','key2':2}
    assert dict_to_array(test) == []

    test = [{'key1': '1', 'key2': 2}, {'key1': '12', 'key2': 22}, {'key1': '13', 'key2': 23},
            {'key1': '14', 'key2': 24}]
    assert dict_to_array(test) == [['1', 2], ['12', 22], ['13', 23], ['14', 24]]
