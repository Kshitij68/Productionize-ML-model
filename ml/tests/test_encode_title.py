import pytest

from ml.titanic import Titanic
__author__ = 'Kshitij Mathur (kshitij@gmail.com)'

@pytest.mark.unit
def test_convert_to_float():
    encode_gender = Titanic.encode_gender

    test = 'male'
    assert encode_gender(test) == [1, 0]

    test = 'female'
    assert encode_gender(test) == [0, 1]

    test = [1,12,2,3,4]
    assert encode_gender(test) == [0, 1]
