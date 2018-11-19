import pytest

from ml.titanic import Titanic
__author__ = 'Kshitij Mathur (kshitij@gmail.com)'

@pytest.mark.unit
def test_get_title():
    get_title = Titanic.get_title

    test = 'Mathur, Mr. Kshitij'
    assert get_title(test) == 'Mr'
