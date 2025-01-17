import os

import mypy.api
import pytest

from tests import TEST_DIR


def _mypy_check_file(fn, raises):
    result = mypy.api.run([os.path.join(TEST_DIR, "testdata", "typing", fn)])
    if raises:
        assert result[2] != 0, result
    else:
        assert result[2] == 0, result


TEST_CASES = [
    ("predict_ok.py", False),
    ("predict_bad.py", True),
    ("predict_pydantic_ok.py", False),
    ("predict_pydantic_bad.py", True),
    ("predict_list.py", False),
    ("library_get_model_ko.py", True),
    ("library_get_model_ok.py", False),
]


@pytest.mark.parametrize("fn, raises", TEST_CASES)
def test_typing_model_predict(fn, raises):
    _mypy_check_file(fn, raises)
