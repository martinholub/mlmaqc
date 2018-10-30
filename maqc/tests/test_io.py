# run from parent of package dir with `python -m pytest`
import pytest
from maqc.io import readers as io
from maqc.utils import MaQcException
from os.path import join as osjoin

@pytest.fixture
def fpath():
    return("maqc/tests/data")
@pytest.fixture
def fext():
    return("json")
@pytest.fixture
def test_dict():
    return {"a": {"b": {"c": ["e", 1], "d": ["f", 2]}}}
@pytest.fixture
def flabels():
    return("maqc/tests/data/labels.tsv")

class TestIO(object):

    def test_load_bad_data(self, fpath, fext):
        dr = io.DataReader(osjoin(fpath, "bad"), fext)
        with pytest.raises(MaQcException):
            _, _ = dr.load_sample_data()

    def test_load_good_data(self, fpath, fext):
        dr = io.DataReader(osjoin(fpath, "good"), fext)
        samples, X = dr.load_sample_data()
        assert samples == ["B"]
        assert all(X[0] == [1, 2, 4])

    def test_flatten_dict(self, test_dict):
        assert io.get_from_dict(test_dict) == ["e", 1, "f", 2]

    def test_read_labels(self, flabels):
        lr = io.LabelReader(flabels = flabels, samples = ["A", "B"], verbose = False)
        y, _ = lr.read_labels()
        assert y.index[0] == "B"
