import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection.base import SelectorMixin

class ValueThreshold(BaseEstimator, SelectorMixin):
    """Threshold Values by mean on column

    The idea is to discard features that are very different from the others, using
    `drop_id = (colmean < -thresh) | (colmean > thresh)`. By default, use thresh=None,
    which will discard only features **constant** in all training samples.

    Subclassing from sklearn's base allows to integrate this class in the Pipeline.

    Parameters
    ---------------
    thresh: int, float, callable or None
        If int or float, this is the threhsold, if None, all features are kept.
        Alternatively, it can be a callable that takes np.ndarray of feature values for
        training samples and returns the value of threshold.
    kept_features_id: array-like of bool
        Feature ids kept during loading of data (i.e. not blacklisted). Allows to
        feed in feature vecors without the information which features to keep. But
        must have same length as on training!

    See also:
        sklearn.base.BaseEstimator, sklearn.feature_selection.baseSelectorMixin
    """
    def __init__(self, thresh = None, kept_features_id = None):
        self.thresh = thresh
        self.kept_features_id = kept_features_id
        # super().__init__()

    @property
    def thresh(self):
        return self._thresh

    @thresh.setter
    def thresh(self, value):
        if not hasattr(value, "__call__"):
            if value is None:
                value = np.inf
            else:
                assert isinstance(value, (int, float))
        self._thresh = value

    @property
    def kept_features_id(self):
        return self._kept_features_id

    @kept_features_id.setter
    def kept_features_id(self, value):
        if value is None: value = np.ndarray(0, np.int)
        assert isinstance(value, (list, tuple, np.ndarray))
        self._kept_features_id = value

    def transform(self, X):
        """Subset based on column-wise index"""
        if np.ndim(X) == 1: X = np.reshape(X, (1, -1))
        drop_id = self.drop_id
        # You may be passing data that was not subset on features.
        # Pass the info on which features were selected on loading the data here.
        if X.shape[1] != len(drop_id):
            X = np.squeeze(X[:, self.kept_features_id], axis = -1)
        return(X[:, np.invert(drop_id)])

    def fit(self, X, y=None):
        """Obtain column-wise drop-index"""
        if hasattr(self.thresh, "__call__"):
            thresh = self.thresh(X)
        if self.thresh is None:
            thresh = np.inf
        elif isinstance(self.thresh, (int, float)):
            thresh = self.thresh
        else:
            raise NotImplementedError

        colmean = np.mean(X, axis = 0)
        colsd = np.std(X, axis = 0)
        drop_id = (colmean < -thresh) | (colmean > thresh) | (colmean == 0.) | (colsd <= 1e-5)

        self.drop_id = drop_id

        return self

    def _get_support_mask(self):
        assert hasattr(self, 'drop_id')
        return np.invert(self.drop_id)
