from sklearn.metrics import confusion_matrix
import numpy as np
from collections import OrderedDict
from itertools import chain

class MaQcException(Exception):
    """Custom Exception"""
    def __init__(self, platform=None, message="MAQC Pipeline Exception."):
        if platform:
            ex_message = "Platform {} not implemented.".format(platform)
        else:
            ex_message = message
        super().__init__(ex_message)


def balanced_accuracy_score(y_true, y_pred, sample_weight=None,
                            adjusted=False):
    """taken from sklearn v20.0
    https://github.com/scikit-learn/scikit-learn/blob/bac89c2/sklearn/metrics/classification.py#L1371

    not using `adjusted` option.
    """
    C = confusion_matrix(y_true, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diag(C) / C.sum(axis=1)
    if np.any(np.isnan(per_class)):
        warnings.warn('y_pred contains classes not in y_true')
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    return score


def adjusted_classes(y_scores, thresh = 0.5):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Assumes binary classification problem.
    """
    return [1 if y >= thresh else 0 for y in y_scores]


def flatten_dict(d, is_init = True):
    """Utility to flatten dictionary keys and values.

    Names of keys are joined with '.', excluding the top-level name (see is_init).

    Parameters
    -------------
    is_init: bool
        True drops top_level dictionary key from the chain of names. Used not to have
        sample name in feature names.


    Notes:
        use OrderedDict to assure insertion order on python below 3.6.

    """
    def items_():
        for key, value in d.items():
            if isinstance(value, dict):
                for subkey, subvalue in flatten_dict(value, False).items():
                    if is_init:
                        yield subkey, subvalue
                    else:
                        yield key + "." + subkey, subvalue
            else:
                yield key, value

    return dict(items_())

def get_values_from_dict(d):
    """Flattens values of a nested dictionary to a single array

    Parameters
        d, dict
    Returns
        list, values from all leaf nodes of dict
    References
        https://codereview.stackexchange.com/a/21035
    """
    ret_dict = flatten_dict(d, is_init = False)
    return list(chain.from_iterable(ret_dict.values()))
