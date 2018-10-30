import numpy as np

from sklearn.utils.class_weight import compute_class_weight

from scipy import stats as scstats
from scipy.stats import rv_continuous, rv_discrete
from scipy.stats import lognorm, gamma, randint, uniform

class ClassWeights(object):
    """
    Draw random variates for cases when parameter is a list, e.g. for class_weights

    You expect two classes that can have different distibution of weights (with
    corresponding different parameters).

    Paramters
    -----------
    dist_names: list
        Names of distributions for class weights, see scipy.stats for available
    dist_params: list of lists, tuples or dicts
        Parameters of the distribution, named (as dict) or positional.


    """
    def __init__(self, y, dist_names = ["gamma"], dist_params = [], *args, **kwargs):
        self.class_weights = compute_class_weight("balanced", np.unique(y), y)
        self.dist_names = dist_names
        self.dist_params = dist_params
        self._make_dists()

    @property
    def dist_names(self):
        return self._dist_names

    @dist_names.setter
    def dist_names(self, value):
        if isinstance(value, (str, )): value = [value]
        len_diff = len(self.class_weights) - len(value)

        if len_diff >= 0 :
            value += [value[0]]*len_diff
        else:
            value = value[0:len_dif]
        self._dist_names = value

    @property
    def dist_params(self):
        return self._dist_params

    @dist_params.setter
    def dist_params(self, value):
        if not isinstance(value, (list, )): value = [value]
        if len(value) > 0:
            if all([isinstance(x, float) for x in value]):
                # Multiplication by a factor
                assert (len(value) == len(self.class_weights))
                value *= self.class_weights
            # else it should be an int treated directly as a param
        if len(value) == 0:
            try:
                value = self.class_weights.tolist()
            except:
                value = self.class_weights

        len_diff = len(self.dist_names) - len(value)
        if len_diff >= 0:
            value += [value[0]]*len_diff
        else:
            value = value[0:len_dif]
        self._dist_params = value

    def _make_dists(self):
        self.dists = []
        n_classes = len(self.class_weights)
        for i in range(n_classes):

            dist_name = self.dist_names[i]
            try:
                dist_ = getattr(scstats, dist_name)
            except AttributeError as e:
                NotImplementedError # should not happend

            param_ = self.dist_params[i]
            # dist_name += str(i)
            # setattr(self, dist_name) = dist_(param)
            if isinstance(param_, (dict, )):
                distr = dist_(**param_)
            elif isinstance(param_, (tuple,list, )):
                distr = dist_(*param_)
            else:
                distr = dist_(param_)

            self.dists.append(distr)

    def rvs(self, *args, **kwargs):
        """override method for drawing random variates"""
        ret_val = {k:v.rvs(*args, **kwargs) for k, v in enumerate(self.dists)}
        return ret_val

# class EstimatorWeights(ClassWeights):
#   """Here you could define distributions also for the estimator weights.
#
#   Given that we use just one or two of them, I opted for simpler aproach and
#   pass in a list with granularity 0.1 instead if complete distribution. Works fine.
#
#    """
#     def __init__(self,y, *args, **kwargs):
#         super(EstimatorWeights, self).__init__(*args, **kwargs)
#
#     def rvs(self):
#         ret_val = [ self.dist0.rvs(*args, **kwargs),
#                     self.dist1.rvs(*args, **kwargs)]
#         return ret_val


# Subclassing approach not fruitful, abandonned.
# This is potentially neater way how to do it, but has more restictions.

# class ContinuousDistribution(rv_continuous):
#     def __init__(self, *args, **kwargs):
#
#         super(ContinuousDistribution, self).__init__(*args, **kwargs)

# class ClassWeights(rv_continuous):
#     """
#     Draw random variates for cases when parameter is a list, e.g. for class_weights
#     Currently very simplistic, but can be extended as needed
#     """
#     def __init__(self,y, *args, **kwargs):
#         self.class_weights = compute_class_weight("balanced", np.unique(y), y)
#         self._make_dist()
#         self.a = 0
#
#         super(ClassWeights, self).__init__(*args, **kwargs)
#
#     # @property
#     # def dist1(self):
#     #     return self._dist1
#     #
#     # @dist1.setter
#     # def dist1(self, value):
#     #     if value is None:
#     #         value = self.dist0
#     #     self._dist1 = value
#
#     def _make_dist(self):
#         self.dist0 = gamma(self.class_weights[0])
#         self.dist1 = gamma(self.class_weights[1])
#
#     def _rvs(self):
#         """override method for drawing random variates"""
#         ret_val = [self.dist0.rvs(1), self.dist1.rvs(1)]
#         return ret_val
#
#     def _pdf(self, x, a):
#         # gamma.pdf(x, a) = x**(a-1) * exp(-x) / gamma(a)
#         return np.exp(self._logpdf(x, a))
