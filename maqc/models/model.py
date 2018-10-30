from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import ( SelectFromModel, SelectKBest, SelectFdr,
                                        SelectFpr, VarianceThreshold, SelectFwe)
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                             GradientBoostingClassifier, ExtraTreesClassifier,
                             VotingClassifier)
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2
from sklearn.metrics import make_scorer, average_precision_score, roc_auc_score

from maqc.utils import balanced_accuracy_score
from maqc.dists.dists import ClassWeights
from maqc.models.feature_selection import ValueThreshold

from scipy.stats import lognorm, gamma, randint, uniform
import numpy as np
import re

class Model(object):
    """Model

    This class encapsulates the functionality of constructing a model.
    """
    def __init__(self, y, random_state = None, n_jobs = 1, verbose = 0):
        self.y = y
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        # Init Vars
        self.model = None
        self.param_dists = {}
        self.estimators = []

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, value):
        if not value:
            value = np.random.randint(0, 2**32-1)
        if not isinstance(value, (int, )): value = np.int(value)
        assert (value>-1 and value < 2**32)
        np.random.seed(value)
        self._random_state = value

    def make_weights(self, nestimators = 2, nweights = 50):
        """Generate Radom Weights that Sum to 1

        Parameters:
            nestimators: int, number of estimators
            nweights: int, number of weight tuple to generate
        Returns:
            weights: list of lists, with each list with weights for each estimator.
                     Returns None if just one estimator (uniform weights)
        """
        if nestimators == 1: return [None] # No point in creating weights

        weights = []
        for k in range(nweights):
            # temp = uniform(0,1.).rvs(nestimators) # no need to be so granular
            temp = np.random.choice(np.linspace(0.1,1., 10), nestimators)
            temp = temp/temp.sum()
            weights.append(temp.tolist())

        return weights

    def _make_scaler(self):
        """Add scaler to pipeline"""
        self.estimators.append(("scaler", StandardScaler()))

    def _make_variance_thresholder(self):
        """Add Zero-Variance Dropper to the pipeline"""
        self.estimators.append(("var_thresholder", VarianceThreshold()))

    def _make_value_thresholder(self, thresh = None, kept_features_id = None):
        """Add Value-based Dropper to the pipeline"""
        self.estimators.append(("val_thresholder", ValueThreshold(thresh, kept_features_id)))

    def make_feat_transformer(self, do_threshold = True, do_scale = True, params = {}):
        """Add feature transformation to the pipeline"""

        if do_threshold: # Implements also Variance Threshold!
            self._make_value_thresholder(params["val_thresh"], params["kept_features_id"])
        # Applies Scaling only after the values have been transformed
        if do_scale:
            self._make_scaler()

    def make_feat_selector(self, which = None):
        """Add feature selection to pipeline

        Feature selectors can be selected by their corresponding id. Their results
        will be concatenated with FeatureUnion. You can add new feature selectors under
        new ids as needed.

        which: int or None
            Prefer which=0 to None for preserving the expected levels of hiearchy.
        """
        if which is None:
            print("Using mock feature_selector as which is `None`.")
            which = [0]
        if isinstance(which, (int, )): which = [which]

        feats_list = []
        param_dists = {}

        for i, wh in enumerate(which):
            if wh is None: continue
            name = "feat" + str(wh)
            dname = "feature_selection__{}__".format(name)

            if wh == 0: # mock feature selector
                feat = SelectKBest(k = "all")
                param_dist = {}

            elif wh in [1, 2, 3]:
                # Use linear SVC as feature selector
                feat = LinearSVC(verbose = 0, max_iter = 20000, penalty = "l1",
                         random_state = self.random_state,
                         loss = "squared_hinge", dual = False)

                weights_distr = ClassWeights(self.y,
                                            dist_names = ["gamma", "gamma"],
                                            dist_params = [10, 1.5])
                param_dist = {
                    dname+"estimator__C": lognorm(s = 1, loc = 0, scale =1),
                    dname+"estimator__class_weight": weights_distr,
                }

                # These are different methods how to get feature-selection from estimator
                if wh == 1:
                    feat = SelectFromModel(feat, prefit = False)
                    param_dist.update({
                        dname+"threshold": uniform(.5, 1.5),
                    })
                elif wh == 2: # Recursive feature elimination
                    feat = RFE(feat, verbose = self.verbose)
                    param_dist.update({
                        dname+"n_features_to_select": randint(10, 50),
                    })
                elif wh == 3: # Recursive feature elimination w/ CV
                    feat = RFECV(feat, verbose = self.verbose)
                    param_dist.update({
                        dname+"min_features_to_select": [1, 10],
                    })

            elif wh == 4: # False Positive Rare
                feat = SelectFpr()
                param_dist = {
                    dname+"alpha": uniform(0.001, 0.01),
                }
            elif wh == 5: # False Discovery rate
                feat = SelectFdr()
                param_dist = {
                    dname+"alpha": uniform(0.001, 0.01),
                }
            elif wh == 6: # Family-wise Error
                feat = SelectFwe()
                param_dist = {
                    dname+"alpha": uniform(0.001, 0.01),
                }
            elif wh == 7:
                feat = SelectKBest(k = "all")
                param_dist = {
                    dname+"k": randint(10, 50),
                    # dname+"score_func": [f_classif, mutual_info_classif]
                }
            else:
                raise NotImplementedError

            feats_list.append((name, feat))
            param_dists.update(param_dist)

        feats = FeatureUnion(feats_list, n_jobs = self.n_jobs)

        self.param_dists.update(param_dists)
        self.estimators.append(("feature_selection", feats))

    def make_classsifier(self, which = 0, weights = None):
        """Make classifier, possibly as combination of multiple

        If multiple classifers selected through which, they will be combined in
        Voting classifer and their weights will be searched-over by RandomizedSearchCV
        (if weights == "random").

        Prefer to include `5` in `which` to do probability calibration. `which` can be
        extended to new classifers as needed.

        I observed the best perfromance for `1` and/or `2` (combined with `5`),
        corresponding to RandomForestClassifier and AdaBoostClassifier.

        Given the strong imbalance in train/test set, important parameter for each
        classifer is the distribution of weights for individual classes. These
        are constructed with ClassWeights and allow for selecting arbiratry distributons
        from scipy.dists together with parameters. The distributions should be kept
        rather broad and give (much) higher weights to minority class. On training,
        these distributions are searched over by RandomizedSearchCV.

        Also other parameters of the classifers should be defined as distributions
        whenever possible.

        Parameters
        -------------
        which: int or list
        weights: None, "random" or list of same length as which
            Weights of the classifers in VotingClassifier

        See also:
            sklearn.calibration.CalibratedClassifierCV
        """
        if isinstance(which, (int, )): which = [which]

        clsfiers = []
        param_dists = {}

        for i, wh in enumerate(which):
            i += 1 # shift to 1-based indexing
            name = "clsf" + str(wh)
            dname = "classification__{}__".format(name)

            if wh == 1:
                clsf = RandomForestClassifier(verbose = self.verbose, n_jobs = self.n_jobs,
                               random_state = self.random_state, oob_score = False)
                weights_distr = ClassWeights(self.y,
                                            dist_names = ["gamma", "gamma"],
                                            dist_params = [{"a": 10}, {"a": 1.5}])
                param_dist = {
                    dname+"n_estimators": randint(3, 10),
                    dname+"criterion": ["gini", "entropy"],
                    dname+"max_features": ["auto", None],
                    dname+"class_weight": weights_distr,
                }
            elif wh == 2:
                clsf = AdaBoostClassifier(base_estimator =
                              DecisionTreeClassifier(max_depth = 1, presort=True),
                           random_state = self.random_state)

                weights_distr = ClassWeights(self.y,
                                            dist_names = ["gamma", "gamma"],
                                            dist_params = [10, 1.5])
                param_dist = {
                    dname+"n_estimators": randint(3, 30),
                    dname+"learning_rate": uniform(0.,1.),
                    dname+"base_estimator__class_weight": weights_distr,
                    dname+"base_estimator__criterion": ["gini", "entropy"],
                    dname+"base_estimator__max_features": ["auto", None],
                }
            elif wh == 3:
                clsf = SVC( probability = True, verbose = self.verbose,
                            random_state=self.random_state, max_iter = 1e7,
                            decision_function_shape="ovr")

                weights_distr = ClassWeights(self.y,
                                            dist_names = ["gamma", "uniform"],
                                            dist_params = [{"a": 10}, {"loc":.1, "scale":1}])
                param_dist = {
                    dname+"C": np.logspace(-2,10,13), #lognorm(s = 1, loc = 0, scale =1),
                    dname+"gamma": np.logspace(-9, 3, 13),
                    dname+"tol": np.logspace(-9,-3,7),
                    dname+"class_weight": weights_distr,
                }
            elif wh == 4:
                clsf = LinearSVC(verbose = self.verbose,
                            random_state=self.random_state, max_iter = 1e5,
                            dual=False, loss= "squared_hinge")

                weights_distr = ClassWeights(self.y,
                                            dist_names = ["gamma", "uniform"],
                                            dist_params = [{"a": 10}, {"loc":.1, "scale":1}])
                param_dist = {
                    dname+"penalty": ["l1", "l2"],
                    dname+"C": np.logspace(-2,10,13), #lognorm(s = 1, loc = 0, scale =1),
                    # dname+"tol": np.logspace(-9,-3,7),
                    dname+"class_weight": weights_distr,
                }
            elif wh == 5:
                # Currently bit cumbersome, later make Calibration default and remove here.
                continue # see below, Crossvalidation CV
            else:
                raise NotImplementedError

            clsfiers.append((name, clsf))
            param_dists.update(param_dist)

        # Make classifier weights
        if weights is not None:
            assert isinstance(weights, (str, list, ))
            which_ = [w for w in which if w != 5]
            if weights == "random":
                weights = self.make_weights(len(which_))
            elif isinstance(weights, (list, )):
                assert len(weights) == len(which_)
            else:
                raise NotImplementedError
        else:
            weights = [weights]

        # Voting Classifier ----------------------------------------------------
        clsf_final = VotingClassifier(
            clsfiers, n_jobs = self.n_jobs, flatten_transform = True,
        )
        # LinearSVC does not support 'soft' weighting
        param_dists.update({
            "classification__voting": ["hard"] if 4 in which else ["soft"], #["soft"]
            "classification__weights": weights,
        })
        # CalibratedClassifierCV -----------------------------------------------
        # Note that this does CV which may introduce randomness.
        if 5 in which: # some high enough number
            clsf_final = CalibratedClassifierCV(clsf_final)
            param_dists = {
                re.sub("classification", "classification__base_estimator", k): v \
                for k,v in param_dists.items()
            }
            param_dist.update({
                "classification__method": ["sigmoid", "isotonic"],
                "classification__cv": [2, 3]
            })

        self.param_dists.update(param_dists)
        self.estimators.append(("classification", clsf_final))

    def make_pipe(  self, feature_transformation = {}, feature_selection = {},
                    classification = {}, *args, **kwargs):
        """Construct Pipeline

        Fills `self.estimators` with tuples ("name", <estimator>) and passes them
        to sklearn.pipeline.Pipeline. Parameters for individual steps are passed in as dicts
        or directly.

        Parameters
        -------------
        feature_transformation: dict
            keys: do_threshold, do_scale, params (keys: val_thresh, kept_features_id)
        feature_selection: dict
            keys: which
        classification: list of ints
        weights: None, "random" or list of same length as classification
            Weights of the classifers in classification in VotingClassifier
        """
        self.make_feat_transformer(**feature_transformation)
        self.make_feat_selector(**feature_selection)
        self.make_classsifier(**classification)
        pipe = Pipeline(self.estimators, memory = None)
        self.pipe = pipe

    def make_search_cv(self, cfg):
        """balanced_accuracy_score

        Parameters
        -----------
        cfg: dict
            Parameters for the Pipeline and estimators (see Example below)

        Returns
        -----------
        model: sklearn.model_selection.RandomizedSearchCV

        Example
        ----------------
        ```
        pipe_cfg = {
            "feature_transformation": {
                "do_threshold": True, # threshold out constant features
                "do_scale": True , # StandardScaler, should be always ON
                "params": {"val_thresh": None, # allow any magnitude of feature
                          "kept_features_id": kept_features}}, # use this to subset if longer vec
            "feature_selection": {"which": [0], # 0 is none (prefered); FPR, FWE worked
                                  },
            # 1 and/or 2 (possibly combined with 5) work.
            # Prefer 5 ON to do probability calibraiton
            "classification": {
                "which": [2, 5],
                "weights": "random", # Will search over weights, prefered.
            },
            "n_iter": 30,
        }
        ```
        """
        self.make_pipe(**cfg)
        model_ = RandomizedSearchCV( # GridSearchCV
            self.pipe, self.param_dists, n_iter = cfg["n_iter"],
            scoring = {
                # "AUC": "roc_auc", # not compatible with voting = "hard"
                "AUC": make_scorer(roc_auc_score, average = "weighted"),
                "BalancedAcc": make_scorer(balanced_accuracy_score),
                "AveragePrecision": make_scorer(average_precision_score, average = "weighted"),
            },
            refit = "BalancedAcc", verbose = self.verbose,
            return_train_score = True,
            error_score = np.nan, random_state=self.random_state,
            n_jobs = self.n_jobs, cv = 3,
        )
        self.model = model_
        return model_
