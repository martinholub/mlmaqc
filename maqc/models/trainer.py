import pandas as pd
import numpy as np
from scipy.stats import lognorm
import pickle
import timeit
import time
import os
import glob
import json
import pprint

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.calibration import CalibratedClassifierCV

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.metrics import classification_report_imbalanced
from sklearn.utils.class_weight import compute_class_weight

from maqc.utils import MaQcException


class ModelTrainer(object):
    """ Model Trainer Class

    This class encapsulates the functionality of training a model
    (e.g. previously created with model.Model class).

    Parameters
    ------------
    model: sklearn.model_selection.RandomizedSearchCV
    verbose: bool or int
    rnd: int, random numberseed
    save_folder: str, which folder to save models into
    allow_calibration: bool
        If classifer does not implement classification yet, should it be calibrated
        with the test data after training? (Prefer False).
    refit: bool
        Refit the model on whole dataset (train+test) after training?
    """
    def __init__(self, model = None, verbose =False, rnd = None, save_folder = None,
                allow_calibration = False, refit = False):

        self.model = model # sklearn model or an empty placeholder
        self.verbose = verbose # bool or int
        self.rnd = rnd # random number seed
        self.save_folder = save_folder # where to save model?
        self.allow_calibration = allow_calibration # allow probability callibration if not done yet?
        self.refit = refit
        # Init
        self.tested_params = []
        self.df_probs = pd.DataFrame()
        self.scores = dict(zip(["train","test","predict"], [np.nan]*3))
        self.confusion_matrix = pd.DataFrame()
        self.report = {}
        self.feature_importances_ = None

    @property
    def rnd(self):
        return self._rnd
    @rnd.setter
    def rnd(self, value):
        if not value:
            value = np.random.randit(0, 2**32-1)
        if not isinstance(value, (int, )): value = np.int(value)
        assert (value>-1 and value < 2**32)
        np.random.seed(value)
        self._rnd = value

    @property
    def save_folder(self):
        return self._save_folder

    @save_folder.setter
    def save_folder(self, value):
        if value is None: value = "saved"
        value = os.path.abspath(os.path.normpath(value))
        self._save_folder = value


    def _fit(self, X , y):
        """Wrapper for fit for models that do not implement class_weight
        Quick fix how to get in some feature importances
        """
        try:
            self.model.fit(X, y)
        except ValueError as e:
            raise e
            # This is DEPRECEATED and should not happen
            # TODO: remove at next revision
            # class_weight = [
            #     np.mean([x[0] for x in self.model.param_distributions["classification__class_weight"]]),
            #     np.mean([x[1] for x in self.model.param_distributions["classification__class_weight"]])
            # ]
            # sample_weight = np.where(y, class_weight[1], class_weight[0])
            # self.model.fit(X, y, sample_weight)

    def _save_model(self, fname = None):
        """Pickles the model to file

        Parameters
        ------------
        fname: str, [default = None]
            Filename, by default a combination of model name and current datetime.
        Returns
        ------------
        fname: str
            Location of the saved model.
        """
        t = time.localtime()
        if not fname:
            fname = type(self.model).__name__ + \
                    "_{:02d}{:02d}{:02d}{:02d}.pkl".format( t.tm_mon, t.tm_mday,
                                                            t.tm_hour, t.tm_min)

        if not (os.path.isdir(self.save_folder)):
            os.makedirs(self.save_folder)
            print("Created dir at {}.".format(self.save_folder))

        fname = os.path.join(self.save_folder, fname)

        dirname = os.path.dirname(fname)
        if not (os.path.isdir(dirname)): # make dir-tree if not existing
            os.makedirs(dirname)

        with open(fname, "wb") as wf:
            pickle.dump(self.model, wf)
        print("Model saved to {}.".format(fname))
        return(fname)

    def _write_report(self, fname):
        """Collect information on trained model in a plain text file

        Should facilitate tracebility and documentability of tried approaches.
        """

        fname = os.path.splitext(fname)[0] + ".txt" # change extension
        # Obtain parameters of top level estimator (RandomizedSearchCV)
        params_dict = self.estimator.get_params()

        if len(self.report) == 0 or self.scores["predict"] == np.nan:
            msg = "Model must be trained, tested and scored before its report " + \
                  "can be saved. Aborting."
            raise MaQcException(msg)

        with open(fname, 'w') as f:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                f.write(fname + "\n\n")

                f.write("\n# Report (prediction):\n")
                pprint.pprint(self.report, f)

                f.write("\n# Confusion Matrix (prediction):\n")
                pprint.pprint(self.confusion_matrix, f)

                f.write("\n# Scores:\n")
                pprint.pprint(self.scores, f)

                f.write("\n# Best Estimator Params:\n")
                # json.dump(params_dict, f, sort_keys = True, indent = 4)
                pprint.pprint(params_dict, f)

                f.write("\n# Wrongly Labeled Examples:\n")
                df = self.df_probs[self.df_probs["y_true"] != self.df_probs["y_hat"]]
                df.to_csv(f, sep = "\t", mode = "a")

                f.write("\n# Some Correctly Labeled Examples:\n")
                df = self.df_probs
                df = pd.concat([
                        df[(df["y_true"]==df["y_hat"]) & (df["y_hat"]==0)].tail(50),
                        df[(df["y_true"] == df["y_hat"]) & (df["y_hat"]==1)].head(50)])
                df.to_csv(f, sep = "\t", mode = "a")

                f.write("\n# Params Tested in CV:\n")
                pprint.pprint(self.tested_params, f)

    def save_model(self, fname = None):
        """Save model together with information"""

        fname = self._save_model(fname)
        self._write_report(fname)

    def load_model(self, fname = None):
        """Loads model given by fname

        Parameters
        -----------
        fname: str
            Path to *.pkl file with model. If None, attempts to read newest model from
            default location.

        """
        # Get filename
        if not fname: # load newest file
            files = os.path.join(self.save_folder, '*')
            fname = sorted( glob.iglob(files), key=os.path.getctime,
                            reverse=True)[0]
        else:
            dirname = os.path.dirname(fname)
            basename = os.path.basename(fname)
            if not dirname: dirname = self.save_folder
            fname = os.path.join(dirname, basename)

        # Load
        fname = os.path.expanduser(fname)
        with open(fname, "rb") as rf:
            model = pickle.load(rf)
        print("Loaded model from {}.".format(fname))

        # Asign
        self.model = model
        try:
            self.estimator = model.best_estimator_
        except AttributeError as e:
            self.estimator = self.model

    def _refit(self, X, y):
        """Refit the model on full dataset

        Do this on concatenation of [train, test] before deployment of final model.

        paramaters
        ---------
        X: np.ndarray
            samples x features matrix, possibly for all data you have
        y: array-like
            corresponding labels
        """
        if self.refit:
            X = np.concatenate(X)
            y = np.concatenate(y)

            X, y = shuffle(X, y)

            # Make sure you refit both instances of model/estimator
            try:
                self.estimator.fit(X, y)
            except:
                self.model.fit(X,y)
            try:
                self.model.fit(X, y)
            except Exception as e:
                raise e

    def _resample(self, X, y):
        """Deal with heavy imbalance in dataset

        This approach has been DEPRECEATED. TODO: Remove at next revision.
        """

        class_counts = np.sort(np.bincount(y))
        # Apply some reasonable thresholds
        is_imbalanced = (class_counts[1] / class_counts[0]) > 5
        is_enough = (class_counts[0] > 500) and (class_counts[1] > 5000)
        if is_imbalanced:
            if is_enough: # Undersample
                X_res, y_res = RandomUnderSampler().fit_sample(X, y)
            else: # Oversample
                # sampler = RandomOverSampler()
                # sampler = SMOTEENN(smote = SMOTE(kind = "svm"))
                sampler = ADASYN(n_neighbors = 3)
                X_res, y_res = sampler.fit_sample(X, y) # has hyperparameters
        else:
            X_res, y_res = X, y

        return X_res, y_res

    def _shuffle_and_split(self, X, y, test_size = 0.25):
        """Shuffles the data and splits to train and test group.

        Preserves class membership probability. If X and y are lists,
        assumes splitting was done elswehre and just shuffles the data.

        Parameters
        ------------
        X: np.ndarray or list of thereof
        y: array-like, list of thereof
        test_size: float
            Relative size of test set.
        """
        # X, y = self._resample(X, y)

        if isinstance(X, (list, tuple, )):
            assert isinstance(y, (list, tuple, ))
            X_train, y_train = shuffle(X[0], y[0])
            X_test, y_test = shuffle(X[1], y[1])
        else:
            sss =  StratifiedShuffleSplit(n_splits = 1, test_size = test_size)

            for train_idx, test_idx in sss.split(X, y):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_test = X[test_idx]
                y_test = y[test_idx]
        return (X_train, y_train, X_test, y_test)

    def _standardize(self, *Xs):
        """Remove mean and scale to unit variance.

        Parameters
        ------------
        Xs: list of np.ndarray
        """
        if not hasattr(self, "scaler"):
            self.scaler = StandardScaler().fit(Xs[0])
        Xs_out = []
        for i,x in enumerate(Xs):
             Xs_out.append(self.scaler.transform(x))
        return Xs_out

    def _train(self, X_train, y_train):
        """Train the model

        Cross validation should be implemented by wrapping the model in
        RandomizedSearchCV.

        Returns
        --------------
        train_score: float
            Score on training set.
        """
        self.model.fit(X_train, y_train)

        try: # Keep track of the parameter combinations searched-over in training
            self.tested_params.append(self.model.cv_results_["params"])
            if self.verbose > 1:
                print("Probed parameter combinations:\n{}".format(self.tested_params[-1]))
        except:
            pass # no cv run

        try:
            best_estimator = self.model.best_estimator_
            if self.verbose:
                print("Returned best estimator: ")
                print(best_estimator)
        except AttributeError as e:
            best_estimator = self.model

        self.estimator = best_estimator

        print("\nResults on training set:\n")
        train_score = self._score(X_train, y_train)
        self.scores["train"] = train_score

        return train_score

    def _predict(self, X_test, proba = False, which = None):
        """Predicts labels for given data

        Parameters
        ------------
        proba: bool
            Attempt to predict probability of class-assignment?
        which: str
            Which estimator (from potentially multiple) from VotingClasifier to use?
            Prefer using all by setting which=None.
        """
        if which is None:
            estimator = self.estimator
        else:
            est_vals = self.estimator.named_steps.classification.estimators_
            est_keys = self.estimator.named_steps.classification.named_estimators.keys()
            est_dict = dict(zip(est_keys, est_vals))
            assert which in est_dict.keys()
            estimator = est_dict[which]
        if proba and hasattr(estimator, "predict_proba"):
            try:
                y_hat = estimator.predict_proba(X_test)
            except AttributeError as e:
                y_hat = estimator.predict(X_test)
        elif proba and hasattr(estimator, "decision_function"):
            y_hat = estimator.decision_function(X_test)
        else:
            y_hat = estimator.predict(X_test)
        return y_hat

    def _report_missclass_proba(self, X, y_true, y_hat = None):
        """Report On Classification Accuracy

        Parameters
        ---------------
        y_true: array-like, true binary labels
        y_hat: array-like
            Predicted binary labels, will be computed from X if None.
        """
        if not hasattr(self.estimator, "predict_proba"): return
        # Get y_hat
        if y_hat is None:
            try:
                y_hat = self.estimator.predict(X)
            except AttributeError as e:
                y_hat_proba = self.estimator.predict_proba(X)
                clss = self.estimator.classes_
                y_hat = np.asarray([clss[0] if x[0]>0.5 else clss[1] for x in y_hat_proba])

        try:
            y_hat_proba = self.estimator.predict_proba(X)
        except AttributeError as e:
            # some parameters of VotingClasifier will disable predict_proba
            return

        # sort decreasingly by probability of classifying as 0
        idxer = np.argsort(y_hat_proba[:,0])[::-1]
        df = pd.DataFrame(y_true[idxer], columns=["y_true"])
        df["y_hat"] = y_hat[idxer]
        for cl in self.estimator.classes_:
            df["probs {}".format(cl)] = y_hat_proba[idxer, cl]
        df["pos"] = idxer

        self.df_probs = df

    def get_prediction_certainity(self, df = None, verbose = None):
        """Convenience extractor of predicitons certainity

        Parameters
        -----------
        df: pandas.DataFrame
            must have clomns (y_hat, y_true, probs 0, probs 1)
        Returns
        ------------
        wrong: pandas.DataFrame
            Probabilities of wrongly labeled (sorted by uncertanity).
        correct: pandas.DataFrame
            Probabilities of correctly labeled (sorted by uncertanity).
        """
        # self._report_missclass_proba(X, y_true, y_hat) # can also call here to get df
        if df is None: df = self.df_probs
        if df.shape[0] == 0:
            return # not fitted yet
        if verbose is None:
            verbose = self.verbose

        # Make Sure that sorted
        try:
            df = df.sort_values("probs 0", axis = 0, ascending = False)
        except AttributeError as e:
            pass # cannot sort

        wrong = df[df["y_hat"] != df["y_true"]].copy()
        try: # sort increasinlgy by probability margin
            wrong["prob_diff"] = np.abs(wrong["probs 0"] - wrong["probs 1"])
            wrong = wrong.sort_values("prob_diff", axis = 0)
        except AttributeError as e:
            pass

        correct = df[df["y_hat"] == df["y_true"]].copy()
        try: # sort increasinlgy by probability margin
            correct["prob_diff"] = np.abs(correct["probs 0"] - correct["probs 1"])
            correct = correct.sort_values("prob_diff", axis = 0)
        except AttributeError as e:
            pass

        correct_0 = correct[correct["y_hat"] == 0]
        correct_1 = correct[correct["y_hat"] == 1]

        if df.shape[0] > 0 and verbose > 1:
            print(  "Probabilities of wrongly labeled (sorted by uncertanity):\n{}".\
                    format(wrong.head(5)))
            print(  "\nProbabilities of the most uncertain but correctly labeled:\n{}".\
                    format(pd.concat([correct_0.head(5), correct_1.head(5)])))

        return wrong, correct

    def _score(self, X_test, y_test, scoring_fun = []):
        """Scores prediciton using previously defined scoring function

        Parameters
        -------------
        X_test: np.ndarray
        y_test: array-like
        scoring_fun: list or None
            Scoring functions to use for scoring as list of callables. By default,
            pass [] and use scoring implemented by the model.

        Returns
        -----------
        score: float, list of thereof
            Score(s) computed by estimator's `score` method or by scoring_fun (if supplied).
        """

        if y_test is None:
            return None

        X_test, y_test = shuffle(X_test, y_test)

        if not isinstance(scoring_fun, (list, )): scoring_fun = [scoring_fun]
        scoring_fun = list(filter(bool, scoring_fun)) # drop empty entries

        if len(scoring_fun) > 0: # Use custom scoring function
            score = []
            for sf in scoring_fun:
                score.append(sf(self._predict(X_test), y_test))
        else: # Score with estimators `score` method
            try:
                y_hat = self.estimator.predict(X_test)
            except AttributeError as e:
                y_hat_proba = self.estimator.predict_proba(X_test)
                clss = self.estimator.classes_
                y_hat = np.asarray([clss[0] if x[0]>0.5 else clss[1] for x in y_hat_proba])

            try:
                scoring_fun = self.model.scoring
                score = self.estimator.score(X_test, y_test)
            except: # Fall back on f1_score if estimator does not implement any scoring.
                scoring_fun = 'f1_score'
                score = f1_score(y_test, y_hat)

        if self.verbose:
            print("The prediction score (scoring: {}) is {}.".format( scoring_fun, score))

        self._report_missclass_proba(X_test, y_test, y_hat)

        # true negatives is C_{0,0}, false negatives is C_{1,0},
        # true positives is C_{1,1} and false positives is C_{0,1}
        # to improve: maximize main diagonal, minimize off-diagonal
        self.confusion_matrix =pd.DataFrame(confusion_matrix(y_test, y_hat),
                                            columns=['pred_neg', 'pred_pos'], index=['neg', 'pos'])
        self.report = classification_report(y_test, y_hat)
        print("Confusion matrix:\n", self.confusion_matrix)
        print("Classification report: \n", self.report)
        # print("Classification report imbalanced: \n", classification_report_imbalanced(y_test, y_hat))
        return score

    def _test(self, X_test, y_test):
        """Accuracy on test
        Wrapper to compute score on test set
        """
        print("\nResults on test set:\n")
        test_score = self._score(X_test, y_test) # test score
        self.scores["test"] = test_score
        return test_score

    def calibrate(self, X, y):
        """Calibrate Probability of prediciton

        Alternatively, wrap classifer to CalibratedClassifierCV (prefered)
        """

        try:
            estimator = self.estimator
        except AttributeError as e:
            estimator = self.model.best_estimator_
            self.estimator = estimator

        steps = list(estimator.named_steps.keys())
        clf_classif = getattr(estimator.named_steps, steps[-1])

        if isinstance (clf_classif, (CalibratedClassifierCV, )):
            # Model has calibrated classification
            print("Model already implements calibration and thus won't be recalibrated.")
            pass
        elif all(v is not None for v in [X, y]) and self.allow_calibration:
            # We will wrap the whole model
            print("Fitted model will be calibrated with CalibratedClassifierCV.")
            clf = CalibratedClassifierCV(estimator, cv = "prefit", method='sigmoid')
            X, y = shuffle(X, y)
            clf.fit(X, y)
            self.estimator = clf


    def _assign_feature_importances(self):
        """Assign feature importances in potentially nested model"""

        # Pull out classification step
        try:
            estimator = self.estimator
            steps = list(estimator.named_steps.keys())
        except AttributeError as e:
            estimator = self.estimator.calibrated_classifiers_[0].base_estimator
            steps = list(estimator.named_steps.keys())

        clf_classif = getattr(estimator.named_steps, steps[-1])

        importances = []
        if isinstance (clf_classif, (CalibratedClassifierCV, )):
            ests = [e for c in clf_classif.calibrated_classifiers_ for e in c.base_estimator.estimators_]
        else:
            ests = clf_classif.estimators_

        # Obtain feature importances from the model
        for est in ests:
            try:
                try:
                    imp = est.feature_importances_
                except AttributeError as e:
                    imp = np.abs(est.coef_) # TODO: Only magnitutde?
            except AttributeError as e: # not even .coef_ useful
                # set weight of features to one
                # TODO: Extend beyond SVC case once needed
                imp = np.repeat([1], est.support_vectors_.shape[1])
            importances.append(imp)

        # Take mean importance of features across estimators
        # TODO: The scale may be different between estimators and there is
        # usually 1-3 of them, what is more robust way here? Scaling by variance?
        importances = np.asarray(importances) # nestimators x nfeatures
        importances = np.mean(importances, axis=0)

        self.feature_importances_ = importances

    def train_test( self, X, y,
                    load = False, save = False, fname = None):
        """Train and test the model

        Splits data to train and test and runs CV for hyperparameters selection
        on train data (model must be wrapped in CV wrapper e.g. RandomizedSearchCV).
        Final model is then refit to the whole dataset and can be otionaly saved.

        Alternatively, model can be loaded from pickle file. Then model is taken as is
        and only score on the whole dataset is printed out.

        Obtained model can be used for predictions on unseen data.

        Parameters
        -------------
        X: ndarray, data
        y: array-like, labels
        load: bool, attempt to load from file?
        save: bool, save to file? Used only if load = False
        fname: str, file name
        folder: str, directory name

        Returns
        --------------
        test_score: float
            Score obtained on test set
        """

        start_time = timeit.default_timer()

        if load or (self.model is None):
            self.load_model(fname)
            test_score = self._score(X, y)
        else:
            X_train, y_train, X_test, y_test = self._shuffle_and_split(X, y)

            train_score = self._train(X_train, y_train) # runs CV on hyperparams

            test_score = self._test(X_test, y_test)
            self.calibrate(X_test, y_test) # done only if model not calibrated yet
            self._refit([X_train, X_test], [y_train, y_test]) # should be done before deployment
            self._assign_feature_importances()

            if save:
                self._save_model(fname)

        end_time = timeit.default_timer()
        if self.verbose:
            print("Finished in {:.3f} s".format(end_time - start_time))

        return test_score

    def predict(self, X, y = None, proba = False, which = None):
        """Predict labels on unseen data

        If true labels available, can report score. Assumes training was done previously,
        yielding a `scaler` object that can be used for standardizing data.

        Parameters
        ----------------
        X: np.ndarray
        y: array-like, optional
            True labels, if provided, prints prediciton scores.
        proba: bool
            Predict with probability?
        which: str
            Name of nested estimator to use for prediciton. By default use the top-level
            estimator. Debug only.

        Returns
        ----------------
        y_hat: np.ndarray
            Label predictions. 1D or nD, where n-number of classes, if proba is True
        """

        y_hat = self._predict(X, proba, which)

        if y is not None:
            print("\nResults for current  prediction:\n")
            score = self._score(X, y)
            self.scores["predict"] = score

        return y_hat

    def get_support(self):
        """Obtain Support (selected feautures) from feature selection step

        Returns
        ----------
        support: list
            List of boolean arrays, indicating if corresponding features were kept.

        """

        try:
            est = self.estimator
        except AttributeError as e:
            est = self.estimator.base_estimator
        featurer = est.named_steps.feature_selection

        transformers = [x[1] for x in featurer.transformer_list]
        for name in {"val_thresholder", "var_thresholder"}: # hardcoed :(
            try:
                transformers.append(getattr(est.named_steps, name))
            except AttributeError as e:
                pass # does not hasattr

        support = []
        for tr in transformers:
            try:
                support_ = tr.get_support()
            except AttributeError as e:
                raise e # hopefully not happening, would fill with [True]*?.shape

            support.append(support_)
        return support
