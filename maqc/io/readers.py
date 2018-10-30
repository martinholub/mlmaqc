import csv
import os
import re
import json
import glob
import difflib
import timeit
import random
from itertools import chain
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from ..utils import MaQcException, flatten_dict, get_values_from_dict


class DataReader(object):
    """DataReader

    This class encapsulates the functionality of reading data from (json) files.
    """
    def __init__(self, load_dir = ".", fext = ".json", feature_blacklist = None, verbose = True, ):
        self.load_dir = load_dir
        self.fext = fext
        self.verbose = verbose
        self.feature_blacklist = feature_blacklist

    @property
    def load_dir(self):
        """Path to directory containing data"""
        return self._load_dir

    @load_dir.setter
    def load_dir(self, value):
        value = value.rstrip("/")
        value = os.path.expanduser(value)
        assert os.path.isdir(value), "Directory {} doesn't exist".format(value)
        self._load_dir = value

    @property
    def fext(self):
        """Extension of data files"""
        return self._fext

    @fext.setter
    def fext(self, value):
        assert isinstance(value, (str, ))
        if not value.startswith("."): value = "." + value
        self._fext = value

    @property
    def feature_blacklist(self):
        """Beginnings of names of features that are to be disregarded."""
        return self._feature_blacklist

    @feature_blacklist.setter
    def feature_blacklist(self, value):
        if len(value) == 0: value = None
        if value is not None:
            if not isinstance(value, (tuple, )): value = tuple(value)
        self._feature_blacklist = value


    def _glob_ext(self, exps = None):
        """Glob for files in given directory with given extension

        Parameters
        ----------
        exps: list, tuple
            List of experiments to obtain, by default all experiments

        Returns
        -----------
        jsons:  list
            List of filepaths to (json) data files
        """
        if isinstance(exps, (list, )): exps = tuple(exps)
        jsons = glob.glob(self.load_dir + "/*" + self.fext)
        if exps:
            jsons = [j for j in jsons if os.path.basename(j).startswith(exps)]
        return jsons

    def _feature_selector(self, *result):
        """Remove features based on criteria

        Only blacklisted features are removed.

        Parameters
        ------------
        *result: list, tuple
            Data as np.ndarray(s)

        Returns
        -------------
        result_out: np.ndarray or list of np.ndarrays
            Data as np.ndarray(s) after feature selection

        Notes
        -----------
            Most of previous functionality was replaced by models.feature_selection.ValueThreshold.
            Here apply only selection based on feature names.

        See Also:  models.feature_selection.ValueThreshold
        """
        if not isinstance(result, (list, tuple, )): result = [result]
        assert len(set(map(len, result))) in (0, 1) # all same length

        feature_names = self._get_feature_names()
        if self.feature_blacklist is not None:
            drop_id_name = [True if x.startswith(self.feature_blacklist) else False \
                            for x in feature_names]
        else:
            drop_id_name = [False]*result[0].shape[1]

        if not hasattr(self, "feature_drop_id"):
            # This functionality was SUPERSEDED by custom FeatureSelector
            drop_id = [False]*len(drop_id_name) # mock
            self.feature_drop_id = np.logical_or(drop_id, drop_id_name)

        result_out = []
        for i,res in enumerate(result):
            result_out.append(res[:, np.invert(self.feature_drop_id)])

        if len(result_out) == 1: result_out = result_out[0]

        return(result_out)

    def load_sample_data(self, nsamples = -1, exps = None):
        """read data from jsons

        You can control how many samples or which experiments are retained
        with `nsamples` and `exps`

        Returns
        -----------
        samples: array-like
            Names of the loaded samples.
        result: np.ndarray
            Samples x Features matrix, same orderd as samples.

        """
        start_time = timeit.default_timer()

        jpaths = self._glob_ext(exps = exps)
        if nsamples > 0: # take random subset of samples
            if nsamples < len(jpaths): # avoid shuffle otherwise
                jpaths = random.sample(jpaths, min(nsamples, len(jpaths)))

        if not isinstance(jpaths, (list)): jpaths = [jpaths]
        regx = re.compile(  "(.*)\\.(txt|gpr|cel)( ?|\\.proc)( ?|\\.gz|\\.zip)$",
                            flags=re.IGNORECASE)

        result = []
        samples = []
        for jpath in jpaths:
            with open(jpath, 'r') as f:
                data = json.load(f)

            flat_vals = get_values_from_dict(data)
            result.append(flat_vals)
            # Clean names from left-over extensions, if any
            sname = regx.sub("\\1", list(data.keys())[0])
            samples.append(sname)

        # pad to common length, if ever needed
        max_len = len(sorted(result, key=len, reverse=True)[0])
        if not len(set([len(x) for x in result])) == 1:
            raise MaQcException(message = "Feature vectors of differing length.")
        result = np.array([x+[None]*(max_len-len(x)) for x in result])

        # If some sample is duplicated in name, drop it
        dupdrop_id = np.array([False if samples.count(x) > 1 else True for x in samples])
        samples = np.asarray(samples)[dupdrop_id].tolist()
        result = result[dupdrop_id, :]
        result = self._feature_selector(result)

        end_time = timeit.default_timer()
        if self.verbose:
            print("Finished reading data in {:.3f} s.".format(end_time - start_time))

        return(samples, result)

    def _load_feature_names(self):
        """Load Names of Features from Nested Dictionary

        Names are determined by looking at the first file in the data directory.
        Additionaly, values are output to facilitate validation of approach.

        Returns
        -----------
        feat_names: list
            Feature Names
        feat_values: list
            Values of Features. Can use this to check that names loaded properly.
        """
        jpath = self._glob_ext()[0]
        if self.verbose > 1:
            print("Choosing sample {}".format(os.path.basename(jpath)))
        with open(jpath, 'r') as f:
            data = json.load(f)
            data = flatten_dict(data)

        feat_names = [k + "." + str(i) for k,v in data.items() for i in range(len(v))]
        feat_values = list(chain.from_iterable(data.values()))

        return feat_names, feat_values

    def interpret_features( self, support = None, importance = None,
                            get_all_names = False, use_untransformed_index = False):
        """
        What features are driving decision?

        Parameters:
        ---------------
        support: list of arrays
            Boolean identifiers of features that passed feature_selection step, for
            each instance of feature selector
        importance: array-like
            Imprtances of features as obtained from classifiers.
        get_all_names: bool
            Bypass feature selection and return information for all features in data.
        use_untransformed_index: bool
            use indexing as in raw data

        Returns:
        -------------------
        features: dict
            dictionary of (id,name) pairs for selected features.
        """
        # Look at one file and pull out hiearchy of keys
        feat_names, feat_vals = self._load_feature_names()

        if get_all_names: # bypass selections and return full names
            dropped_id = None
            support = None
        else:
            try:
                dropped_id = self.feature_drop_id
            except AttributeError as e:
                dropped_id = None

        # Recast to full-selector
        if dropped_id is None: dropped_id = [False]*len(feat_names)

        # IDs of selected columns and reindex for data that was subset
        # see also `_feature_selector`
        full_col_id = np.arange(0, len(dropped_id))
        col_id_sel = full_col_id[np.invert(dropped_id)]

        # Obtain IDx that keeps features preserved by feature selectors
        if support is None:
            support_all = [True]*len(col_id_sel)
        else:
            if len(support) > 1 and len(set(map(len, support))) != 1:
                # If supports are not same lenght, can happen when feat selection in two steps
                support.sort(key=len, reverse = True) # sort longest first
                id_kept_by_transformer = support.pop(0) # get and remove first
                # Apply first step of selection
                col_id_sel = col_id_sel[id_kept_by_transformer]

        # Logical OR on list of arrays
            support_all = np.logical_or.reduce(support)

        # Sum importances for each feature
        # Importance is 1D array with length (sum(support))
        if importance is not None:
            idxs = [] # IDx of feature
            imp_vals = {} # importance value
            for s in support: # Loop over supports for each feature selector
                cnt = 0
                for i, tf in enumerate(s): # Loop over elements of 1D array
                    if tf: # If feature has support == True
                        # Append importance for this particular feature
                        if i in imp_vals.keys():
                            imp_vals[i].append(importance[cnt])
                        else:
                            imp_vals[i] = [importance[cnt]]
                        cnt += 1 # == sum(support) @ the end
                        idxs.append(i)
            # Sum importance for each feature for all feature selection steps
            importance = np.asarray(list(map(sum, imp_vals.values())))

        # Feature names in data before passed to scaling and transforming
        feat_names_sel_raw = np.asarray([feat_names[x] for x in col_id_sel])
        # Indices retained after feature selection
        col_id_sel_reindex = np.arange(0, len(col_id_sel))
        col_id_sel = col_id_sel[support_all]
        col_id_sel_reindex = col_id_sel_reindex[support_all]
        # Feature names retained after feature_selection
        feat_names_sel = [feat_names[x] for x in col_id_sel]
        feat_vals_sel = [feat_vals[x] for x in col_id_sel]

        # Return index as corresponding to RAW data (after _feature_selector)
        if use_untransformed_index:
            col_id_sel = col_id_sel_reindex

        # Return featues optionally sorted by their (lump) importance
        if importance is not None:
            data = np.transpose(np.asarray([feat_names_sel, feat_vals_sel, importance]))
            features = pd.DataFrame(data,
                index = col_id_sel, columns=["name", "value", "importance (sum)"])
            features = features.sort_values("importance (sum)", ascending = False)
        else:
            data = np.transpose(np.asarray([feat_names_sel, feat_vals_sel]))
            features = pd.DataFrame(data,
                index = col_id_sel, columns=["name", "value"])

        if self.verbose < 2: # drop debug-level information
            features.drop(columns = "value", inplace=True)

        self.features = features

        # If the features are cocatenation of multiple feature extractors we can
        # get the names of the ones beyond the basic ones here.
        # Note that this will be nonsensical if feature have been transformed to
        # some new space and do not have their previous meening anymore.

        if  support is not None and \
            isinstance(support[0], (list, tuple, np.ndarray )) \
            and len(support) > 1:
            feat_names_sel = np.asarray(feat_names_sel)[idxs]

        # additionaly return features unsorted by importances
        return features, (np.asarray(feat_names_sel), feat_names_sel_raw)

    def _get_feature_names(self):
        """Gets feature names for later subsetting"""

        features, _ = self.interpret_features(get_all_names = True)
        feature_names = features["name"].values.tolist()
        return feature_names

class LabelReader(object):
    """Label Reader Class

    This class encapsulates the functionality of reading labels for samples from file.
    """
    def __init__(self, flabels = ".", verbose = True):
        self.flabels = flabels
        self.verbose = verbose

    @property
    def samples(self):
        """Array-like, samples to read labels for."""
        return self._samples

    @samples.setter
    def samples(self, value):
        assert isinstance(value, (list, ))
        self._samples = value

    @property
    def flabels(self):
        """Path to file containing labels"""
        return self._flabels

    @flabels.setter
    def flabels(self, value):
        value = value.rstrip("/")
        value = os.path.expanduser(value)
        assert os.path.isfile(value), "File {} doesn't exist".format(value)
        self._flabels = value

    def _subset_data(self, y, nkeep = -1):
        """
        Subset data, keeping all negative samples and as much of others to get `nkeep` in total.
        """
        if nkeep > -1 and nkeep < len(y):
            keep_id = np.where(y.values == 0, True, False)
            nsub = max(0, nkeep-sum(keep_id))
            keep_id[[i for i,x in enumerate(keep_id) if not x][0:nsub]] = True
            y = y[keep_id]

        return y

    def read_labels(self, samples, delimiter = "\t", nkeep = -1):
        """Read sample labels

        Our approach allows for sample naming convention to be either <sample> or
        <exp>_<sample> where <exp> must be '[A-Z]{2}-[0-9]{5}'.

        Returns
        -----------
        y: pandas.Series
            Labels for samples
        data_keep_id: np.ndarray
            Indexer for subsetting data for samples that were discarded.
        """
        self.samples = samples
        samples_set = set(self.samples) # set of sample names, may or may not be prefixed
        # look at couple of files and decide: is the name in format <exp>_<sample>?
        is_prefixed = all(map( lambda x: re.match("[A-Z]{2}-[0-9]{5}_.*",x),
                                list(samples_set)[1:5]))

        labels = {}
        start_time = timeit.default_timer()
        with open(self.flabels, 'r') as rf:
            rf = csv.reader(rf, delimiter = delimiter)

            for i, row in enumerate(rf):
                sname = "_".join(row[0:2]) # prefixed name

                # Check if this label is in set of our samples.
                sname_check = "_".join(row[0:2]) if is_prefixed else row[1]
                if sname_check not in samples_set: continue

                    # This solves namin incesistency but is slow, see 3.1 in labels.ipynb
                    # It occurs rarely anyway, so we dont use it.
                    # match = difflib.get_close_matches(sname, self.samples, n = 1, cutoff = 0.6)
                    # if match:
                    #     str_diff = ""
                    #     for d in difflib.ndiff(sname,match[0]):
                    #         if d.startswith(("+", "-")): str_diff += d[2]
                    #     if re.search("(cel|gz|zip|txt|gpr)", str_diff, re.IGNORECASE):
                    #         sname = match[0]
                    # else:
                    #     continue # we do not care about this sample

                # Add to dict
                if row[0] not in labels.keys():
                    labels[row[0]] = {"samples": [],"labels": []}
                labels[row[0]]["samples"].append(sname)
                labels[row[0]]["labels"].append(row[2])

        # Pull out labels and sample names, see if some information missing
        samp_labs = list(chain.from_iterable([map(int, x["labels"]) for x in labels.values()]))
        samples_ = list(chain.from_iterable([x["samples"] for x in labels.values()]))
        missing_samples = set.difference(samples_set, set(samples_))

        if is_prefixed:
            samples_clean = samples_
        else: # if you expect <sample> as name, adjust here
            samples_clean = list(map(lambda x: re.sub("[A-Z]{2}-[0-9]{5}_(.*)", "\\1", x),
                                    samples_))
        idxer = [samples_clean.index(x) for x in self.samples if x in samples_clean]

        # Get unique sample labeling
        series_data = [(samples_[i], samp_labs[i]) for i in idxer]
        # may not be best in terms of performance but is simple and readable
        series_data = sorted(set(series_data), key = lambda x: series_data.index(x))
        y = pd.Series(  data = [x[1] for x in series_data],
                        index = [x[0] for x in series_data])
        # If index appears multiple times, the labels are ambiguous
        dups_id = y.index.duplicated(keep = False)
        y = y[np.invert(dups_id)]

        # Limit samples if desired
        y = self._subset_data(y, nkeep)
        selected_samples = y.index.tolist()


        if is_prefixed:
            selected_samples_clean = selected_samples
        else: # if you expect <sample> as name, adjust here
            selected_samples_clean = list(map(lambda x: re.sub("[A-Z]{2}-[0-9]{5}_(.*)", "\\1", x),
                                    selected_samples))
        # self.samples cannot be set because unordered!
        data_keep_id = [True if s in selected_samples_clean else False for s in self.samples]

        assert len(y) == sum(data_keep_id), "Number of labels not corresponding to number of datapoints"

        if self.verbose: # print out sutff
            samp_labs = y.values.tolist()
            print(  "{} samples do not have known labels (example: {})".\
                    format(len(missing_samples), list(missing_samples)[0:3]))

            print(  "Class membership: 0: {0}, 1: {1}".\
                    format(samp_labs.count(0), samp_labs.count(1)))

        if self.verbose:
            end_time = timeit.default_timer()
            print("Finished reading labels in {:.3f} s.".format(end_time - start_time))

        return y, np.asarray(data_keep_id)

class Reader(object):
    """
    Reader is convenience wrapper combining functionality of reading samples and their labels

    Currently supports reading from single location and single file with labels.
    Assumption on naming convention of files is that each is
    <neb_exp_id>_<sample_name><fext>, which allows for quick association of samples
    with experiments.

    Parameters:
        load_dir: str, directory containing both train and test data [default = "."]
        fext: str, extension of data files [default=".json"]
        flabels: str, valid path to a file with sample labels
        verbose: int or bool, verbosity parameter
        random_state: int or None, state of random number generator

    TODO:
        extend to multiple locations and files with labels.

    See also:
        DataReader, LabelReader
    """
    def __init__(   self, load_dir = ".", fext = ".json", flabels = ".",
                    feature_blacklist = None, verbose = True, random_state = None):
        self.load_dir = load_dir
        self.fext = fext
        self.flabels = flabels
        self.random_state = random_state
        self.verbose = verbose
        self.feature_blacklist = feature_blacklist

    @property
    def load_dir(self):
        return self._load_dir

    @load_dir.setter
    def load_dir(self, value):
        assert isinstance(value, (list, tuple, str, ))
        if isinstance(value, (list, tuple, )) and len(value) == 1:
            value = value[0]
            assert isinstance(value, (str, ))
        self._load_dir = value

    @property
    def flabels(self):
        return self._flabels

    @flabels.setter
    def flabels(self, value):
        assert isinstance(value, (list, tuple, str, ))
        if isinstance(value, (list, tuple, )) and len(value) == 1:
            value = value[0]
            assert isinstance(value, (str, ))
            assert os.path.isfile(value)
        elif isinstance(value, (list, tuple, )):
            assert all([os.path.isfile(f) for f in value])
            assert len(value) == len(self.load_dir)
        self._flabels = value

    def load_train_test_data(self, nsamples = -1 , test_size = 0.25, train_exps = None):
        """Read train and test data assuring disjoint experiments

        To remove confounding effects, samples from any given experiment must be
        exclusively in either train or test set (or excluded alltogether).

        Parameters:
        ------------
        nsamples: int [default = -1]
            Number of samples to load. Default loads all experiments in folder.

        test_size: float [default = -1]
            Fraction of experiments for test set. Note that fraction of samples may
            be different due to differing number of samples in experiments.

        Returns:
        ----------
        train: tuple
            (train_samples, train_data) where former is pandas.Series and latter np.ndarray
        test: tuple
            (test_samples, test_data) where former is pandas.Series and latter np.ndarray
        """
        if isinstance(self.load_dir, (list, tuple, )) and len(self.load_dir) > 1:
            raise NotImplementedError
            # train, test = self._load_from_multiple(nsamples, test_size)
        else:
            train, test = self._load_from_single(nsamples, test_size, train_exps)

        return train, test

    def _load_from_multiple(self, nsamples = -1, test_size = 0.25):
        """load from multiple locations"""
        pass

    def _load_from_single(self, nsamples = -1, test_size = 0.25, train_exps = None):
        """load from single location

        Parameters
        ----------
        train_exps: array-like, optional
            List of experiment names to be used for training
        n_samples: int, optional
            Number of samples to load in total for both train and test.
        test_size: float, [default = 0.25]
            Size of test set.
        """
        # DataReader Object
        self.data_reader = DataReader(  self.load_dir, self.fext, self.feature_blacklist,
                                        self.verbose)
        # get all ''*.fext' in load_dir
        jpaths = self.data_reader._glob_ext()
        # Pull out experiment names from filename
        reg = re.compile("(^[A-Z]{2}-[0-9]{5})")
        all_exps = set([reg.search(os.path.basename(x))[0] for x in jpaths])

        assert test_size > 0. and test_size < 1., "test_size must be fraction 1..0"
        # Split experiments
        train_exps_, test_exps = train_test_split(
            list(all_exps), test_size = test_size,
            random_state = self.random_state, shuffle = True,
        )
        if train_exps is None or len(train_exps) == 0: train_exps = train_exps_

        if not set.isdisjoint(set(train_exps), set(test_exps)):
            test_exps = [t for t in test_exps if t not in train_exps]

        # Make sure we are loading correct proportions
        if nsamples > 0:
            n_train = int(nsamples * (1 - test_size))
            n_test = int(nsamples * test_size)
        else:
            n_train = n_test = nsamples

        # Load training and test set
        samples_train, X_train = self.data_reader.load_sample_data(n_train, train_exps)
        samples_test, X_test = self.data_reader.load_sample_data(n_test, test_exps)
        assert set.isdisjoint(set(samples_train), set(samples_test))

        # Load Labels for train and test set
        self.label_reader = LabelReader(flabels = self.flabels, verbose = self.verbose)
        y_train, keepid = self.label_reader.read_labels(samples_train)
        X_train = X_train[keepid, :] ## subsetting as some labels ambiguous
        y_test, keepid = self.label_reader.read_labels(samples_test)
        X_test = X_test[keepid, :] ## subsetting as some labels ambiguous

        return (X_train, y_train), (X_test, y_test)
