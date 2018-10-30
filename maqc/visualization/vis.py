import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import pandas as pd
import numpy as np
import random
import os

from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection as GRP
from sklearn.metrics import (   roc_curve, auc, average_precision_score,
                                precision_recall_curve, f1_score,
                                classification_report, confusion_matrix,
                                precision_recall_fscore_support)

from maqc.utils import adjusted_classes, balanced_accuracy_score


class Visualizer(object):
    """Some Visualizations for maqc ml pipeline

    Usually you need y_true, y_predict_proba to use plots below
    """
    def __init__(self):
        # Init
        self.confusion_matrix = pd.DataFrame()
        self.report = {}

    def dim_reduce(self, X, ncomp = 4, what = "pca", scale = True):
        """Perform dimensionality reduction on the input data in X

        Parameters
        ----------
        ncomp: int
            Number of dimensions to retain after dim reduction.
        what: str
            Codenane for dimensionality reduction method
        scale: bool
            Set to true if data is was not scaled (e.g. raw).
        """
        if scale: X = StandardScaler().fit_transform(X)
        what = str.lower(what)

        if what == "pca":
            pca = PCA(n_components = ncomp, whiten= False)
            Xout = pca.fit_transform(X)
            print("Explained variance ratio: {}".format(pca.explained_variance_ratio_))

        elif what == "ica":
            ica = FastICA(n_components = ncomp, max_iter = 10000)
            Xout = ica.fit_transform(X)

        elif what == "random":
            grp = GRP(n_components = ncomp)
            Xout = grp.fit_transform(X)

        elif what == "kpca":
            kpca = KernelPCA(n_components = ncomp, kernel = "rbf", max_iter = 10000)
            Xout = kpca.fit_transform(X)

        return (Xout)


    def plot_roc_curve(self, y_true, y_scores, class_id = 1):
        """Plotting ROC

        Implemented for binary classification with classes [0, 1]

        Parameters
        -----------------
        y_true: array-like
            1D array-like with true labels
        y_scores: array-like
            nD array-like with probabilities for n classes

        See also:
            sklearn.metrics.{roc_curve,auc,average_precision_score}
        """
        if y_scores.ndim ==1: y_scores = y_scores.reshape((-1, 1))
        n_cl = y_scores.shape[1]
        if n_cl != 2: raise NotImplementedError
        counts = np.bincount(y_true) # How many occurences of unique?

        fig, axs = plt.subplots(1, n_cl, figsize = (12, 6), frameon = False)

        for cl_id, ax in enumerate(axs.flat):
            y_score = y_scores[:, cl_id]
            num = counts[cl_id]

            fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=cl_id)
            auc_val = auc(fpr, tpr)
            avg_precision = average_precision_score(y_true, y_score, average=None)

            ax.plot(fpr, tpr, color = "darkorange", lw = 2)
            ax.plot([0,1], [0, 1], color = "navy", lw = 1, linestyle = "--")
            ax.legend( ["ROC Curve (AUC = {:.2f})".format(auc_val), "Baseline"],
                        loc="lower right")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.0])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            title = "ROC Curve for {:d} datapoints (class {:d})".format(num, cl_id)
            ax.set_title(title)

            avpr = "Average PR score: {:.2f}".format(avg_precision)
            ax.text(.5, .15, avpr, transform = ax.transAxes)


    def plot_precision_recall(self, y_true, y_scores, thresh = .5):
        """ Plot Precision-Recall Tradeoffs

        Allows to inspect precision/recall tradeoff at given decision threshold.

        See also:
            sklearn.metrics.{precision_recall_curve,average_precision_score}
        """
        if y_scores.ndim ==1: y_scores = y_scores.reshape((-1, 1))
        n_cl = y_scores.shape[1]
        if n_cl != 2: raise NotImplementedError
        counts = np.bincount(y_true)

        fig, axs = plt.subplots(nrows = n_cl, ncols = 2, figsize = (12, 6))

        for cl_id, ax_row in enumerate(axs):
            y_score = y_scores[:, cl_id]
            num = counts[cl_id]
            ax = ax_row[0]; bx = ax_row[1]

            prec, recall, threshs = precision_recall_curve(y_true, y_score, pos_label=cl_id)
            avg_precision = average_precision_score(y_true, y_score, average=None)

            # Plotting ---------------------------------------------------------

            ## A
            ax.step(recall, prec, color = "b", alpha = 0.2, where = "post")
            ax.fill_between(recall, prec, alpha = 0.2, color = "b", step = "post")

            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_ylim([0.0, 1.05])
            ax.set_xlim([0.0, 1.0])
            ax.set_title("PR Curve, AverageP = {:.2f}".format(avg_precision))

            # Identify P/R tradeoff for current threshold
            id_nearest = np.argmin(np.abs(threshs - thresh))
            ax.plot(recall[id_nearest], prec[id_nearest]-.01, "^", c = "k", markersize = 15)
            ax.axvline(recall[id_nearest], alpha = 0.3, c = "red", ls = ":")
            ax.axhline(prec[id_nearest], alpha = 0.3, c = "red", ls = ":")

            ## B
            bx.plot(threshs, prec[:-1], "b--", label="Precision")
            bx.plot(threshs, recall[:-1], "g-", label="Recall")
            bx.set_ylim([0.0, 1.05])
            # bx.set_xlim([0.0, 1.0])
            bx.set_title("P and R Scores (class {}, #{})".format(cl_id, counts[cl_id]))
            bx.set_ylabel("Score")
            bx.set_xlabel("Decision Threshold")
            bx.legend(loc="center left")
            bx.plot(thresh, 0.0, "^", c = "k", markersize = 15)
            bx.axvline(thresh, alpha = 0.3, c = "red", ls = ":")

            # msg = "P = {:.2f},\nR = {:.2f}.".format(
            #     thresh, prec[id_nearest], recall[id_nearest])
            # bx.text(.25, .15, msg, transform = bx.transAxes)

        fig.tight_layout()

    def report_metrics(self, y_true, y_scores, thresh = 0.5):
        """
        Report classification metrics for the `1` class.

        Note: implementation is for binary classification task.
        """
        y_score = y_scores[:, 1]
        y_pred_adj = adjusted_classes(y_score, thresh)
        _, _, fbeta_adj, support_adj = precision_recall_fscore_support(
            y_true, y_pred_adj, warn_for = ("fscore", ))

        print("Classification metrics at adjusted thresh = {}:".format(thresh))
        print("Score: {}".format(fbeta_adj))
        self._report_metrics(y_true, y_pred_adj)

        y_pred = adjusted_classes(y_score)
        _, _, fbeta, support = precision_recall_fscore_support(
            y_true, y_pred, warn_for = ("fscore", ))

        print("\nClassification metrics at default thresh = {}:".format(0.5))
        print("Score: {}".format(fbeta))
        self._report_metrics(y_true, y_pred)

        return (fbeta_adj, fbeta)

    def _report_metrics(self, y_true, y_hat):
        """Print out confusion matrix, classification report and balanced_accuracy_score

        Parameters
        ----------
        y_true: array-like, true labels
        y_hat: array-like, predicted (binary) labels

        See also:
            sklearn.metrics.{confusion_matrix,classification_report,balanced_accuracy_score}
        """

        self.confusion_matrix =pd.DataFrame(confusion_matrix(y_true, y_hat),
                                            columns=['pred_neg', 'pred_pos'], index=['neg', 'pos'])
        self.report = classification_report(y_true, y_hat)
        self.balanced_accuracy = balanced_accuracy_score(y_true, y_hat)

        print("\nConfusion matrix:\n", self.confusion_matrix)
        print("\nClassification report: \n", self.report)
        print("\nBalanced Acc. Score: \n", self.balanced_accuracy)

    def plot_leading_dims(  self, Xs, ys, which = None, support = None, nsamp = 50,
                            dim_names = None):
        """Do pair plot of leading dimensions for dataset

        Parameters
        ----------
        Xs: list
            Data to visualize, samples x features, possibly mutliple as list of arrays
        ys: list
            List of array-like, labels for samples in Xs
        which: string or array-like
            Which features to visualize, either as ids or as name of dimensionality reduction.
        nsamp: int
            How many samples to render per class.
        support: array-like of bool
            T/F for which features are kept by the pipeline
        dim_names: array-like of str
            Names of features in Xs,

        See also:
            dim_reduce

        """
        if not isinstance(Xs, (list, tuple, )): Xs = [Xs]
        if not isinstance(ys, (list, tuple, )): ys = [ys]
        assert len(Xs) == len(ys)

        X_vis = []
        ident = []
        for i, (X, y) in enumerate(zip(Xs, ys)):
            if hasattr(y, "values"): y = y.values # pull from pandas.Series

            for cl in np.unique(y):
                temp = X[y==cl, :] # select given class
                idx = np.random.randint(0, temp.shape[0], size = min(temp.shape[0], nsamp))
                temp = temp[idx, :]
                X_vis.append(temp)
                ident.append("X_{} (class {})".format(i, cl))

        counts = [len(x) for x in X_vis]
        X_vis = np.concatenate(X_vis)
        y_vis = np.concatenate([np.repeat([i], c) for i,c in zip(ident, counts)])

        # Keep only features retained by feature selection estimators
        if support is not None:
            X_vis = X_vis[:, support]

        # Select feaetures to visualize
        if isinstance(which, (tuple, list, np.ndarray, )):
            X_vis = X_vis[:, which]
        elif isinstance(which, (str, )):
            X_vis = self.dim_reduce(X_vis, what = which, scale = False)
        elif which is None:
            pass
        else:
            NotImplementedError

        if dim_names is None:
            dim_names = ["dim_" + str(x+1) for x in range(X_vis.shape[1])]
        else:
            dim_names = dim_names[which]
        X_vis = pd.DataFrame(X_vis, columns = dim_names)
        X_vis["labels"] = y_vis

        sns.pairplot(
            X_vis, vars = dim_names, hue = "labels", plot_kws = {"alpha": .5},
            markers = ["D", "X", "s", "+"])

    def plot_features(self, features, X, labels):
        """ Plot each feature individually for each sample togetehr with QC decision

        This allows you to eye-ball informative features. So far it seems that
        for affy, the informative features are: "norm_rle, raw_rle, norm_corr,
        raw_corr, nuse, (hclust)".

        Paramters
        -------------
        features: list
            Feature Names.
        X: np.ndarray
            Matrix of data samples X features.
        labels: array-like
            Labels for samples in X.

        Returns
        -------------
        None. Renders 'plot_features.pdf' in cwd.
        """

        n_samples = X.shape[0]
        if n_samples > 5000 : # downsample
            idxs = random.sample(range(0, n_samples), k = 5000)
            X = X[idxs, :]
            labels = labels[idxs]
        n_feats = len(features)
        nrow = n_feats // 4
        x_ax = np.linspace(0, 1, X.shape[0])
    #     col_seq = ["r" if l == 0 else "g" for l in labels]
        lab_id = np.asarray([True if l == 1 else False for l in labels])

        fig, axs = plt.subplots(nrows = nrow + 1,  ncols = 4, squeeze = False,
                               figsize = (33.1, 46.8*3), dpi = 100) # 3 A0 vertically in portrait


        for i, ax in enumerate(axs.flatten()):
            ax.scatter(x_ax[lab_id], X[lab_id, i], c = "g")
            ax.scatter(x_ax[~lab_id], X[~lab_id, i], c = "r")

            ax.set_xticklabels([])
            ax.set_title(features[i])

            if i == X.shape[1] -1 :break

        plt.tight_layout()

        fout = os.path.join(os.getcwd(), 'plot_features.pdf')
        plt.savefig(fout)
        print("Saved plot to: {}.".format(fout))
