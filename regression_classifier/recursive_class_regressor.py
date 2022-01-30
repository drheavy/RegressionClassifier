import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.dummy import DummyClassifier, DummyRegressor

from sklearn.linear_model import LogisticRegression, LinearRegression
from .utils import bins_calc


class ClassRegressorSplit(BaseEstimator, ClassifierMixin):
    def __init__(self, n_bins=2, bins_calc_method='equal'):
        self.n_bins = n_bins
        self.bins_calc_method = bins_calc_method

        self.bin_borders = None
        self.bin_idx = None

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        X = np.array(X)
        y = np.array(y)

        bin_edges = bins_calc(y, n_bins=self.n_bins, method=self.bins_calc_method)
        self.y_classes = pd.cut(y, bins=bin_edges, labels=False, include_lowest=True)

        self.bin_borders = np.zeros((len(bin_edges) - 1, 2))
        for i in range(len(bin_edges) - 1):
            self.bin_borders[i] = np.array([bin_edges[i], bin_edges[i+1]])


        if X.shape[1] > X.shape[0]:
            self.model = DummyClassifier(strategy='most_frequent')
        else:
            self.model = LogisticRegression(n_jobs=1)
        self.model.fit(X, self.y_classes)
        return self

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.array(X)

        return self.model.predict_proba(X)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.array(X)

        return self.model.predict(X)


class ClassRegressorTree(BaseEstimator, RegressorMixin):
    def __init__(self,
                 n_bins=2,
                 n_splits=2,
                 bins_calc_method='equal',
                 leaf_model_cls_name='DummyRegressor',
                 leaf_model_kwargs={},
                 leaf_size=1,
                 level=0):
        self.n_bins = n_bins
        self.n_splits = n_splits
        self.bins_calc_method = bins_calc_method
        self.leaf_model_cls_name = leaf_model_cls_name
        self.leaf_model_kwargs = leaf_model_kwargs
        self.leaf_size = leaf_size

        self.level = level
        self.split = None
        self.child_models = {}
        self.bin_idx = None

    @property
    def leaf_model_cls(self):
        str_to_cls = {
            'DummyRegressor': DummyRegressor,
            'LinearRegression': LinearRegression,
        }
        return str_to_cls[self.leaf_model_cls_name]

    def get_child_model(self, X, y):
        y_uniq = len(np.unique(y))
        if (self.level >= self.n_splits) or (len(y) < self.leaf_size) or (y_uniq < self.n_bins) or (y_uniq < 2):
            return self.leaf_model_cls(**(self.leaf_model_kwargs))
        else:
            return ClassRegressorTree(
                level=self.level+1,
                n_bins=self.n_bins,
                n_splits=self.n_splits,
                bins_calc_method=self.bins_calc_method,
                leaf_model_cls_name=self.leaf_model_cls_name,
                leaf_size=self.leaf_size,
            )

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        X = np.array(X)
        y = np.array(y)

        split_model = ClassRegressorSplit(
            n_bins=self.n_bins,
            bins_calc_method=self.bins_calc_method,
        )
        split_model.fit(X, y)
        self.split = split_model

        for i, bin_border in enumerate(self.split.bin_borders):
            if i > 0:
                bin_idx = (y > bin_border[0]) & (y <= bin_border[1])
            else:
                bin_idx = (y >= bin_border[0]) & (y <= bin_border[1])

            X_subset, y_subset = X[bin_idx], y[bin_idx]
            if len(y_subset) == 0:
                continue

            child_model = self.get_child_model(X_subset, y_subset)
            child_model.fit(X_subset, y_subset)
            self.child_models[i] = child_model

    def predict(self, X, classification=False):

        proba = self.split.predict_proba(X)
        pred = np.argmax(proba, axis=1)
        if classification:
            return pred, proba

        preds = np.zeros((len(X),))
        for bin_i, child_model in self.child_models.items():
            child_prediction = child_model.predict(X)
            preds += proba[:, bin_i] * child_prediction
        return preds


class RecursiveClassRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 n_bins=2,
                 n_splits=2,
                 bins_calc_method='equal',
                 leaf_model_cls_name='DummyRegressor', # Have to make it a string or sklearn hparam opt doesnt work
                 leaf_model_kwargs={},
                 leaf_size=1,
     ):
        self.n_bins = n_bins
        self.n_splits = n_splits
        self.bins_calc_method = bins_calc_method
        self.leaf_model_cls_name = leaf_model_cls_name
        self.leaf_model_kwargs = leaf_model_kwargs
        self.leaf_size = leaf_size

        self.tree = ClassRegressorTree(
            n_bins=self.n_bins,
            n_splits=self.n_splits,
            bins_calc_method=self.bins_calc_method,
            leaf_model_cls_name=self.leaf_model_cls_name,
            leaf_model_kwargs=self.leaf_model_kwargs,
            leaf_size=self.leaf_size,
        )

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        X = np.array(X)
        y = np.array(y)

        self.tree.fit(X, y)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.array(X)

        return self.tree.predict(X)
