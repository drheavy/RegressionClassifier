import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor

from sklearn.linear_model import LogisticRegression

from .utils import bins_calc


class ClassRegressor():
    """Модель, обучающая классификатор по заданным границам таргета"""
    def __init__(self, n_bins=2, bins_calc_method='equal', leaf_model_cls=DummyRegressor):
        """
        Инициализация
        n_bins - количество бинов, на которые делятся данные на каждом уровне
        bins_calc_method - метод разделения таргет-переменной на бины ('equal', 'percentile')
        leaf_model_cls - модель регрессии на листовых бинах
        """
        self.n_bins = n_bins
        self.bins_calc_method = bins_calc_method
        self.leaf_model_cls = leaf_model_cls

        self.bin_borders = None
        self.leaf_model_ex = {}

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def fit(self, X, y):
        """
        Обучение модели
        X - таблица с входными данными
        y - столбец с таргет-переменной
        """

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        X = np.array(X)
        y = np.array(y)

        bin_edges = bins_calc(y, n_bins=self.n_bins, method=self.bins_calc_method)
        self.bin_borders = np.zeros((len(bin_edges) - 1, 2))

        for i in range(len(bin_edges) - 1):
            self.bin_borders[i] = np.array([bin_edges[i], bin_edges[i+1]])

        self.y_classes = pd.cut(y, bins=bin_edges, labels=False, include_lowest=True)
        for label, _ in enumerate(self.bin_borders):
            bin_y = y[self.y_classes == label]
            bin_X = X[self.y_classes == label]
            self.leaf_model_ex[label] = self.leaf_model_cls()
            self.leaf_model_ex[label].fit(bin_X, bin_y)

        if X.shape[1] > X.shape[0]:
            self.model = DummyClassifier(strategy='most_frequent')
        else:
            self.model = LogisticRegression(n_jobs=1)
        self.model.fit(X, self.y_classes)

        return self

    def predict(self, X, regression=False):
        """
        Предиктор
        X - таблица с входными данными
        """

        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.array(X)

        pred = self.model.predict(X)
        if regression:
            preds = np.zeros((len(X),))
            for pred_class in np.unique(pred):
                idx = np.array(range(len(X)))[pred==pred_class]
                class_X = X[idx]
                preds[idx] = self.leaf_model_ex[pred_class].predict(class_X)
            return preds
        return pred


class ClassRegressorOnelevel():
    """Модель, обучающая бинарный классификатор по заданной границе таргета"""

    def __init__(self, bin_edges, leaf_model_cls=None):
        """
        Инициализация
        bin_edges - граница для деления данных на 2 бина
        leaf_model_cls - модель регрессии на листовых бинах
        """
        self.bin_edges = bin_edges
        self.leaf_model_cls = leaf_model_cls

        self.bin_borders = {}
        self.bin_predictions = np.zeros((2, ))
        self.leaf_model_ex = {}

    def fit(self, X, y):
        """
        Обучение модели
        X - таблица с входными данными
        y - столбец с таргет-переменной
        """

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        X = np.array(X)
        y = np.array(y)

        for i in range(len(self.bin_edges) - 1):
            self.bin_borders[i] = np.array([self.bin_edges[i], self.bin_edges[i+1]])

        self.y_classes = np.digitize(y, self.bin_edges, right=True) - 1

        if not self.leaf_model_cls:
            self.bin_predictions[0] = self.bin_edges[1]
            self.bin_predictions[1] = self.bin_edges[1]
        else:
            for label in [0, 1]:
                bin_y = y[self.y_classes == label]
                bin_X = X[self.y_classes == label]
                self.leaf_model_ex[label] = self.leaf_model_cls()
                self.leaf_model_ex[label].fit(bin_X, bin_y)

        if X.shape[1] > X.shape[0]:
            self.model = DummyClassifier(strategy='most_frequent')
        else:
            self.model = LogisticRegression(n_jobs=1)

        self.model.fit(X, self.y_classes)

        return self

    def predict(self, X, regression=False):
        """
        Предиктор
        X - таблица с входными данными
        """

        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.array(X)

        pred = self.model.predict(X)

        if regression:
            if not self.leaf_model_cls:
                pred = self.bin_predictions[pred]
            else:
                pred = [self.leaf_model_ex[p].predict(X[i].reshape(1, -1)) for i, p in enumerate(pred)]

        return pred
