import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from .utils import bins_calc


class ClassRegressor:
    def __init__(self, n_bins=2, bins_calc_method='equal', leaf_model=None):
        """
        Инициализация
        n_bins - количество бинов, на которые делятся данные на каждом уровне
        bins_calc_method - метод разделения таргет-переменной на бины ('equal', 'percentile')
        leaf_model - модель регрессии на листовых бинах
        """
        self.n_bins = n_bins
        self.bins_calc_method = bins_calc_method
        self.leaf_model = leaf_model

        self.bin_borders = np.zeros((n_bins, 2))
        self.bin_predictions = np.zeros((n_bins, ))

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

        if self.leaf_model:
            self.model = self.leaf_model()
            self.model.fit(X, y)
        else:
            bin_edges = bins_calc(y, n_bins=self.n_bins, method=self.bins_calc_method)

            for i in range(len(bin_edges) - 1):
                self.bin_borders[i] = np.array([bin_edges[i], bin_edges[i+1]])

            self.y_classes = pd.cut(y, bins=bin_edges, labels=False, include_lowest=True)

            for label, _ in enumerate(self.bin_borders):
                bin_y = y[np.nonzero((self.y_classes == label).astype(int))]
                # bin_y = y[self.y_classes == label]
                self.bin_predictions[label] = np.mean(bin_y)

            self.model = LogisticRegression()
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

        if regression and not self.leaf_model:
            pred = self.bin_predictions[pred]

        return pred
