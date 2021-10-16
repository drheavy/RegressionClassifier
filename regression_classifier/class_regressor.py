import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression


class ClassRegressor:
    def __init__(self, n_bins=2):
        """
        Инициализация
        n_bins - количество бинов, на которые делятся данные на каждом уровне
        """
        self.n_bins = n_bins

        # Словарь соответствия новых классов с соответствующими диапазонами таргета
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

        bin_edges = np.histogram_bin_edges(y, bins=self.n_bins)

        for i in range(len(bin_edges) - 1):
            self.bin_borders[i] = np.array([bin_edges[i], bin_edges[i+1]])

        # Hack for np.digitize
        # to make sure the values that have exactly the same value as the left bin corner are included in the first bin
        bin_edges[0] = bin_edges[0] - 1e-10
        self.y_classes = np.digitize(y, bin_edges, right=True) - 1

        for label, _ in enumerate(self.bin_borders):
            bin_y = y[np.nonzero((self.y_classes == label).astype(int))]
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

        if regression:
            pred = self.bin_predictions[pred]

        return pred
