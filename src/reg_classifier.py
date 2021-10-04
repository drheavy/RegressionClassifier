import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn import metrics


class ClassRegressor():
    """Модель, делающая разбиение на бины одного уровня"""

    def __init__(self, n_bins=2):
        """
        Инициализация
        n_bins - количество бинов, на которые делятся данные на каждом уровне
        """
        self.n_bins = n_bins

        # Словарь соответствия новых классов с соответствующими диапазонами таргета
        self.bin_borders = {}

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

        bin_borders = np.histogram(y, bins=self.n_bins)[1]

        self.bin_borders = {i: [bin_borders[i], bin_borders[i+1]] for i in range(len(bin_borders)-1)}

        # Hack for np.digitize
        # to make sure the values that have exactly the same value as the left bin corner are included in the first bin
        bin_borders[0] = bin_borders[0] - 1e-10 
        self.y_classes = np.digitize(y, bin_borders, right=True) - 1

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

        pred = self.model.predict(X)

        if regression:
            pred = np.mean([self.bin_borders[y_class] for y_class in pred], axis=1)

        return pred


class ClassRegressorEnsemble():
    """Комплексная модель с ансамблем одноуровневых моделей классификации"""

    def __init__(self, n_bins=2, n_levels=2):
        """
        Инициализация
        n_bins - количество бинов, на которые делятся данные на каждом уровне
        n_levels - количество уровней деления
        """
        self.n_bins = n_bins
        self.n_levels = n_levels
        # Cловарь соответствия пары уровень-класс и обученной модели классификатора
        self.level_class_model_dict = {}

        self.models = {}

    def _fit_recur(self, X, y, level, bin_index, prev_model_key):
        if level >= self.n_levels:
            return

        model = ClassRegressor(n_bins=self.n_bins)
        model.fit(X, y)
        self.models[(level, bin_index, prev_model_key)] = model

        for i, (bin_class, bin_border) in enumerate(model.bin_borders.items()):
            bin_idx = (y >= bin_border[0]) & (y <= bin_border[1])

            X_subset, y_subset = X[bin_idx], y[bin_idx]

            self._fit_recur(
                X_subset, 
                y_subset, 
                level=level+1, 
                bin_index=i,
                prev_model_key=(level, bin_index),
            )

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

        self._fit_recur(X, y, 0, 0, None)


    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        pred = np.empty((X.shape[0], ))
        for i, x in enumerate(X):
            prev_model_key = None
            cur_level = 0
            cur_bin = 0 
            clf = None

            while cur_level <= self.n_levels:
                if (cur_level, cur_bin, prev_model_key) in self.models:
                    clf = self.models[(cur_level, cur_bin, prev_model_key)]
                    predicted_class = clf.predict([x])[0]

                    prev_model_key = (cur_level, cur_bin)
                    cur_level += 1
                    cur_bin = predicted_class
                else:
                    pred[i] = np.mean(clf.bin_borders[cur_bin])
                    break

        return pred


# Функция вычисления метрики MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Функция для вычисления всех метрик
def dataframe_metrics(y_test, y_pred):
    stats = [
       metrics.mean_absolute_error(y_test, y_pred),
       np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
       metrics.r2_score(y_test, y_pred),
       mean_absolute_percentage_error(y_test, y_pred)
    ]
    return stats
