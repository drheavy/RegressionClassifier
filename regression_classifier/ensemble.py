import pandas as pd
import numpy as np
from .class_regressor import ClassRegressor
from sklearn import metrics


class ClassRegressorEnsemble:
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

    def _fit_recur(self, X, y, level, bin_index):

        bin_index_tuple = tuple(bin_index)

        if level >= self.n_levels:
            return

        model = ClassRegressor(n_bins=self.n_bins)
        model.fit(X, y)
        # self.models[(level, bin_index, prev_model_key)] = model
        self.models[(level, bin_index_tuple)] = model

        # for i, (bin_class, bin_border) in enumerate(model.bin_borders.items()):
        for i, bin_border in enumerate(model.bin_borders):
            bin_idx = (y >= bin_border[0]) & (y <= bin_border[1])
            X_subset, y_subset = X[bin_idx], y[bin_idx]

            self._fit_recur(
                X_subset, 
                y_subset, 
                level=level+1, 
                bin_index=bin_index_tuple + (i,),
                # prev_model_key=(level, bin_index),
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

        X = np.array(X)
        y = np.array(y)

        self._fit_recur(X, y, 0, [0])

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.array(X)

        pred = np.empty((X.shape[0], ))
        for i, x in enumerate(X):
            # prev_model_key = None
            cur_level = 0
            cur_bin = tuple([0])
            clf = None

            while cur_level <= self.n_levels:
                # if (cur_level, cur_bin, prev_model_key) in self.models:
                if (cur_level, cur_bin) in self.models:
                    clf = self.models[(cur_level, cur_bin)]
                    predicted_class = clf.predict([x])[0]

                    # prev_model_key = (cur_level, cur_bin)
                    cur_level += 1
                    cur_bin += (predicted_class,)
                else:
                    pred[i] = clf.predict([x], regression=True)[0]
                    break

        return pred


