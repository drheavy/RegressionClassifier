import numpy as np
import pandas as pd
from .class_regressor import ClassRegressor
from sklearn import metrics


class ClassRegressorEnsemble:
    """Комплексная модель с ансамблем одноуровневых моделей классификации"""

    def __init__(self, n_bins=2, n_levels=2, bins_calc_method='equal', leaf_size=1, leaf_model=None):
        """
        Инициализация
        n_bins - количество бинов, на которые делятся данные на каждом уровне
        n_levels - количество уровней деления
        bins_calc_method - метод разделения таргет-переменной на бины ('equal', 'percentile')
        leaf_size - минимальный размер листового (неделимого) бина
        leaf_model - модель регрессора для предсказаний на листовых бинах
        """
        self.n_bins = n_bins
        self.n_levels = n_levels
        self.bins_calc_method = bins_calc_method
        self.leaf_size = leaf_size
        self.leaf_model = leaf_model

        self.models = {}
        self.models_reg = {}

    def _fit_recur(self, X, y, level, bin_index):

        bin_index_tuple = tuple(bin_index)

        y_uniq = len(np.unique(y))
        leaf_model = None
        is_leaf = None

        if level >= self.n_levels or len(y) < self.leaf_size or y_uniq < self.n_bins:
            if self.leaf_model:
                leaf_model = self.leaf_model
                is_leaf = True
            else:
                return

        model = ClassRegressor(n_bins=self.n_bins, bins_calc_method=self.bins_calc_method, leaf_model=leaf_model)
        model.fit(X, y)
        self.models[(level, bin_index_tuple)] = model

        if is_leaf:
            return

        for i, bin_border in enumerate(model.bin_borders):
            if i > 0:
                bin_idx = (y > bin_border[0]) & (y <= bin_border[1])
            else:
                bin_idx = (y >= bin_border[0]) & (y <= bin_border[1])

            X_subset, y_subset = X[bin_idx], y[bin_idx]
            if len(y_subset) == 0:
                continue

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
            cur_level = 0
            cur_bin = tuple([0])
            clf = None

            while cur_level <= self.n_levels:
                if (cur_level, cur_bin) in self.models:
                    clf = self.models[(cur_level, cur_bin)]

                if (cur_level, cur_bin) in self.models and not self.models[(cur_level, cur_bin)].leaf_model:
                    predicted_class = clf.predict([x])[0]
                    cur_level += 1
                    cur_bin += (predicted_class,)
                else:
                    if (cur_level, cur_bin) in self.models or clf.leaf_model:
                        pred[i] = clf.predict([x])[0]
                    else:
                        pred[i] = clf.predict([x], regression=True)[0]
                    break

        return pred


