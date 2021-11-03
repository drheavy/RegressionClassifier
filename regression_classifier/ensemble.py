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
        # Cловарь соответствия пары уровень-класс и обученной модели классификатора
        # self.level_class_model_dict = {}

        self.models = {}
        self.models_reg = {}

    def _fit_recur(self, X, y, level, bin_index):

        bin_index_tuple = tuple(bin_index)

        if level >= self.n_levels or len(y) < self.leaf_size or min(y) == max(y):
            if self.leaf_model:
                model_reg = self.leaf_model()
                model_reg.fit(X, y)
                self.models_reg[(level, bin_index_tuple)] = model_reg
            return

        model = ClassRegressor(n_bins=self.n_bins, bins_calc_method=self.bins_calc_method)
        model.fit(X, y)
        # self.models[(level, bin_index, prev_model_key)] = model
        self.models[(level, bin_index_tuple)] = model

        # for i, (bin_class, bin_border) in enumerate(model.bin_borders.items()):
        for i, bin_border in enumerate(model.bin_borders):
            # bin_idx = (y >= bin_border[0]) & (y <= bin_border[1])
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
                    if self.leaf_model and (cur_level, cur_bin) in self.models_reg:
                        pred[i] = self.models_reg[(cur_level, cur_bin)].predict([x])[0]
                    else:
                        pred[i] = clf.predict([x], regression=True)[0]
                    break

        return pred


