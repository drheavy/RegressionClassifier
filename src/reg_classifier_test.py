import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn import metrics


class ClassRegressor():
    """Модель, делающая разбиение на бины одного уровня"""

    def __init__(self, n_bins=100, bins_calc_method='equal', perc=50, full_mode=True):
        """
        Инициализация
        n_bins - количество бинов, на которые делятся данные на каждом уровне
        bins_calc_method - метод разделения таргет-переменной на бины ('equal', 'percentile')
        """
        self.n_bins = n_bins
        self.bins_calc_method = bins_calc_method

        # Словарь соответствия новых классов с соответствующими диапазонами таргета
        self.bin_borders = {}

        self.perc = perc
        self.full_mode = full_mode

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

        bin_borders = bins_calc(y, n_bins=self.n_bins, method=self.bins_calc_method, perc=self.perc, full_mode=self.full_mode)
        if bin_borders is False:
            raise Exception('Unknown bins_calc_method')

        self.bin_borders = {i: [bin_borders[i], bin_borders[i+1]] for i in range(len(bin_borders)-1)}

        # Hack for np.digitize
        # to make sure the values that have exactly the same value as the left bin corner are included in the first bin
        # bin_borders[0] = bin_borders[0] - 1e-10
        # self.y_classes = np.digitize(y, bin_borders, right=True) - 1

        # ↑↑↑ - Inaccurate calculation of bins borders for level >= 2
        self.y_classes = pd.cut(y, bins=self.n_bins, labels=False, include_lowest=True)

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

    def __init__(self, n_bins=100, bins_calc_method='equal', leaf_model=None, full_mode=True):
        """
        Инициализация
        n_bins - количество бинов, на которые делятся данные на каждом уровне
        n_levels - количество уровней деления
        bins_calc_method - метод разделения таргет-переменной на бины ('equal', 'percentile')
        leaf_size - минимальный размер листового (неделимого) бина
        leaf_model - модель регрессора для предсказаний на листовых бинах
        """
        self.n_bins = n_bins
        self.bins_calc_method = bins_calc_method
        self.leaf_model = leaf_model
        self.full_mode = full_mode

        self.models = {}
        self.models_reg = {}

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

        self.step = int(100/self.n_bins)
        for bin_border in range(self.step, 100, self.step):
            model = ClassRegressor(n_bins=self.n_bins, bins_calc_method=self.bins_calc_method, perc=bin_border, full_mode=self.full_mode)
            model.fit(X, y)
            self.models[bin_border] = model

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        pred = np.empty((X.shape[0], ))

        for i, x in enumerate(X):
            # cur_level = 0
            cur_bin = 50
            clf = None

            clf = self.models[cur_bin]
            start_class = clf.predict([x])[0]
            #start_class = class_recal(start_class)

            if start_class == 0:
                bins_range = list(range(cur_bin, 0, -self.step))
            elif start_class == 1:
                bins_range = list(range(cur_bin, 100, self.step))
            else:
                raise Exception('Bin error')

            prev_class = start_class
            prev_clf = clf
            found_flag = False
            for cur_bin in bins_range[1:]:
                clf = self.models[cur_bin]
                cur_class = clf.predict([x])[0]
                #cur_class = class_recal(cur_class)

                if cur_class != prev_class:
                    found_flag = True
                    break
                prev_class = cur_class
                prev_clf = clf

            if found_flag:
                if cur_bin < 50:
                    # print(clf.bin_borders[1], prev_clf.bin_borders[0])
                    pred[i] = np.mean([clf.bin_borders[0][1], prev_clf.bin_borders[0][1]])
                else:
                    pred[i] = np.mean([clf.bin_borders[1][0], prev_clf.bin_borders[1][0]])
            else:
                if cur_bin < 50:
                    pred[i] = np.mean(clf.bin_borders[0])
                else:
                    pred[i] = np.mean(clf.bin_borders[1])

        return pred


def bins_calc(y, n_bins=100, method='equal', perc=50, full_mode=True):
    """
    Вычисление границ бинов разными методами
    y - столбец с таргет-переменной
    n_bins - количество бинов, на которые делятся данные на каждом уровне
    method - метод разделения таргет-переменной на бины ('equal', 'percentile')
    """

    if isinstance(y, pd.Series):
        y = y.values

    if method == 'percentile':
        bin_borders = pd.qcut(y, q=n_bins, labels=False, retbins=True, duplicates='drop')[1]
        if len(bin_borders) < 3:
            bin_borders = bins_calc(y, n_bins=n_bins, method='equal')
    elif method == 'equal':
        bin_borders = np.histogram(y, bins=n_bins)[1]
    else:
        return False

    idx_bin = int(perc / int(100/n_bins))
    # print(f"n_bins = {n_bins}, perc = {perc}, idx_bin = {idx_bin}")
    if full_mode:
        bin_borders = [bin_borders[0], bin_borders[idx_bin], bin_borders[-1]]
    else:
        bin_borders = [bin_borders[idx_bin-1], bin_borders[idx_bin], bin_borders[idx_bin+1]]

    return bin_borders


def class_recal(cl):
    if cl > 1:
        return 1
    elif cl < 0:
        return 0
    else:
        return cl


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
