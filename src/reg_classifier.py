import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn import metrics


class ClassRegressor():
    """Модель, делающая разбиение на бины одного уровня"""

    def __init__(self, n_bins=2, bins_calc_method='equal'):
        """
        Инициализация
        n_bins - количество бинов, на которые делятся данные на каждом уровне
        bins_calc_method - метод разделения таргет-переменной на бины ('equal', 'percentile')
        """
        self.n_bins = n_bins
        self.bins_calc_method = bins_calc_method

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

        bin_borders = bins_calc(y, n_bins=self.n_bins, method=self.bins_calc_method)
        if bin_borders is False:
            raise Exception('Unknown bins_calc_method')

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

        # Если достигнут листовой бин
        if level >= self.n_levels or len(y) < self.leaf_size or min(y) == max(y):
            if self.leaf_model:
                model_reg = self.leaf_model()
                model_reg.fit(X, y)
                self.models_reg[(level, bin_index_tuple)] = model_reg
            return

        model = ClassRegressor(n_bins=self.n_bins, bins_calc_method=self.bins_calc_method)
        model.fit(X, y)
        self.models[(level, bin_index_tuple)] = model

        for i, (bin_class, bin_border) in enumerate(model.bin_borders.items()):
            bin_border[0] = bin_border[0] - 1e-10
            bin_idx = (y > bin_border[0]) & (y <= bin_border[1])

            X_subset, y_subset = X[bin_idx], y[bin_idx]
            if len(y_subset) == 0:
                continue

            self._fit_recur(
                X_subset,
                y_subset,
                level=level+1,
                bin_index=bin_index_tuple + (i,),
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

        self._fit_recur(X, y, 0, [0])

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        pred = np.empty((X.shape[0], ))
        for i, x in enumerate(X):
            cur_level = 0
            cur_bin = tuple([0])
            clf = None

            while cur_level <= self.n_levels:
                if (cur_level, cur_bin) in self.models:
                    clf = self.models[(cur_level, cur_bin)]
                    predicted_class = clf.predict([x])[0]

                    cur_level += 1
                    cur_bin += (predicted_class,)
                else:
                    if self.leaf_model and (cur_level, cur_bin) in self.models_reg:
                        pred[i] = self.models_reg[(cur_level, cur_bin)].predict([x])[0]
                    else:
                        pred[i] = np.mean(clf.bin_borders[cur_bin[-1]])
                    break

        return pred


class ClassRegressorEnsembleLog():
    """Внешний класс для ClassRegressorEnsemble, преобразующий таргет в симметричное распределение и обратно"""

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

    def fit(self, X, y):
        # В данном методе вычисляется сумма левой и правой частей гистограммы, после чего для таргет-переменной
        #       применяется логарифмическое или экспоненциальное преобразование
        y_mid = (y.max() - y.min()) / 2 + y.min()

        left_sum = len(y[y <= y_mid])
        right_sum = len(y[y > y_mid])

        SUMS_DIFF_MAX = 1.3

        self.class_reg_ens = ClassRegressorEnsemble(n_bins=self.n_bins, n_levels=self.n_levels,
                                                    bins_calc_method=self.bins_calc_method, leaf_size=self.leaf_size,
                                                    leaf_model=self.leaf_model)

        if right_sum == 0 or left_sum / right_sum > SUMS_DIFF_MAX:
            self.log_exp = 'log'

            # Добавить проверку на нули в таргете
            self.class_reg_ens.fit(X, np.log(y))

        elif left_sum == 0 or right_sum / left_sum > SUMS_DIFF_MAX:
            self.log_exp = 'exp'

            self.class_reg_ens.fit(X, np.exp(y))

        else:
            self.log_exp = 'norm'

            self.class_reg_ens.fit(X, y)

    def predict(self, X):
        # Преобразование таргет переменной, обратное тому, которое было выполнено в методе fit
        if self.log_exp == 'log':
            pred = self.class_reg_ens.predict(X)
            return np.exp(pred)

        elif self.log_exp == 'exp':
            pred = self.class_reg_ens.predict(X)
            # Добавить проверку на нули в предиктах
            return np.log(pred)

        else:
            pred = self.class_reg_ens.predict(X)
            return pred

def bins_calc(y, n_bins=2, method='equal'):
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

    return bin_borders


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
