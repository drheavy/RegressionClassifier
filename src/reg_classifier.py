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

        self.model = LogisticRegression(class_weight='balanced')

        self.model.fit(X, self.y_classes)

        return self

    def predict(self, X, regression=False):
        """
        Предиктор
        X - таблица с входными данными
        """

        if isinstance(X, pd.DataFrame):
            X = X.values

        # Список предсказанных лейблов
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

    def fit(self, X, Y):
        """
        Обучение модели
        X - таблица с входными данными
        y - столбец с таргет-переменной
        """

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(Y, pd.Series):
            Y = Y.values

        cur_level = 0
        cur_class = 0
        # Переменная для хранения иерархической цепочки классов текущего бина
        cur_class_full = [cur_class]

        # Запуск рекурсивной функции для заполнения словаря
        self.level_class_model_dict = recur_func(X, Y, self.n_bins, self.n_levels, cur_level, cur_class_full,
                                                 self.level_class_model_dict)

    def predict(self, X):
        """
        Предиктор
        X - таблица с входными данными
        """

        if isinstance(X, pd.DataFrame):
            X = X.values

        # Список предиктов
        pred_list = []

        # Цикл перебора векторов входных данных
        for x in X:

            # Сброс переменных с классом рассматриваемого бина
            cur_class = 0
            cur_class_full = [cur_class]

            # Цикл обхода всех уровней
            for cur_level in range(self.n_levels):

                # Проверка существования ключа с парой уровень-класс в словаре
                if (cur_level, tuple(cur_class_full)) in self.level_class_model_dict.keys():
                    # Вытаскиваем очередную модель обученного ранее классификатора
                    cur_model = self.level_class_model_dict[(cur_level, tuple(cur_class_full))]
                    # Предсказываем класс следующего бина
                    cur_class = cur_model.predict(x.reshape(1, -1))[0]
                    # Вычисляем полный класс следующего бина
                    cur_class_full.append(cur_class)

            # Раскодировка таргета в исходный формат
            pred_borders = cur_model.y_classes_borders_dict[cur_class]

            # Добавляем в список среднее значение границ бина
            pred_list.append((pred_borders[0] + pred_borders[1]) / 2)

        return pred_list


def recur_func(X, Y, n_bins, n_levels, cur_level, cur_class_full, level_class_model_dict):
    """
    Основная расчётная функция с рекурсивным обходом всех бинов
    X - таблица с входными данными
    y - столбец с таргет-переменной
    n_bins - количество бинов, на которые делятся данные на каждом уровне
    n_levels - количество уровней деления
    cur_level - текущий уровень обхода
    cur_class_full - текущий класс бина
    level_class_model_dict - словарь соответствия пары уровень-класс и обученной модели классификатора
    """
    # Если в текущем бине осталось единственное значение таргета, дальнейшее деление отменяем
    if min(Y) == max(Y):
        return

    # Запускаем классификатор на текущем диапазоне Y
    class_reg = ClassRegressor(n_bins=n_bins)
    class_reg.fit(X, Y)

    # Сохраняем словарь с границами бинов в локальную переменную
    y_classes_borders_dict = class_reg.y_classes_borders_dict

    # Обновление словаря соответствия пары уровень-класс и обученной модели классификатора
    level_class_model_dict.update({(cur_level, tuple(cur_class_full)): class_reg})
    # Является ли текущий бин последним в данной группе
    is_last_border = False

    # Цикла с обходом бинов из нового набора
    for cur_class in y_classes_borders_dict.keys():

        # Список индексов значений Y, попадающих в новый бин
        idx_list = []

        min_border = y_classes_borders_dict[cur_class][0]
        max_border = y_classes_borders_dict[cur_class][1]

        # Проверка, является ли текущий бин последним
        if cur_class == list(y_classes_borders_dict.keys())[-1]:
            is_last_border = True

        for idx, y in enumerate(Y):
            # Вторая часть условия - если последний бин не достигнут, то его правая граница не включается, и наоборот
            if (y >= min_border) and ((not is_last_border and y < max_border) or (is_last_border and y <= max_border)):
                idx_list.append(idx)

        # Вырезаем новый бин
        cur_bin_X = X[idx_list]
        cur_bin_Y = Y[idx_list]

        # Переменная для передачи класса нового бина в рекурсивную функцию
        cur_class_full_temp = cur_class_full.copy()
        cur_class_full_temp.append(cur_class)

        # Если не достигнут последний уровень, снова запускаем рекурсивную функцию
        if cur_level < n_levels:
            recur_func(cur_bin_X, cur_bin_Y, n_bins, n_levels, cur_level + 1, cur_class_full_temp,
                       level_class_model_dict)

    return level_class_model_dict


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
