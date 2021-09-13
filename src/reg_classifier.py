import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn import metrics

from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")

from sklearn.base import BaseEstimator, ClassifierMixin
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier

def borders_no_empty_bins(ser_target, hist):
    """
    Функция определения количества непустых бинов, 
    в которые попадают значения переданного сегмента 
    таргет-переменной ser_target
    """
    
    bins_borders_pairs_list = []
    prev_border = None
    for b in hist:
        if prev_border:
            # Проверка - попадает ли хоть одно значение таргета в рассматриваемый бин
            if b != hist[-1]:
                # Если текущий бин не последний, то его правая граница исключается
                if len(ser_target[(ser_target >= prev_border) & (ser_target < b)]) > 0:
                    bins_borders_pairs_list.append([prev_border, b])
            else:
                # Если текущий бин последний, то его правая граница включается
                if len(ser_target[(ser_target >= prev_border) & (ser_target <= b)]) > 0:
                    bins_borders_pairs_list.append([prev_border, b])
        prev_border = b
    return bins_borders_pairs_list

class RegressionClassifier(BaseEstimator, ClassifierMixin):
    """Решение задачи регрессии, используя модели классификации"""
    
    #def __init__(self, model=LGBMClassifier, N=5, bins_numb=100):
    def __init__(self, model_class=LogisticRegression, N=5, bins_numb=100):
        """
        Инициализация
        model - модель классификатора
        N - количество бинов, на которые делятся данные на каждом уровне
        bins_nums - итоговое число бинов
        """
        self.model_class = model_class
        self.N = N
        self.bins_numb = bins_numb

        # Массив для хранения индексов новых классов
        self.class_array = {}
        # Массив для пересчёта нового таргета в старый
        self.target_class_arr = {}
        # Массив для обученных классификаторов
        self.model_clf_arr = {}
        
        self.bins_total_borders = 0
        

    def split_n_bins(self, target, n):
        """
        Функция деления датафрейма на n примерно равных частей 
        (границы разбивки сдвигаются к ближайшим границам бинов)
        target: ???
        n: ???
        """

        hist, bin_edges = np.histogram(target, bins=n)
        bin_edges_mapping = [bin_edges[i:i+2] for i, edge in enumerate(bin_edges) if len(bin_edges[i:i+2]) == 2]
        bin_indices_mapping = []
        for bin in bin_edges_mapping:
            target_in_bin = ((target < bin[1]) & (target >= bin[0])).astype(int)
            n_elements_in_bin = sum(target_in_bin)
            first_nonzero_idx = (target_in_bin != 0).argmax(axis=0) # index of first element in bin
            last_nonzero_idx = first_nonzero_idx + n_elements_in_bin - 1 # index of last element in bin
            bin_indices = (first_nonzero_idx, last_nonzero_idx)
            bin_indices_mapping.append(bin_indices)
        return bin_indices_mapping, bin_edges_mapping 
    
    def save_class(self, ind_arr_new, class_array):
        """Функция вычисления новых лейблов для последующей работы классификаторов
        ind_arr_new - массив с диапазонами индексов, разделённых на бины (функция split_n_bins)
        class_array - внешний массив, хранящий вычисленные новые лейблы таргета для всех индексов датасета
        class_array_cur - возвращаемый массив с новыми лейблами только для текущей части датасета
        """
        class_array_cur = {}
        for i in range(len(ind_arr_new)):
            for j in range(ind_arr_new[i][0], ind_arr_new[i][1]):
                if j in class_array:
                    class_array[j] = class_array[j] + '-' + str(i)
                else:
                    class_array[j] = '0-' + str(i)
                class_array_cur[j] = class_array[j]
        return class_array_cur


    def split_recursion(self, X, y, n, cl):
        """
        Рекурсивная функция для деления текущего сегмента датафрейма на n бинов и обучения классификаторов
        n: ???
        cl: лейбл текущего бина, для которого запускается функция split_recursion для дальнейшего деления
        """
        
        # Получение массива индексов для каждого бина после деления
        bin_indices_mapping, bin_edges_mapping = self.split_n_bins(y, n)
        
        # Вычисление новых лейблов для текущего сегмента данных
        class_array_cur = self.save_class(bin_indices_mapping, self.class_array)

        # Перебор текущих бинов, определение нового класса текущего бина, запуск текущей функции для дальнейшего деления на бины
        for i in range(len(bin_indices_mapping)):
            bin_start_idx = bin_indices_mapping[i][0]
            bin_end_idx = bin_indices_mapping[i][1]
            bin_idx = range(bin_start_idx, bin_end_idx)
            # Сохраняем текущий лейбл бина
            class_cur = cl

            # Проверка правильности вычисленного нового лейбла??
            class_new = class_cur + '-' + str(i)
            class_new_2 = class_array_cur[bin_indices_mapping[i][0]]
            if class_new != class_new_2:
                raise Exception(f"Error, class_new = {class_new}, class_new_2 = {class_new_2}")

            # Выделение текущего бина в новый Series
            y_cur = y[bin_idx]

            # Если в новом сегменте таргета более одного итогового бина, запускаем текущую функцию снова для дальнейшего деления
            bins_cur_borders = borders_no_empty_bins(y_cur, self.bins_total_borders)
            if len(bins_cur_borders) > 1:
                self.split_recursion(X[bin_idx], y_cur, n, class_new)

            # Иначе дополнение массива для пересчёта нового таргета в старый
            else:
                self.target_class_arr[class_new] = (bins_cur_borders[0][0] + bins_cur_borders[0][1]) / 2

        # Создание и обучение классификатора на текущем сегменте данных с новым лэйблом
        self.model_clf_arr[class_cur] = self.model_class()
        self.model_clf_arr[class_cur].fit(X, pd.Series(class_array_cur))
    

    def predict_row(self, df_row, cl):
        """Предсказание таргета (старого) для текущего ряда"""
        
        if cl in self.model_clf_arr:
            pred = self.model_clf_arr[cl].predict(df_row.reshape(1, -1))[0]
            class_result = self.predict_row(df_row, pred)
        else:
            return self.target_class_arr[cl]

        return class_result


    def itertuples_func(self, df):
        """Предсказание таргета (старого) для всего датасета"""
        
        return np.array([self.predict_row(np.array(row), '0') for row in df.itertuples(index=False)])
        

    def fit(self, X, y, random_state=None):
        """Обучение модели"""
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        idx_sorted = np.argsort(y)
        y_sorted = y[idx_sorted]
        X_sorted = X[idx_sorted]

        
        # Расчёт границ итогового набора бинов
        self.bins_total_borders = np.histogram(y_sorted, bins=self.bins_numb)[1]

        # Запуск основной функции, которая заполняет созданные выше массивы
        self.split_recursion(X_sorted, y_sorted, self.N, '0')
        
        return self
    

    def predict(self, X):
        """Предиктор"""

        p = self.itertuples_func(X)
        
        return p
