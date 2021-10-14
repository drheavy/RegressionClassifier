from src.reg_classifier import *

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

        else:
            self.log_exp = 'norm'

            self.class_reg_ens.fit(X, y)

    def predict(self, X):
        # Преобразование таргет переменной, обратное тому, которое было выполнено в методе fit
        if self.log_exp == 'log':
            pred = self.class_reg_ens.predict(X)
            return np.exp(pred)

        else:
            pred = self.class_reg_ens.predict(X)
            return pred

