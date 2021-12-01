import numpy as np
import pandas as pd
from sklearn import metrics


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
        bin_borders = np.histogram_bin_edges(y, bins=n_bins)
    else:
        raise Exception("Unknown method")

    return bin_borders


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def dataframe_metrics(y_test, y_pred):
    stats = [
       metrics.mean_absolute_error(y_test, y_pred),
       np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
       metrics.r2_score(y_test, y_pred),
       mean_absolute_percentage_error(y_test, y_pred)
    ]
    return stats
