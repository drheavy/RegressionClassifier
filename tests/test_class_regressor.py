import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from regression_classifier import ClassRegressor


class TestClassRegressor:
    @pytest.mark.parametrize("bins_calc_method", ['equal', 'percentile'])
    def test_fit_two_bins(self, bins_calc_method):
        clf = ClassRegressor(n_bins=2, bins_calc_method=bins_calc_method)

        X = [[1], [2]]
        y = [1, 2]

        clf.fit(X, y)

        assert clf.bin_borders.shape == (2, 2)
        assert clf.y_classes.tolist() == [0, 1]

        assert clf.predict(X).tolist() == [0, 1]
        assert clf.predict(X, regression=True).tolist() == [1, 2]

    @pytest.mark.parametrize("bins_calc_method", ['equal', 'percentile'])
    def test_fit_three_bins(self, bins_calc_method):
        clf = ClassRegressor(n_bins=3, bins_calc_method=bins_calc_method)

        X = [[1], [2], [3]]
        y = [1, 2, 3]

        clf.fit(X, y)

        assert clf.bin_borders.shape == (3, 2)
        assert clf.y_classes.tolist() == [0, 1, 2]

        assert clf.predict(X).tolist() == [0, 1, 2]
        assert clf.predict(X, regression=True).tolist() == [1, 2, 3]

    @pytest.mark.parametrize("bins_calc_method", ['equal', 'percentile'])
    def test_better_than_dummy(self, airbnb_split, bins_calc_method):
        X_train_scaled, X_test_scaled, y_train, y_test = airbnb_split
        clf = ClassRegressor(n_bins=2, bins_calc_method=bins_calc_method)
        clf.fit(X_train_scaled, y_train)

        pred_train = clf.predict(X_train_scaled, regression=True)
        pred_test = clf.predict(X_test_scaled, regression=True)
        train_mae = mean_absolute_error(y_train, pred_train)
        test_mae = mean_absolute_error(y_test, pred_test)

        dummy_regr = DummyRegressor(strategy="mean")
        dummy_regr.fit(X_train_scaled, y_train)

        dummy_pred_train = dummy_regr.predict(X_train_scaled)
        dummy_pred_test = dummy_regr.predict(X_test_scaled)
        dummy_train_mae = mean_absolute_error(y_train, dummy_pred_train)
        dummy_test_mae = mean_absolute_error(y_test, dummy_pred_test)

        assert train_mae <= dummy_train_mae
        assert test_mae <= dummy_test_mae

    @pytest.mark.parametrize("bins_calc_method", ['equal', 'percentile'])
    def test_classes_are_classes(self, airbnb_split, bins_calc_method):
        X_train_scaled, X_test_scaled, y_train, y_test = airbnb_split
        N_BINS = 10
        clf = ClassRegressor(n_bins=N_BINS, bins_calc_method=bins_calc_method)
        clf.fit(X_train_scaled, y_train)

        classes_list = clf.y_classes.tolist()
        assert min(classes_list) == 0
        assert max(classes_list) == N_BINS-1
        assert classes_list == [int(classes_list) for classes_list in classes_list]

        pred_test_classes = clf.predict(X_test_scaled)
        pred_classes_list = np.unique(pred_test_classes).tolist()
        assert min(pred_classes_list) >= 0
        assert max(pred_classes_list) <= N_BINS-1
        assert pred_classes_list == [int(pred_classes_list) for pred_classes_list in pred_classes_list]

    @pytest.mark.parametrize("bins_calc_method, output", [('equal', [[1.0, 5.0], [5.0, 9.0]]),
                                                         ('percentile', [[1.0, 2.5], [2.5, 9.0]])])
    def test_bins_equal(self, bins_calc_method, output):
        clf = ClassRegressor(n_bins=2, bins_calc_method=bins_calc_method)

        X = [[1], [2], [3], [9]]
        y = [1, 2, 3, 9]

        clf.fit(X, y)

        assert clf.bin_borders.tolist() == output

    def test_linreg_is_better_than_none(self):
        X = np.array(list(range(100))).reshape(-1, 1).tolist()
        y = list(range(100))

        clf = ClassRegressor(n_bins=5)
        clf.fit(X, y)

        pred_train = clf.predict(X, regression=True)
        train_mae = mean_absolute_error(y, pred_train)

        clf_linreg = ClassRegressor(n_bins=5, leaf_model=LinearRegression)
        clf_linreg.fit(X, y)

        pred_train_linreg = clf_linreg.predict(X, regression=True)
        train_mae_linreg = mean_absolute_error(y, pred_train_linreg)

        assert train_mae_linreg < train_mae

    def test_perc_better_than_equal(self, airbnb_split):
        X_train_scaled, X_test_scaled, y_train, y_test = airbnb_split

        clf_eq = ClassRegressor(n_bins=2, bins_calc_method='equal')
        clf_eq.fit(X_train_scaled, y_train)

        pred_train_eq = clf_eq.predict(X_train_scaled, regression=True)
        pred_test_eq = clf_eq.predict(X_test_scaled, regression=True)
        train_mse_eq = mean_squared_error(y_train, pred_train_eq)
        test_mse_eq = mean_squared_error(y_test, pred_test_eq)

        clf_perc = ClassRegressor(n_bins=2, bins_calc_method='percentile')
        clf_perc.fit(X_train_scaled, y_train)

        pred_train_perc = clf_perc.predict(X_train_scaled, regression=True)
        pred_test_perc = clf_perc.predict(X_test_scaled, regression=True)
        train_mse_perc = mean_squared_error(y_train, pred_train_perc)
        test_mse_perc = mean_squared_error(y_test, pred_test_perc)

        assert train_mse_perc < train_mse_eq
        assert test_mse_perc < test_mse_eq
