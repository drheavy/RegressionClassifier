import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from regression_classifier import ClassRegressorOnelevel, ClassRegressorOnelevelEnsemble, ClassRegressorEnsemble


class TestOnelevelEnsemble:
    @pytest.mark.parametrize("bins_calc_method", ['equal', 'percentile'])
    def test_fit_two_bins_two_levels(self, bins_calc_method):
        X = [[1], [2]]
        y = [1, 2]

        model = ClassRegressorOnelevelEnsemble(n_bins=4, bins_calc_method=bins_calc_method)

        model.fit(X, y)

        assert len(model.models) == 3
        assert model.predict(X).tolist() == [1.25, 1.75]

    @pytest.mark.parametrize("bins_calc_method", ['equal', 'percentile'])
    def test_better_than_dummy(self, airbnb_split, bins_calc_method):
        X_train_scaled, X_test_scaled, y_train, y_test = airbnb_split
        model = ClassRegressorOnelevelEnsemble(n_bins=10, bins_calc_method=bins_calc_method)
        model.fit(X_train_scaled, y_train)

        pred_train = model.predict(X_train_scaled)
        pred_test = model.predict(X_test_scaled)
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

    @pytest.mark.parametrize("bins_calc_method, output", [('equal', [3, 5, 7, 9]),
                                                         ('percentile', [1.75, 2.5, 4.5, 9.0])])
    def test_bins_equal(self, bins_calc_method, output):
        clf = ClassRegressorOnelevelEnsemble(n_bins=4, bins_calc_method=bins_calc_method)

        X = [[1], [2], [3], [9]]
        y = [1, 2, 3, 9]

        clf.fit(X, y)

        assert clf.bin_edges[1:].tolist() == output

    def test_linreg_is_better_than_none(self):
        X = np.array(list(range(100))).reshape(-1, 1).tolist()
        y = list(range(100))

        clf = ClassRegressorOnelevelEnsemble(n_bins=10)
        clf.fit(X, y)

        pred_train = clf.predict(X)
        train_mae = mean_absolute_error(y, pred_train)

        clf_linreg = ClassRegressorOnelevelEnsemble(n_bins=10, leaf_model=LinearRegression)
        clf_linreg.fit(X, y)

        pred_train_linreg = clf_linreg.predict(X)
        train_mae_linreg = mean_absolute_error(y, pred_train_linreg)

        assert train_mae_linreg < train_mae

    def test_perc_better_than_equal(self, airbnb_split):
        X_train_scaled, X_test_scaled, y_train, y_test = airbnb_split

        clf_eq = ClassRegressorOnelevelEnsemble(n_bins=10, bins_calc_method='equal')
        clf_eq.fit(X_train_scaled, y_train)

        pred_train_eq = clf_eq.predict(X_train_scaled)
        pred_test_eq = clf_eq.predict(X_test_scaled)
        train_mse_eq = mean_squared_error(y_train, pred_train_eq)
        test_mse_eq = mean_squared_error(y_test, pred_test_eq)

        clf_perc = ClassRegressorOnelevelEnsemble(n_bins=10, bins_calc_method='percentile')
        clf_perc.fit(X_train_scaled, y_train)

        pred_train_perc = clf_perc.predict(X_train_scaled)
        pred_test_perc = clf_perc.predict(X_test_scaled)
        train_mse_perc = mean_squared_error(y_train, pred_train_perc)
        test_mse_perc = mean_squared_error(y_test, pred_test_perc)

        assert train_mse_perc < train_mse_eq
        assert test_mse_perc < test_mse_eq

    def test_onelevel_is_better_than_normal(self, airbnb_split):
        X_train_scaled, X_test_scaled, y_train, y_test = airbnb_split

        clf_onelevel = ClassRegressorOnelevelEnsemble(n_bins=20, bins_calc_method='equal')
        clf_onelevel.fit(X_train_scaled, y_train)

        pred_train_onelevel = clf_onelevel.predict(X_train_scaled)
        pred_test_onelevel = clf_onelevel.predict(X_test_scaled)
        train_mse_onelevel = mean_squared_error(y_train, pred_train_onelevel)
        test_mse_onelevel = mean_squared_error(y_test, pred_test_onelevel)

        clf_norm = ClassRegressorEnsemble(n_bins=2, n_levels=5, bins_calc_method='equal')
        clf_norm.fit(X_train_scaled, y_train)

        pred_train_norm = clf_norm.predict(X_train_scaled)
        pred_test_norm = clf_norm.predict(X_test_scaled)
        train_mse_norm = mean_squared_error(y_train, pred_train_norm)
        test_mse_norm = mean_squared_error(y_test, pred_test_norm)

        assert train_mse_norm > train_mse_onelevel
        assert test_mse_norm > test_mse_onelevel
