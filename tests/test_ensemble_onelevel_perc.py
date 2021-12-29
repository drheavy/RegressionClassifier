import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from regression_classifier import ClassRegressorOnelevel, ClassRegressorOnelevelEnsemble


class TestClassRegressor:
    def test_fit_two_bins_two_levels(self):
        X = [[1], [2]]
        y = [1, 2]

        model = ClassRegressorOnelevelEnsemble(n_bins=4, bins_calc_method='percentile')

        model.fit(X, y)

        assert len(model.models) == 3
        assert model.predict(X).tolist() == [1.25, 1.75]

    def test_better_than_dummy(self, airbnb_split):
        X_train_scaled, X_test_scaled, y_train, y_test = airbnb_split
        model = ClassRegressorOnelevelEnsemble(n_bins=10, bins_calc_method='percentile')
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

    def test_bins_perc(self):
        clf = ClassRegressorOnelevelEnsemble(n_bins=4, bins_calc_method='percentile')

        X = [[1], [2], [3], [9]]
        y = [1, 2, 3, 9]

        clf.fit(X, y)

        assert clf.bin_edges[1:].tolist() == [1.75, 2.5, 4.5, 9.0]

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
