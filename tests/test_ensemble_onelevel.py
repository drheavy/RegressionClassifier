import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from regression_classifier import ClassRegressorOnelevel, ClassRegressorOnelevelEnsemble


class TestClassRegressor:
    def test_fit_two_bins_two_levels(self):
        X = [[1], [2]]
        y = [1, 2]

        model = ClassRegressorOnelevelEnsemble(n_bins=4)

        model.fit(X, y)

        assert len(model.models) == 3
        assert model.predict(X).tolist() == [1.25, 1.75]

    def test_better_than_dummy(self, airbnb_split):
        X_train_scaled, X_test_scaled, y_train, y_test = airbnb_split
        model = ClassRegressorOnelevelEnsemble(n_bins=10)
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

    def test_bins_equal(self):
        clf = ClassRegressorOnelevelEnsemble(n_bins=4)

        X = [[1], [2], [3], [9]]
        y = [1, 2, 3, 9]

        clf.fit(X, y)

        assert clf.bin_edges[1:].tolist() == [3, 5, 7, 9]

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
