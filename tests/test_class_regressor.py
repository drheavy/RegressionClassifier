from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error

from regression_classifier import ClassRegressor


class TestClassRegressor:
    def test_fit_two_bins(self):
        clf = ClassRegressor(n_bins=2)

        X = [[1], [2]]
        y = [1, 2]

        clf.fit(X, y)

        assert clf.bin_borders.shape == (2, 2)
        assert clf.y_classes.tolist() == [0, 1]

        assert clf.predict(X).tolist() == [0, 1]
        assert clf.predict(X, regression=True).tolist() == [1, 2]

    def test_fit_three_bins(self):
        clf = ClassRegressor(n_bins=3)

        X = [[1], [2], [3]]
        y = [1, 2, 3]

        clf.fit(X, y)

        assert clf.bin_borders.shape == (3, 2)
        assert clf.y_classes.tolist() == [0, 1, 2]

        assert clf.predict(X).tolist() == [0, 1, 2]
        assert clf.predict(X, regression=True).tolist() == [1, 2, 3]

    def test_better_than_dummy(self, airbnb_split):
        X_train_scaled, X_test_scaled, y_train, y_test = airbnb_split
        clf = ClassRegressor(n_bins=2)
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


