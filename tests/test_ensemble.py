from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error

from regression_classifier import ClassRegressorEnsemble


class TestEnsemble:
    def test_fit_two_bins_two_levels(self):
        X = [[1], [2], [3], [4]]
        y = [1, 2, 3, 4]

        model = ClassRegressorEnsemble(n_bins=2, n_levels=2)

        model.fit(X, y)

        print(model.models)
        assert len(model.models) == 3
        assert model.models[(0, (0,))].predict(X).tolist() == [0, 0, 1, 1]
        assert model.models[(1, (0, 0))].predict(X).tolist() == [0, 1, 1, 1]
        assert model.models[(1, (0, 1))].predict(X).tolist() == [0, 0, 0, 1]

        assert model.predict(X).tolist() == y

    def test_better_than_dummy(self, airbnb_split):
        X_train_scaled, X_test_scaled, y_train, y_test = airbnb_split
        model = ClassRegressorEnsemble(n_bins=2, n_levels=2)
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

    def test_fit_many_levels_better_than_dummy(self, airbnb_split):
        X_train_scaled, X_test_scaled, y_train, y_test = airbnb_split
        model = ClassRegressorEnsemble(n_bins=2, n_levels=4)
        model.fit(X_train_scaled, y_train)

        assert len(model.models) == 1 + 2 + 2*2 + 4*2

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
