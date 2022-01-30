import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from regression_classifier import ClassRegressorSplit, ClassRegressorTree, RecursiveClassRegressor


class TestClassRegressorTree:
    @pytest.mark.parametrize("bins_calc_method", ['equal', 'percentile'])
    def test_fit_two_bins(self, bins_calc_method):
        clf = ClassRegressorTree(n_bins=2,
                                 n_splits=1,
                                 bins_calc_method=bins_calc_method)

        X = [[1], [2]]
        y = [1, 2]

        clf.fit(X, y)

        assert clf.split.bin_borders.shape == (2, 2)
        assert clf.split.y_classes.tolist() == [0, 1]

        assert isinstance(clf.child_models[0], clf.leaf_model_cls)
        assert isinstance(clf.child_models[1], clf.leaf_model_cls)

        pred, proba = clf.predict(X, classification=True)
        assert pred.tolist() == [0, 1]
        reg_pred = clf.predict(X)
        assert 1.5 > reg_pred[0] >= 1
        assert 2 >= reg_pred[1] > 1.5

    def test_fit_two_splits(self):
        clf = ClassRegressorTree(n_bins=2,
                                 n_splits=2)

        X = [[1], [2], [3], [4]]
        y = [1, 2, 3, 4]

        clf.fit(X, y)

        assert clf.split.bin_borders.shape == (2, 2)
        assert clf.split.y_classes.tolist() == [0, 0, 1, 1]

        assert isinstance(clf.child_models[0], ClassRegressorTree)
        assert isinstance(clf.child_models[0].child_models[0], clf.leaf_model_cls)
        assert isinstance(clf.child_models[0].child_models[1], clf.leaf_model_cls)
        assert isinstance(clf.child_models[1], ClassRegressorTree)
        assert isinstance(clf.child_models[1].child_models[0], clf.leaf_model_cls)
        assert isinstance(clf.child_models[1].child_models[1], clf.leaf_model_cls)

        pred, proba = clf.predict(X, classification=True)
        assert pred.tolist() == [0, 0, 1, 1]
        reg_pred = clf.predict(X)
        assert 2 > reg_pred[0]
        assert 3 >= reg_pred[1]
        assert 3 >= reg_pred[2]
        assert 4 >= reg_pred[3]

    @pytest.mark.parametrize("bins_calc_method", ['equal', 'percentile'])
    def test_fit_three_bins(self, bins_calc_method):
        clf = ClassRegressorTree(n_bins=3,
                                 n_splits=1,
                                 bins_calc_method=bins_calc_method)

        X = [[1], [2], [3]]
        y = [1, 2, 3]

        clf.fit(X, y)

        assert clf.split.bin_borders.shape == (3, 2)
        assert clf.split.y_classes.tolist() == [0, 1, 2]

        assert isinstance(clf.child_models[0], clf.leaf_model_cls)
        assert isinstance(clf.child_models[1], clf.leaf_model_cls)

        pred, proba = clf.predict(X, classification=True)
        assert pred.tolist() == [0, 1, 2]
        reg_pred = clf.predict(X)
        assert 2 > reg_pred[0] >= 1
        assert 2.2 >= reg_pred[1] > 1.5
        assert 3 >= reg_pred[2] > 2.2

    @pytest.mark.parametrize("bins_calc_method", ['equal', 'percentile'])
    def test_better_than_dummy(self, airbnb_split, bins_calc_method):
        X_train_scaled, X_test_scaled, y_train, y_test = airbnb_split
        clf = ClassRegressorTree(n_bins=2,
                                 n_splits=1,
                                 bins_calc_method=bins_calc_method)
        clf.fit(X_train_scaled, y_train)

        pred_train = clf.predict(X_train_scaled)
        pred_test = clf.predict(X_test_scaled)
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
        n_bins = 5
        clf = ClassRegressorTree(n_bins=n_bins,
                                 n_splits=1,
                                 bins_calc_method=bins_calc_method)
        clf.fit(X_train_scaled, y_train)

        classes_list = clf.split.y_classes.tolist()
        assert min(classes_list) == 0
        assert max(classes_list) == n_bins - 1
        assert classes_list == [int(classes_list) for classes_list in classes_list]

        pred_test_classes, probas = clf.predict(X_test_scaled, classification=True)
        pred_classes_list = np.unique(pred_test_classes).tolist()
        assert min(pred_classes_list) >= 0
        assert max(pred_classes_list) <= n_bins - 1
        assert pred_classes_list == [int(pred_classes_list) for pred_classes_list in pred_classes_list]

    def test_linreg_is_better_than_none(self):
        X = np.array(list(range(100))).reshape(-1, 1).tolist()
        y = list(range(100))

        clf = ClassRegressorTree(n_bins=5)
        clf.fit(X, y)

        pred_train = clf.predict(X)
        train_mae = mean_absolute_error(y, pred_train)

        clf_linreg = ClassRegressorTree(n_bins=5, leaf_model_cls_name='LinearRegression')
        clf_linreg.fit(X, y)

        pred_train_linreg = clf_linreg.predict(X)
        train_mae_linreg = mean_absolute_error(y, pred_train_linreg)

        assert train_mae_linreg < train_mae

    def test_perc_better_than_equal(self, airbnb_split):
        X_train_scaled, X_test_scaled, y_train, y_test = airbnb_split

        clf_eq = ClassRegressorTree(n_bins=2, bins_calc_method='equal')
        clf_eq.fit(X_train_scaled, y_train)

        pred_train_eq = clf_eq.predict(X_train_scaled)
        pred_test_eq = clf_eq.predict(X_test_scaled)
        train_mse_eq = mean_squared_error(y_train, pred_train_eq)
        test_mse_eq = mean_squared_error(y_test, pred_test_eq)

        clf_perc = ClassRegressorTree(n_bins=2, bins_calc_method='percentile')
        clf_perc.fit(X_train_scaled, y_train)

        pred_train_perc = clf_perc.predict(X_train_scaled)
        pred_test_perc = clf_perc.predict(X_test_scaled)
        train_mse_perc = mean_squared_error(y_train, pred_train_perc)
        test_mse_perc = mean_squared_error(y_test, pred_test_perc)

        assert train_mse_perc < train_mse_eq
        assert test_mse_perc < test_mse_eq


class TestClassRegressorSplit:
    @pytest.mark.parametrize("bins_calc_method", ['equal', 'percentile'])
    def test_fit_two_bins(self, bins_calc_method):
        clf = ClassRegressorSplit(n_bins=2, bins_calc_method=bins_calc_method)

        X = [[1], [2]]
        y = [1, 2]

        clf.fit(X, y)

        assert clf.bin_borders.shape == (2, 2)
        assert clf.y_classes.tolist() == [0, 1]

        pred = clf.predict(X)
        proba = clf.predict_proba(X)
        assert pred.tolist() == [0, 1]
        assert proba.shape == (2, 2)

    @pytest.mark.parametrize("bins_calc_method, output", [('equal', [[1.0, 5.0], [5.0, 9.0]]),
                                                          ('percentile', [[1.0, 2.5], [2.5, 9.0]])])
    def test_bins_equal(self, bins_calc_method, output):
        clf = ClassRegressorSplit(n_bins=2, bins_calc_method=bins_calc_method)

        X = [[1], [2], [3], [9]]
        y = [1, 2, 3, 9]

        clf.fit(X, y)

        assert clf.bin_borders.tolist() == output


class TestRecursiveClassRegressor:
    @pytest.mark.parametrize("bins_calc_method", ['equal', 'percentile'])
    def test_fit_two_bins_two_levels(self, bins_calc_method):
        X = [[1], [2], [3], [4]]
        y = [1, 2, 3, 4]

        model = RecursiveClassRegressor(
            n_bins=2,
            n_splits=2,
            bins_calc_method=bins_calc_method,
        )
        model.fit(X, y)


        assert isinstance(model.tree, ClassRegressorTree)
        assert isinstance(model.tree.split, ClassRegressorSplit)
        assert len(model.tree.child_models) == 2
        assert isinstance(model.tree.child_models[0], ClassRegressorTree)
        assert isinstance(model.tree.child_models[1], ClassRegressorTree)

        pred, proba = model.tree.predict(X, classification=True)
        assert pred.tolist() == [0, 0, 1, 1]

        child = model.tree.child_models[0]
        assert isinstance(child, ClassRegressorTree)
        assert len(child.child_models) == 2
        assert isinstance(child.child_models[0], child.leaf_model_cls)
        assert isinstance(child.child_models[1], child.leaf_model_cls)
        pred, proba = child.predict(X, classification=True)
        assert pred.tolist() == [0, 1, 1, 1]

        child = model.tree.child_models[1]
        assert isinstance(child, ClassRegressorTree)
        assert len(child.child_models) == 2
        assert isinstance(child.child_models[0], child.leaf_model_cls)
        assert isinstance(child.child_models[1], child.leaf_model_cls)
        pred, proba = child.predict(X, classification=True)
        assert pred.tolist() == [0, 0, 0, 1]

        pred = model.predict(X)

        assert len(pred) == len(y)

    @pytest.mark.parametrize("bins_calc_method", ['equal', 'percentile'])
    def test_better_than_dummy(self, airbnb_split, bins_calc_method):
        X_train_scaled, X_test_scaled, y_train, y_test = airbnb_split
        model = RecursiveClassRegressor(n_bins=2, n_splits=2, bins_calc_method=bins_calc_method)
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

    @pytest.mark.parametrize("bins_calc_method", ['equal', 'percentile'])
    def test_fit_many_levels_better_than_dummy(self, airbnb_split, bins_calc_method):
        X_train_scaled, X_test_scaled, y_train, y_test = airbnb_split
        model = RecursiveClassRegressor(n_bins=2, n_splits=4, bins_calc_method=bins_calc_method)
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
