import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from tqdm.auto import tqdm
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet

from regression_classifier import ClassRegressorEnsemble, ClassRegressorOnelevelEnsemble


def load_dataframe():
    df = pd.read_csv('./data/housing.csv')
    df = df.dropna().reset_index(drop=True)
    df = pd.get_dummies(df, columns=['ocean_proximity'], prefix='ocean', drop_first=True)
    return df


def run_benchmark(X, y, model, hparam_space, search_n_iter=10):
    fold_scores = []
    kf = KFold(n_splits=4, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        search = RandomizedSearchCV(model,
                                    cv=KFold(n_splits=3),
                                    param_distributions=hparam_space,
                                    scoring=make_scorer(mean_absolute_error),
                                    verbose=1,
                                    n_iter=search_n_iter)
        search.fit(X_train, y_train)
        pred_test = search.predict(X_test)

        mae = mean_absolute_error(y_test, pred_test)
        fold_scores.append(mae)

    benchmark_result = {
        'fold_scores': fold_scores,
        'mean_score': np.mean(fold_scores),
        'std_score': np.std(fold_scores),
    }
    return benchmark_result


def run_benchmarks():
    df = load_dataframe()
    target_name = 'median_house_value'
    X, y = df.drop(columns=[target_name]), df[target_name]

    pipelines = [
        Pipeline([
            ('inputer', SimpleImputer()),
            ('scaler', StandardScaler()),
            ('model', ClassRegressorEnsemble()),
        ]),
        Pipeline([
            ('inputer', SimpleImputer()),
            ('scaler', StandardScaler()),
            ('model', ClassRegressorOnelevelEnsemble()),
        ]),
        Pipeline([
            ('inputer', SimpleImputer()),
            ('scaler', StandardScaler()),
            ('model', ElasticNet()),
        ]),
    ]

    hparam_spaces = [
        {
            'model__n_bins': [2, 5],
            'model__n_levels': [2, 5, 10, 30],
            'model__bin_calc_method': ['equal', 'percentile'],
            'model__leaf_size': [10, 50, 100],
            'model__leaf_model_cls': [DummyRegressor, LinearRegression],
        },
        {
            'model__n_bins': [10, 20, 30],
            'model__bin_calc_method': ['equal', 'percentile'],
            'model__leaf_model_cls': [DummyRegressor, LinearRegression],
        },
        {
            'model__alpha': np.random.uniform(0, 1),
            'model__l1_ratio': np.random.normal(0.5, 1),
        },

    ]

    results = {}
    for model, hparam_space in tqdm(zip(pipelines, hparam_spaces), total=len(pipelines)):
        results[model.named_steps.model.__class__.__name__] = run_benchmark(X, y, model, hparam_space)

    return results


if __name__ == '__main__':
    results = run_benchmarks()
    print(results)
