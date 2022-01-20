import numpy as np
import pandas as pd
import scipy.stats
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from tqdm.auto import tqdm
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
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


def run_benchmark(train_X, test_X, train_Y, test_Y, model, hparam_space, search_n_iter=10):
    search = RandomizedSearchCV(model,
                                cv=KFold(n_splits=4),
                                param_distributions=hparam_space,
                                scoring=make_scorer(mean_absolute_error),
                                verbose=8,
                                n_jobs=4,
                                n_iter=search_n_iter)
    search.fit(train_X, train_Y)
    pred_test = search.predict(test_X)

    mae = mean_absolute_error(test_Y, pred_test)

    benchmark_result = {
        'best_params': search.best_params_,
        'score': mae,
    }
    print(benchmark_result)
    return benchmark_result


def run_benchmarks():
    df = load_dataframe()
    target_name = 'median_house_value'
    X, y = df.drop(columns=[target_name]), df[target_name]
    train_X, test_X, train_Y, test_Y = train_test_split(X, y)

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
            'model__alpha': scipy.stats.norm(0, 1),
            'model__l1_ratio': scipy.stats.norm(0.5, 1),
        },

    ]

    results = {}
    for model, hparam_space in tqdm(zip(pipelines, hparam_spaces), total=len(pipelines)):
        results[model.named_steps.model.__class__.__name__] = run_benchmark(train_X, test_X, train_Y, test_Y, model, hparam_space)

    return results


if __name__ == '__main__':
    results = run_benchmarks()
    print(results)
