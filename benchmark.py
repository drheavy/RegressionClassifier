import json

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
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor

from regression_classifier import RecursiveClassRegressor


def load_dataframe():
    df = pd.read_csv('./data/housing.csv')
    df = df.dropna().reset_index(drop=True)
    df = pd.get_dummies(df, columns=['ocean_proximity'], prefix='ocean', drop_first=True)
    return df


def run_benchmark(train_X, test_X, train_Y, test_Y, model, hparam_space, search_n_iter=50):
    search = RandomizedSearchCV(model,
                                cv=KFold(n_splits=4),
                                param_distributions=hparam_space,
                                scoring='neg_mean_absolute_error',
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
    return search, benchmark_result


def run_benchmarks():
    df = load_dataframe()
    target_name = 'median_house_value'
    X, y = df.drop(columns=[target_name]), df[target_name]
    train_X, test_X, train_Y, test_Y = train_test_split(X, y)

    pipelines = [
        Pipeline([
            ('inputer', SimpleImputer()),
            ('scaler', StandardScaler()),
            ('model', RecursiveClassRegressor()),
        ]),
        Pipeline([
            ('inputer', SimpleImputer()),
            ('scaler', StandardScaler()),
            ('model', ElasticNet()),
        ]),
        Pipeline([
            ('inputer', SimpleImputer()),
            ('scaler', StandardScaler()),
            ('model', GradientBoostingRegressor()),
        ]),
        Pipeline([
            ('inputer', SimpleImputer()),
            ('scaler', StandardScaler()),
            ('model', LGBMRegressor()),
        ]),
    ]

    hparam_spaces = [
        { # RecursiveClassRegressor
            'model__n_bins': [2, 3, 5],
            'model__n_splits': [2, 3, 5, 10],
            'model__bins_calc_method': ['equal', 'percentile'],
            'model__leaf_size': [10, 50, 100],
            'model__leaf_model_cls_name': ['DummyRegressor', 'LinearRegression'],
        },
        { # ElasticNet
            'model__alpha': scipy.stats.norm(0.5, 1),
            'model__l1_ratio': scipy.stats.norm(0.5, 0.15),
        },
        # {  # GradientBoostingRegressor
        #     'model__max_depth': np.arange(-1, 20, 2),
        #     'model__subsample': np.arange(0.2, 1.2, 0.2),
        #     'model__n_estimators': np.arange(10, 310, 40),
        # },
        # { # LGBMRegressor
        #     'model__max_depth': np.arange(-1, 20, 2),
        #     'model__subsample': np.arange(0.2, 1.2, 0.2),
        #     'model__n_estimators': np.arange(10, 310, 40),
        # },
    ]

    searches = {}
    results = {}
    for model, hparam_space in tqdm(zip(pipelines, hparam_spaces), total=len(pipelines)):
        model_name = model.named_steps.model.__class__.__name__
        searches[model_name], results[model_name] = run_benchmark(train_X, test_X, train_Y, test_Y, model, hparam_space)

    return searches, results


if __name__ == '__main__':
    searches, results = run_benchmarks()
    print(json.dumps(results, sort_keys=True, indent=4, default=str))
