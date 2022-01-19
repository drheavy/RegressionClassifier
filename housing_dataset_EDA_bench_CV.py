import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor

from regression_classifier import *

df = pd.read_csv('./data/housing.csv')
df = df.dropna().reset_index(drop=True)
df = pd.get_dummies(df, columns=['ocean_proximity'], prefix='ocean', drop_first=True)
target_name = 'median_house_value'

X, y = df.drop(columns=[target_name]), df[target_name]

kf = KFold(n_splits=4)
kf.get_n_splits(X)

class_reg = ClassRegressorEnsemble(n_bins=2, n_levels=5, bins_calc_method='equal', leaf_size=5000, leaf_model=None)
class_reg_onelevel = ClassRegressorOnelevelEnsemble(n_bins=20, bins_calc_method='equal')
lin_reg = LinearRegression()
lgbm_reg = LGBMRegressor()
models = [class_reg, class_reg_onelevel, lin_reg, lgbm_reg]
scaler = StandardScaler()

scores = {}

for model in models:
    model_name = model.__class__.__name__
    print(f"Scores for {model_name} model")

    scores[model_name] = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        model.fit(X_train_scaled, y_train)
        pred_test = model.predict(X_test_scaled)

        mae = metrics.mean_absolute_error(y_test, pred_test)
        print("MAE = ", np.round(mae, 3))
        scores[model_name].append(mae)

print("")

for model in models:
    model_name = model.__class__.__name__
    score_mean = np.round(np.mean(scores[model_name]), 3)
    print(f"MAE mean for {model_name} = {score_mean}")
