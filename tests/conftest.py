import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@pytest.fixture(scope='module')
def airbnb_df():
    return pd.read_csv('data/AB_NYC_2019_EDA.csv').drop(columns=['Unnamed: 0'])


@pytest.fixture(scope='module')
def airbnb_split(airbnb_df):
    df = airbnb_df
    target_name = 'price'
    X, y = df.drop(columns=[target_name]), df[target_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train_scaled, X_test_scaled, y_train, y_test
