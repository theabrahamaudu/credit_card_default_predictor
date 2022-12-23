"""
This module is used to preprocess raw input data and user data uploaded on the web UI
"""

import pandas as pd
from pandas import DataFrame
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def preprocess_input(df: DataFrame):
    """
    Takes raw dataset in Pandas dataframe format as input and returns preprocessed features and expected outcomes
    as X and y respectively.

    - The ID column is dropped for simplicity
    - 'EDUCATION' and 'MARRIAGE' columns are one-hot encoded and the encoder is saved.
    - The dataset is split into X and y
    - The X dataset is scaled using StandardScaler and the scaler object is saved
    - X and y dataframes are returned
    Args:
        df: Dataset dataframe

    Returns:
        X (DataFrame): DataFrame
        y (DataFrame): DataFrame
    """
    df = df.copy()

    # Drop ID
    df = df.drop('ID', axis=1)

    # Perform one-hot encoding
    categorical_cols = ['EDUCATION', 'MARRIAGE']

    onehotencoder = OneHotEncoder(sparse=False)

    transformed_data = onehotencoder.fit_transform(df[categorical_cols])
    joblib_file = f"encoder.pkl"
    joblib.dump(onehotencoder, joblib_file)
    print("Encoder saved successfully")
    # the above transformed_data is an array so convert it to dataframe
    encoded_data = pd.DataFrame(transformed_data, index=df.index, columns=onehotencoder.get_feature_names_out())

    # now concatenate the original data and the encoded data using pandas
    concatenated_data = pd.concat([df, encoded_data], axis=1)
    df = concatenated_data.drop(columns=categorical_cols)

    # split df into x,y
    y: DataFrame = df['default.payment.next.month'].copy()
    X = df.drop('default.payment.next.month', axis=1).copy()

    # Scale X with a standard scaler
    scaler = StandardScaler()
    X: DataFrame = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    joblib_file = f"scaler.pkl"
    joblib.dump(scaler, joblib_file)
    print("Scaler saved successfully")

    return X, y


def preprocess_website_input(df: DataFrame):
    """
    Takes raw single user credit card data in Pandas dataframe format as input and returns preprocessed features as X.

    - The ID column is dropped for simplicity
    - 'EDUCATION' and 'MARRIAGE' columns are one-hot encoded using previously saved encoder
    - The dataset is scaled using previously saved scaler
    - Preprocessed dataframe is returned
    Args:
        df (DataFrame): Single user credit card data

    Returns:
        df (DataFrame): Preprocessed data
    """
    df = df.copy()

    # Drop ID
    df = df.drop('ID', axis=1)

    # Perform one-hot encoding
    categorical_cols = ['EDUCATION', 'MARRIAGE']

    onehotencoder = joblib.load('encoder.pkl')

    transformed_data = onehotencoder.transform(df[categorical_cols])

    # the above transformed_data is an array so convert it to dataframe
    encoded_data = pd.DataFrame(transformed_data, index=df.index, columns=onehotencoder.get_feature_names_out())

    # now concatenate the original data and the encoded data using pandas
    concatenated_data = pd.concat([df, encoded_data], axis=1)
    df = concatenated_data.drop(columns=categorical_cols)

    # Scale X with a standard scaler
    scaler = joblib.load(f"scaler.pkl")
    df: DataFrame = pd.DataFrame(scaler.transform(df), columns=df.columns)

    return df
