"""
This module is used to preprocess raw input data and user data uploaded on the web UI
"""

import pandas as pd
from pandas import DataFrame
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from utils.pipeline_log_config import pipeline as logger


def class_split(df: DataFrame):
    """
    Used to split credit card dataset based on the target column as 0 or 1
    Args:
        df (DataFrame):

    Returns:
        data_default (DataFrame):
        data_non_default (DataFrame):
    """
    data_default: DataFrame = df[df['default.payment.next.month'] == 1]
    data_non_default: DataFrame = df[df['default.payment.next.month'] == 0]

    return data_default, data_non_default


def over_sample_dataset(df: DataFrame):
    """
    Used to over-sample credit card dataset with defaulters (1) as minority
    Args:
        df (DataFrame):

    Returns:
        df_train_over (DataFrame):
    """
    data_1, data_0 = class_split(df)
    data_1_over = data_1.sample(len(data_0), replace=True, random_state=234)
    df_train_over = pd.concat([data_1_over, data_0], axis=0)
    df_train_over = df_train_over.sample(frac=1, random_state=234)

    return df_train_over


def SMOTE_oversample_dataset(df: DataFrame):
    """
    Used to over-sample credit card dataset using SMOTE with minority strategy
    Args:
        df (DataFrame):

    Returns:
        df_train_smote_over (DataFrame):
    """
    smote_x = df.drop('default.payment.next.month', axis=1)
    smote_y = df['default.payment.next.month']
    smote = SMOTE(sampling_strategy='minority')
    X_sm, y_sm = smote.fit_resample(smote_x, smote_y)
    df_train_smote_over = pd.concat([X_sm, y_sm], axis='columns')

    return df_train_smote_over


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
    joblib_file = f"../models/encoder.pkl"
    joblib.dump(onehotencoder, joblib_file)
    logger.info("Encoder saved successfully")
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
    joblib_file = f"../models/scaler.pkl"
    joblib.dump(scaler, joblib_file)
    logger.info("Scaler saved successfully")

    return X, y


def preprocess_test_input(df: DataFrame):
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

    onehotencoder = joblib.load('../models/encoder.pkl')

    transformed_data = onehotencoder.transform(df[categorical_cols])

    # the above transformed_data is an array so convert it to dataframe
    encoded_data = pd.DataFrame(transformed_data, index=df.index, columns=onehotencoder.get_feature_names_out())

    # now concatenate the original data and the encoded data using pandas
    concatenated_data = pd.concat([df, encoded_data], axis=1)
    df = concatenated_data.drop(columns=categorical_cols)

    # split df into x,y
    y: DataFrame = df['default.payment.next.month'].copy()
    X = df.drop('default.payment.next.month', axis=1).copy()

    # Scale X with a standard scaler
    scaler = joblib.load(f"../models/scaler.pkl")
    X: DataFrame = pd.DataFrame(scaler.transform(X), columns=X.columns)
    #     logger.info("Web user data processed successfully")

    return X, y


def train_test_preprocess(train_df: DataFrame, test_df: DataFrame):
    """
    Custom function for preprocessing train and test dataset without leakage

    Takes split raw datasets and preprocesses them saparately using uniform parameters
    Args:
        train_df (DataFrame):
        test_df (DataFrame):

    Returns:
        X_train (DataFrame):
        y_train (DataFrame):
        X_test (DataFrame):
        y_test (DataFrame):
    """

    X_train, y_train = preprocess_input(train_df)
    X_test, y_test = preprocess_test_input(test_df)

    return X_train, y_train, X_test, y_test


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

    onehotencoder = joblib.load('../models/encoder.pkl')

    transformed_data = onehotencoder.transform(df[categorical_cols])

    # the above transformed_data is an array so convert it to dataframe
    encoded_data = pd.DataFrame(transformed_data, index=df.index, columns=onehotencoder.get_feature_names_out())

    # now concatenate the original data and the encoded data using pandas
    concatenated_data = pd.concat([df, encoded_data], axis=1)
    df = concatenated_data.drop(columns=categorical_cols)

    # Scale X with a standard scaler
    scaler = joblib.load(f"../models/scaler.pkl")
    df: DataFrame = pd.DataFrame(scaler.transform(df), columns=df.columns)
    logger.info("Web user data processed successfully")

    return df


