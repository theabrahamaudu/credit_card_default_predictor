"""
This module is used to preprocess raw input data and user data uploaded on the web UI
"""

import pandas as pd
import numpy as np
from pandas import DataFrame
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from utils.pipeline_log_config import pipeline as logger


def get_categotical_features(data: DataFrame, train: bool):
    """_summary_

    Args:
        data (DataFrame): _description_
        train (bool): _description_

    Returns:
        _type_: _description_
    """

    cat_featutes_path = "./models/cat_features.pkl"

    if train:
        # Get categorical features
        categorical_features = [feature for feature in data.columns if data[feature].nunique() < 20\
                                and feature != 'default.payment.next.month']
        # Save categorical features
        joblib.dump(categorical_features, cat_featutes_path)

    else:
        # Load categorical features
        categorical_features = joblib.load(cat_featutes_path)

    return categorical_features


def fit_encoder(data: DataFrame, categorical_features: list, train: bool):
    """_summary_

    Args:
        data (DataFrame): _description_
        categorical_features (list): _description_
        train (bool): _description_

    Returns:
        _type_: _description_
    """

    # Set encoder path
    encoder_path = "./models/encoder.pkl"

    # Get categotical features
    data_to_transform = data[categorical_features]

    if train:
        # Initialize and fit encoder 
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoder.fit(data_to_transform)

        # Save encoder
        joblib.dump(encoder, encoder_path)
        logger.info("Encoder saved successfully")

    return data_to_transform   


def transform_data(data: DataFrame,
                   data_to_transform: DataFrame,
                   categorical_features: list):
    """_summary_

    Args:
        data (DataFrame): _description_
        data_to_encode (DataFrame): _description_
        categorical_features (list): _description_

    Returns:
        _type_: _description_
    """

    # Set encoder path
    encoder_path = "./models/encoder.pkl"

    # Load encoder
    encoder = joblib.load(encoder_path)

    # Transform data with encoder
    encoded_columns = encoder.transform(data_to_transform)
    encoded_df = pd.DataFrame(encoded_columns, 
                            columns=encoder.get_feature_names_out(categorical_features))
    # Remove old versions of transformed columns
    data_transformed = data
    data_transformed.drop(categorical_features, axis=1, inplace=True)

    # Add transformed columns to the dataframe
    data_transformed[encoder.get_feature_names_out(categorical_features)] = \
        encoded_df [encoder.get_feature_names_out(categorical_features)].iloc[0]
    

    return data_transformed


def scale_data(features: DataFrame, train: bool):
    """_summary_

    Args:
        features (DataFrame): _description_
        train (bool): _description_

    Returns:
        _type_: _description_
    """

    # Set scaler path
    scaler_path = f"./models/scaler.pkl"

    if train:
        # Fit scaler to features
        scaler = StandardScaler().fit(features)

        # Save scaler
        joblib.dump(scaler, scaler_path)
        logger.info("Scaler saved successfully")

        # Scale train features
        scaler = joblib.load(scaler_path)
        features_scaled = scaler.transform(features)
        logger.info("features scaled successfully")

    else:
        # Load scaler
        scaler = joblib.load(scaler_path)

        # Scale inference features
        features_scaled = scaler.transform(features)
        logger.info("features scaled successfully")

    return features_scaled


def undersample_by_value_counts(data: DataFrame, label_column: str):
    """_summary_

    Args:
        data (DataFrame): _description_
        label_column (str): _description_

    Returns:
        _type_: _description_
    """

    value_counts = data[label_column].value_counts()
    mean_count = value_counts.mean()

    undersampled_data = pd.DataFrame(columns=data.columns)

    for value, count in value_counts.items():
        if count > mean_count:
            undersampled_count = int((count / value_counts.sum()) * (mean_count/2))
            subset = data[data[label_column] == value].sample(n=undersampled_count, random_state=42)
            undersampled_data = pd.concat([undersampled_data, subset], ignore_index=True)
        else:
            subset = data[data[label_column] == value]
            undersampled_data = pd.concat([undersampled_data, subset], ignore_index=True)

    # Randomize the undersampled data
    randomized_data = undersampled_data.sample(frac=1, random_state=42)

    # Typecast label column to int64
    randomized_data[label_column] = randomized_data[label_column].astype('int64')

    return randomized_data


def preprocess_train_input(raw_data: DataFrame):
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
    data = raw_data.copy()

    logger.info("Starting train data preprocess")
    try:
        # Drop ID
        data = data.drop('ID', axis=1)

        # Get categorical features
        categorical_features = get_categotical_features(data=data, train=True)

        # Perform one-hot encoding
        data_to_transform = fit_encoder(data=data,
                                        categorical_features=categorical_features,
                                        train=True)

        # Transform data
        transformed_data = transform_data(data=data,
                                        data_to_transform=data_to_transform,
                                        categorical_features=categorical_features)

        # Undersample the data
        undersampled_data_encoded = undersample_by_value_counts(transformed_data,
                                                                'default.payment.next.month')

        # Split data into into features and target
        X = undersampled_data_encoded.drop('default.payment.next.month', axis=1) # Inputs
        y = undersampled_data_encoded['default.payment.next.month'] # Target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                            random_state=42, 
                                                            stratify=y,
                                                            shuffle=True)

        # Scale input data
        X_train_scaled = scale_data(features=X_train, train=True)
        X_test_scaled = scale_data(features=X_test, train=False)
        logger.info('Train data preprocessed succesfully')
        return X_train_scaled, X_test_scaled, y_train, y_test
    except Exception as e:
        logger.error(f"Error preprocessing train data: {e}")
        raise e
    

def preprocess_inference_input(raw_data: DataFrame):
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
    data = raw_data.copy()

    logger.info("Starting web user data preprocessing")
    try:
        # Drop ID
        data = data.drop('ID', axis=1)

        # Get categorical features
        categorical_features = get_categotical_features(data=data, train=False)
        
        # Perform one-hot encoding
        data_to_transform = fit_encoder(data=data,
                                        categorical_features=categorical_features,
                                        train=False)

        # Transform data
        transformed_data = transform_data(data=data,
                                        data_to_transform=data_to_transform,
                                        categorical_features=categorical_features)
    
        # Scale inputs with a StandardScaler
        scaled_data = scale_data(features=transformed_data, train=False)
        logger.info("Web user data preprocessed successfully")
        return scaled_data
    except Exception as e:
        logger.error(f"Error preprocessing web user data: {e}")



