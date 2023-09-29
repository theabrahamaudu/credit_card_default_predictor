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
    """Extracts list of categorical columns by filtering for columns with
    less than 20 unique values.

    Saves the list to pickle and loads the list for inference preprocessing.

    Args:
        data (DataFrame): full dataset
        train (bool): train/inference toggle

    Returns:
        list: List of columns with categorical data
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
    """Fits encoder to sub-dataset with categorical features if train is true.

    Saves encoder to pickle.

    Args:
        data (DataFrame): full dataset
        categorical_features (list): names of categorical features
        train (bool): train/inference toggle

    Returns:
        DataFrame: sub-dataset with categorical features
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
    """Transforms categorical features with pre-fit encoder from memory.

    Args:
        data (DataFrame): full dataset
        data_to_encode (DataFrame): sub-dataset with categorical features
        categorical_features (list): categorical feature names

    Returns:
        DataFrame: Full dataset with numerical and transformed categorical features
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
    """Fits scaler to train set and saves scaler to memory if train is True.

    Loads scaler from memory to scale inference data.

    Args:
        features (DataFrame): full dataset without targets column
        train (bool): train/inference toggle

    Returns:
        DataFrame: scaled features
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
    """Undersample data points with label count greater than the mean label count of the dataset.

    Undersampling Strategy:
        `(label count / total label count) * mean label count`

    This ensures that the overpopulated labels are trimmed proportionally, as opposed to 
    trimming all oversampled points to a fixed number, thus retaining the underlying
    difference in frequency, but still preventing excessive skew in distribution.

    Args:
        data (DataFrame): DataFrame to be undersampled
        label_column (str): Column holding the labels

    Returns:
        DataFrame: Undersampled DataFrame
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
    Preprocesses raw training data for machine learning models.

    Args:
        raw_data (DataFrame): The raw training data as a Pandas DataFrame.

    Returns:
        tuple: A tuple containing the following elements:
            - X_train_scaled (DataFrame): Scaled and preprocessed features for training.
            - X_test_scaled (DataFrame): Scaled and preprocessed features for testing.
            - y_train (pd.Series): Target labels for training.
            - y_test (pd.Series): Target labels for testing.

    Raises:
        Exception: If any error occurs during the preprocessing process.

    This function performs the following preprocessing steps:
    1. Drops the 'ID' column from the raw data.
    2. Identifies categorical features using the `get_categotical_features` function.
    3. Performs one-hot encoding on the categorical features using the `fit_encoder` function.
    4. Transforms the data using the `transform_data` function.
    5. Undersamples the data based on the 'default.payment.next.month' column using
       `undersample_by_value_counts` function.
    6. Splits the data into training and testing sets using stratified sampling.
    7. Scales the input features using the `scale_data` function.

    Note:
    - The 'default.payment.next.month' column is assumed to be the target variable.
    - The `logger` is used to log informational and error messages during the process.
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
    Preprocesses raw web user data for further analysis or predictions.

    Args:
        raw_data (pd.DataFrame): The raw web user data as a Pandas DataFrame.

    Returns:
        pd.DataFrame: Preprocessed and scaled web user data.

    Raises:
        Exception: If any error occurs during the preprocessing process.

    This function performs the following preprocessing steps:
    1. Drops the 'ID' column from the raw data.
    2. Identifies categorical features using the `get_categotical_features` function.
    3. Performs one-hot encoding on the categorical features using the `fit_encoder` function.
    4. Transforms the data using the `transform_data` function.
    5. Scales the transformed data using a StandardScaler.

    Note:
    - The `logger` is used to log informational and error messages during the process.
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



