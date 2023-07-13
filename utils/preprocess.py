"""
This module is used to preprocess raw input data and user data uploaded on the web UI
"""

import pandas as pd
from pandas import DataFrame
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from utils.pipeline_log_config import pipeline as logger


def undersample_by_value_counts(data, label_column):
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

    return randomized_data.astype('int64')


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

    # Drop ID
    data = data.drop('ID', axis=1)

    # Get categorical features
    categorical_features = [feature for feature in data.columns if data[feature].nunique() < 20\
                             and feature != 'default.payment.next.month']
    # Save categorical features
    cat_featutes_path = "./models/cat_features.pkl"
    joblib.dump(categorical_features, cat_featutes_path)

    # Perform one-hot encoding
    data_temp = data.copy()
    data_to_encode = data_temp[categorical_features]

    # Initialize and fit encoder 
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoder.fit(data_to_encode)

    # Save encoder
    encoder_path = "./models/encoder.pkl"
    joblib.dump(encoder, encoder_path)
    logger.info("Encoder saved successfully")

    # Transform data
    encoder = joblib.load(encoder_path)
    encoded_columns = encoder.transform(data_to_encode)
    encoded_df = pd.DataFrame(encoded_columns, 
                              columns=encoder.get_feature_names_out(categorical_features))

    data_temp.drop(categorical_features, axis=1, inplace=True)
    data_encoded = pd.concat([data_temp, encoded_df], axis=1)

    # Undersample the data
    undersampled_data_encoded = undersample_by_value_counts(data_encoded,
                                                            'default.payment.next.month')

    # split df into x,y
    X = undersampled_data_encoded.drop('default.payment.next.month', axis=1) # Inputs
    y = undersampled_data_encoded['default.payment.next.month'] # Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42, 
                                                        stratify=y,
                                                        shuffle=True)

    # Fit StandardScaler
    scaler = StandardScaler().fit(X_train)
    scaler_path = f"./models/scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info("Scaler saved successfully")

    # Scale input data
    scaler = joblib.load(scaler_path)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


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

    # Drop ID
    data = data.drop('ID', axis=1)

    # Load categorical features
    cat_featutes_path = "./models/cat_features.pkl"
    categorical_features = joblib.load(cat_featutes_path)
    
    # Perform one-hot encoding
    data_temp = data.copy()
    data_to_encode = data_temp[categorical_features]

    # Transform data
    encoder_path = "./models/encoder.pkl"
    encoder = joblib.load(encoder_path)
    encoded_columns = encoder.transform(data_to_encode)
    encoded_df = pd.DataFrame(encoded_columns, 
                              columns=encoder.get_feature_names_out(categorical_features))

    data_temp.drop(categorical_features, axis=1, inplace=True)
    data_encoded = pd.concat([data_temp.reset_index(drop=True), encoded_df], axis=1)
  
    # Scale inpputs with a StandardScaler
    scaler_path = f"./models/scaler.pkl"
    scaler = joblib.load(scaler_path)
    scaled_data = scaler.transform(data_encoded)
    logger.info("Web user data processed successfully")

    return scaled_data


