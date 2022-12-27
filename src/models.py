"""
This module is used to train and test the ML models used on the dataset

It saves trained models to local storage. Thus, as the dataset expands (for real life situations), the models can be
retrained by running this script to account for data drift.
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
from utils.preprocess import preprocess_input
from logs.pipeline_log_config import pipeline as logger


# models dictionary
models_dict = {
    LogisticRegression(): 'Logistic Regression',
    SVC(): 'Support Vector Machine',
    MLPClassifier(): 'Neural Network',
    RandomForestClassifier(): 'Random Forest'
    }


def train_models(models: dict, X_train, y_train):
    """
    Takes dictionary of initialized models, training features and outcomes as input and saves the trained models.

    Within a for loop, each model is fitted to X-train and y_train, with the resulting model being saved to memory
    Args:
        models:
        X_train:
        y_train:

    Returns:

    """

    for model in models.keys():
        model.fit(X_train, y_train)
        joblib_file = f"../models/{model}.pkl"
        joblib.dump(model, joblib_file)
        logger.info(f"{model} trained and saved")
    logger.info("All models trained and saved successfully")


def test_models(models: dict, X_test, y_test):
    """
    Takes dictionary of initialized models, test features and outcomes as input and prints prediction scores to console.

    - Loads saved models using models dictionary for reference purpose
    - Prints test score for each model to console

    Args:
        models:
        X_test:
        y_test:

    Returns:

    """
    # Test models
    for model, name in models.items():
        saved_model = joblib.load(f"../models/{model}.pkl")
        score = saved_model.score(X_test, y_test) * 100
        logger.info(f"{name}: {score:.4f}% test accuracy")


if __name__=="__main__":
    # Preprocess the data
    X, y = preprocess_input(df=pd.read_csv(r"../data/UCI_Credit_Card.csv"))

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=123)

    # models dictionary
    models_dict = {
        LogisticRegression(): 'Logistic Regression',
        SVC(): 'Support Vector Machine',
        MLPClassifier(): 'Neural Network',
        RandomForestClassifier(): 'Random Forest'
    }

    train_models(models=models_dict, X_train=X_train, y_train=y_train)
    test_models(models=models_dict, X_test=X_test, y_test=y_test)