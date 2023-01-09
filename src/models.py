"""
This module is used to train and test the ML models used on the dataset

It saves trained models to local storage. Thus, as the dataset expands (for real life situations), the models can be
retrained by running this script to account for data drift.
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
from utils.preprocess import (preprocess_input,
                              over_sample_dataset,
                              train_test_preprocess,
                              SMOTE_oversample_dataset)

from utils.pipeline_log_config import pipeline as logger


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

    for model, name in models.items():
        model.fit(X_train, y_train)
        joblib_file = f"../models/{name}.pkl"
        joblib.dump(model, joblib_file)
        logger.info(f"{name} trained and saved")
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
    for mdl, name in models.items():
        model = joblib.load(f'../models/{name}.pkl')
        y_true = y_test.copy()
        y_pred = model.predict(X_test)
        accuracy = model.score(X_test, y_test)
        MCC = matthews_corrcoef(y_true, y_pred)
        F1_SCORE = f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None,
                            zero_division='warn')
        logger.info(f"{name}: \naccuracy --> {accuracy} \nMCC --> {MCC} \nf1_score --> {F1_SCORE}\n")


if __name__=="__main__":
    # # Preprocess the data
    # X, y = preprocess_input(df=pd.read_csv(r"../data/UCI_Credit_Card.csv"))
    #
    # # Train-Test Split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=123)

    data = pd.read_csv(r"../data/UCI_Credit_Card.csv")

    # split the raw data
    train_data = data.sample(frac=0.7, random_state=234)
    test_data = data.drop(train_data.index)

    # over-sample the train dataset
    df_train_oversampled = over_sample_dataset(train_data)

    # custom train-test split
    X_train, y_train, X_test, y_test = train_test_preprocess(df_train_oversampled, test_data)

    # # models dictionary
    # models_dict = {
    #     LogisticRegression(solver='liblinear',
    #                        C=1): 'Logistic_Regression',
    #     SVC(kernel='rbf',
    #         gamma='auto',
    #         C=51): 'C_Support_Vector_Classification',
    #     MLPClassifier(solver='adam',
    #                   max_iter=950,
    #                   hidden_layer_sizes=900,
    #                   activation='tanh'): 'Neural_Network_(Multi_layer_Perceptron_classifier)',
    #     RandomForestClassifier(max_features='log2',
    #                            criterion='gini'): 'Random_Forest'
    # }

    # models dictionary
    models_dict = {
        LogisticRegression(): 'Logistic_Regression',
        SVC(): 'C_Support_Vector_Classification',
        MLPClassifier(): 'Neural_Network_(Multi_layer_Perceptron_classifier)',
        RandomForestClassifier(): 'Random_Forest'
    }
    train_models(models=models_dict, X_train=X_train, y_train=y_train)
    test_models(models=models_dict, X_test=X_test, y_test=y_test)