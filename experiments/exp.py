import numpy as np
import pandas as pd
from pandas import DataFrame
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()
from imblearn.over_sampling import SMOTE
import joblib
from exp_log_config import exp as logger


from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import matthews_corrcoef, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv(r"C:\Users\Abraham Audu\Documents\Py-Self-Learn\credit_card_default_prediction\data\UCI_Credit_Card.csv")


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


def under_sample_dataset(df: DataFrame):
    """
    Used to under-sample credit card dataset with defaulters (1) as minority
    Args:
        df (DataFrame):

    Returns:
        df_train_under (DataFrame):
    """
    class_1, class_0 = class_split(df)
    class_0_under = class_0.sample(len(class_1), random_state=234)
    df_train_under = pd.concat([class_0_under, class_1], axis=0)
    df_train_under: DataFrame = df_train_under.sample(frac=1, random_state=234)

    return df_train_under


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
    joblib_file = f"encoder.pkl"
    joblib.dump(onehotencoder, joblib_file)
#     logger.info("Encoder saved successfully")
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
#     logger.info("Scaler saved successfully")

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

    onehotencoder = joblib.load('encoder.pkl')

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
    scaler = joblib.load(f"scaler.pkl")
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


model_params = {
    'Logistic Regression': {
        'model': LogisticRegression(),
        'params': {
            'C': list(range(1, 61, 5)),
            'solver': ['liblinear', 'newton-cg']
        }
    },

    'C-Support Vector Classification': {
        'model': SVC(),
        'params': {
            'C': list(range(1, 61, 5)),
            'kernel': ['poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto']
        }
    },

    'Neural Network (Multi-layer Perceptron classifier)': {
        'model': MLPClassifier(),
        'params': {
            'hidden_layer_sizes': list(range(100, 1000, 50)),
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'max_iter': list(range(200, 1000, 50))
        }
    },

    'Random Forest': {
        'model': RandomForestClassifier(),
        'params': {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_features': ["sqrt", "log2", None]
        }
    }
}

if __name__ == "__main__":

    # Uncomment line 216 to 219 to use Normal data
    # X, y = preprocess_input(data)

    # Train-Test Split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 123)


    # Uncomment line 223 to 230 to use under-sampled data
    # split the raw data
    train_data = data.sample(frac=0.7, random_state=234)
    test_data = data.drop(train_data.index)

    # # under-sample the train dataset
    # df_train_undersampled = under_sample_dataset(train_data)

    # # over-sample the train dataset
    # df_train_oversampled = over_sample_dataset(train_data)
    #
    # SMOTE over-sample the train dataset
    SMOTE_df_trainoversampled = SMOTE_oversample_dataset(train_data)

    X_train, y_train, X_test, y_test = train_test_preprocess(SMOTE_df_trainoversampled, test_data)

    scores = []

    dataset_version = "SMOTE Over-Sampled"

    # # GridSearchCV
    # logger.info("GridSearchCV\n"
    #             "============")
    # for model_name, mp in model_params.items():
    #     run = GridSearchCV(mp['model'], mp['params'], return_train_score=False, n_jobs=-1, scoring='f1', verbose=2)
    #     run.fit(X_train, y_train)
    #     scores.append({
    #         'model': model_name,
    #         'best_score': run.best_score_,
    #         'best_params': run.best_params_
    #     })
    #     logger.info(f"model: {model_name},\n"
    #                 f"best_score: {run.best_score_},\n"
    #                 f"best_params: {run.best_params_}\n")

    # RandomizedSearchCV
    logger.info(f"RandomizedSearchCV --> {dataset_version} REAL\n"
                "==================")
    for model_name, mp in model_params.items():
        run = RandomizedSearchCV(mp['model'], mp['params'], return_train_score=False,
                                 n_jobs=-1, scoring='f1', verbose=3, n_iter=10)
        run.fit(X_train, y_train)

        y_true = y_test.copy()
        y_pred = run.predict(X_test)

        accuracy = run.score(X_test, y_test)
        MCC = matthews_corrcoef(y_true, y_pred)
        F1_SCORE = f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None,
                            zero_division='warn')
        print(f"{model_name}: \naccuracy --> {accuracy} \nMCC --> {MCC} \nf1_score --> {F1_SCORE}\n")

        scores.append({
            'model': model_name,
            'best_score': run.best_score_,
            'best_params': run.best_params_
        })
        logger.info(f"model: {model_name}, --> {dataset_version} REAL\n"
                    f"best_score: {run.best_score_},\n"
                    f"best_params: {run.best_params_}\n")