from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
from preprocess import preprocess_input

# models dictionary
models_dict = {
    LogisticRegression(): 'Logistic Regression',
    SVC(): 'Support Vector Machine',
    MLPClassifier(): 'Neural Network',
    RandomForestClassifier(): 'Random Forest'
    }


def train_models(models: dict, X_train, y_train):
    # Train models
    for model in models.keys():
        model.fit(X_train, y_train)
        joblib_file = f"{model}.pkl"
        joblib.dump(model, joblib_file)
        print(f"{model} trained and saved")
    print("All models trained and saved successfully")


def test_models(models: dict, X_test, y_test):
    # Test models
    for model, name in models.items():
        saved_model = joblib.load(f"{model}.pkl")
        print(name, ": {:.4f}%".format(saved_model.score(X_test, y_test) * 100))


if __name__=="__main__":
    # Preprocess the data
    X, y = preprocess_input(df=pd.read_csv(r"UCI_Credit_Card.csv"))

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