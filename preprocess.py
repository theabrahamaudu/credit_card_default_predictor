import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def preprocess_input(df):
    df = df.copy()

    # Drop ID
    df = df.drop('ID', axis = 1)

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
    y = df['default.payment.next.month'].copy()
    X = df.drop('default.payment.next.month', axis = 1).copy()

    # Scale X with a standard scaler
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    joblib_file = f"scaler.pkl"
    joblib.dump(scaler, joblib_file)
    print("Scaler saved successfully")

    return X, y


def onehot_encode(df, column_dict):
    df = df.copy()

    for column, prefix in column_dict.items():
        dummies = pd.get_dummies(df[column], prefix=prefix)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(column, axis=1)
    return df


def preprocess_website_input(df):
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
    df = pd.DataFrame(scaler.transform(df), columns=df.columns)

    return df