"""
This module is used to validate customer credit card data uploaded on the web UI
"""
from pandas import DataFrame
from utils.frontend_log_config import frontend as logger


def validate(df: DataFrame):
    """
    Takes dataframe as input and checks dataframe columns and column data types w.r.t training data

    Ensures that uploaded data contains the required columns named accordingly and with the right data type
    in each column.

    Raises an exception if:
    - data has more columns than expected
    - data has fewer columns than expected
    - data does not follow column naming convention
    - data type of columns do not match
    Args:
        df(DataFrame): credit card user data

    Returns:
        str: validation message as "validated" or the specific error based on the checks.
    """
    types_dict = {'ID': 'int64',
                  'LIMIT_BAL': 'float64',
                  'SEX': 'int64',
                  'EDUCATION': 'int64',
                  'MARRIAGE': 'int64',
                  'AGE': 'int64',
                  'PAY_0': 'int64',
                  'PAY_2': 'int64',
                  'PAY_3': 'int64',
                  'PAY_4': 'int64',
                  'PAY_5': 'int64',
                  'PAY_6': 'int64',
                  'BILL_AMT1': 'float64',
                  'BILL_AMT2': 'float64',
                  'BILL_AMT3': 'float64',
                  'BILL_AMT4': 'float64',
                  'BILL_AMT5': 'float64',
                  'BILL_AMT6': 'float64',
                  'PAY_AMT1': 'float64',
                  'PAY_AMT2': 'float64',
                  'PAY_AMT3': 'float64',
                  'PAY_AMT4': 'float64',
                  'PAY_AMT5': 'float64',
                  'PAY_AMT6': 'float64'}
    for df_column in df.columns:
        if df_column not in types_dict.keys():
            logger.error(f"Column '{df_column}' not expected in uploaded data")
            msg = f"Column '{df_column}' not expected in uploaded data"
            return msg

        else:
            msg = "validated"

    for column, expected_dtype in zip(types_dict.keys(), types_dict.values()):
        if column not in df.columns:
            logger.error(f"No column '{column}' in uploaded data")
            msg = f"No column '{column}' in uploaded data"
            return msg

        elif str(expected_dtype) not in str(df[column].dtype):
            logger.error(f"'{column}' column data type does not match required data type. Expected {expected_dtype}")
            msg = f"'{column}' column data type does not match required data type. Expected {expected_dtype}"
            return msg
        else:
            msg = "validated"
    
    return msg

    