import sys
import os
sys.path.append(f"{os.getcwd()}")
from utils import preprocess
import pandas as pd

data = {'default.payment.next.month': [1, 0, 0, 0, 1, 0, 0, 0,
                                       1, 0, 0, 0, 1, 0, 0, 0,
                                       1, 0, 0, 0, 1, 0, 0,
                                       ]}

data2 = {'default.payment.next.month': [0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0,
                                       ]}
df = pd.DataFrame(data)
df2 = pd.DataFrame(data2)


def test_undersample_by_value_counts_multiclass():
    """
    Test the undersample_by_value_counts function with a multiclass target variable.

    This function tests the behavior of the undersample_by_value_counts function when dealing with a multiclass target
    variable. It performs undersampling based on the 'default.payment.next.month' column and checks whether the
    undersampled dataset contains the expected number of samples for one of the classes (class 0).

    Raises:
        AssertionError: If the undersampled dataset does not contain the expected number of samples (4) for class 0.

    Note:
    - The variable 'df' should contain the dataset to be undersampled.
    - The 'default.payment.next.month' column is assumed to be the target variable.
    """

    data = preprocess.undersample_by_value_counts(df,
                                                  'default.payment.next.month')

    zeros = data[data['default.payment.next.month']==0]

    assert len(zeros) == 4, "Unexpexted behaviour"


def test_undersample_by_value_counts_oneclass():
    """
    Test the undersample_by_value_counts function with a multiclass target variable.

    This function tests the behavior of the undersample_by_value_counts function when dealing with a multiclass target
    variable. It performs undersampling based on the 'default.payment.next.month' column and checks whether the
    undersampled dataset contains the expected number of samples for one of the classes (class 0).

    Raises:
        AssertionError: If the undersampled dataset does not contain the expected number of samples (23) for class 0.

    Note:
    - The variable 'df2' should contain the dataset to be undersampled.
    - The 'default.payment.next.month' column is assumed to be the target variable.
    """
    
    data = preprocess.undersample_by_value_counts(df2,
                                                  'default.payment.next.month')

    zeros = data[data['default.payment.next.month']==0]

    assert len(zeros) == 23, "Unexpexted behaviour"
