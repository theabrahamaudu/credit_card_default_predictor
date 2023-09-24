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
    data = preprocess.undersample_by_value_counts(df,
                                                  'default.payment.next.month')

    zeros = data[data['default.payment.next.month']==0]

    assert len(zeros) == 4, "Unexpexted behaviour"


def test_undersample_by_value_counts_oneclass():
    data = preprocess.undersample_by_value_counts(df2,
                                                  'default.payment.next.month')

    zeros = data[data['default.payment.next.month']==0]

    assert len(zeros) == 23, "Unexpexted behaviour"
