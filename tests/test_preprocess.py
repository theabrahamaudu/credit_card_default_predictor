from utils import preprocess
import pandas as pd

data = {'default.payment.next.month': [1, 0, 0, 0, 1, 0]}
df = pd.DataFrame(data)


def test_class_split():
    ones, zeros = preprocess.class_split(df)

    assert len(ones) == 2
    assert len(zeros) == 4
    for i in ones['default.payment.next.month']:
        assert i == 1
    for i in zeros['default.payment.next.month']:
        assert i == 0
