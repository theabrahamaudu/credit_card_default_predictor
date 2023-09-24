import sys
import os
sys.path.append(f"{os.getcwd()}")
from utils import upload_validator
from utils.frontend_log_config import frontend as logger
import pandas as pd


data = pd.read_csv("./data/sample_user_data.csv")



def test_valid_data(caplog):
    validation = upload_validator.validate(data)
    
    log_msg = caplog.text
    assert validation == "validated", "Unexpected behaviour"
    assert log_msg is '', "Unexpected behaviour"

def test_invalid_data(caplog):
    invalid_data = data.copy()
    invalid_data['IF'] = invalid_data['ID']


    validation = upload_validator.validate(invalid_data)

    log_msg = caplog.text
    assert validation == "Column 'IF' not expected in uploaded data", "Unexpected behaviour"
    assert "Column 'IF' not expected in uploaded data" in log_msg, "Unexpected behaviour"


def test_incomplete_data(caplog):
    incomplete_data = data.copy()
    incomplete_data = incomplete_data.drop('AGE', axis=1)

    validation = upload_validator.validate(incomplete_data)

    log_msg = caplog.text
    assert validation == "No column 'AGE' in uploaded data", "Unexpected behaviour"
    assert "No column 'AGE' in uploaded data" in log_msg, "Unexpected behaviour"


def test_invalid_datatype(caplog):
    invalid_datatype = data.copy()
    invalid_datatype['AGE'] = invalid_datatype['AGE'].astype('object')

    validation = upload_validator.validate(invalid_datatype)

    log_msg = caplog.text
    assert "does not match required data type. Expected int64" in validation, "Unexpected behaviour"
    assert "does not match required data type. Expected int64" in log_msg, "Unexpected behaviour"