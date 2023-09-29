import sys
import os
sys.path.append(f"{os.getcwd()}")
from utils import upload_validator
from utils.frontend_log_config import frontend as logger
import pandas as pd


data = pd.read_csv("./data/sample_user_data.csv")



def test_valid_data(caplog):
    """
    Test the data validation process using upload_validator.

    This function tests the data validation process using the upload_validator on the 'data' variable. It checks whether
    the validation result is "validated" and whether there are no log messages captured by the caplog object.

    Args:
        caplog (pytest.LogCaptureFixture): A Pytest fixture for capturing log messages.

    Raises:
        AssertionError: If the validation result is not "validated" or if log messages are present.

    Note:
    - The 'data' variable should contain the data to be validated.
    - The function assumes that the validation process returns "validated" on successful validation.
    """

    validation = upload_validator.validate(data)
    
    log_msg = caplog.text
    assert validation == "validated", "Unexpected behaviour"
    assert log_msg is '', "Unexpected behaviour"

def test_invalid_data(caplog):
    """
    Test the data validation process using upload_validator with invalid data.

    This function tests the data validation process using the upload_validator on the 'invalid_data' variable, which is
    created by copying 'data' and adding an unexpected column 'IF'. It checks whether the validation result is the
    expected error message and whether the error message is present in the captured log messages.

    Args:
        caplog (pytest.LogCaptureFixture): A Pytest fixture for capturing log messages.

    Raises:
        AssertionError: If the validation result is not the expected error message or if the error message is not
        present in the log messages.

    Note:
    - The 'invalid_data' variable should contain the data with an unexpected column.
    - The function assumes that the validation process returns an error message when unexpected columns are present.
    """

    invalid_data = data.copy()
    invalid_data['IF'] = invalid_data['ID']

    validation = upload_validator.validate(invalid_data)

    log_msg = caplog.text
    assert validation == "Column 'IF' not expected in uploaded data", "Unexpected behaviour"
    assert "Column 'IF' not expected in uploaded data" in log_msg, "Unexpected behaviour"


def test_incomplete_data(caplog):
    """
    Test the data validation process using upload_validator with incomplete data.

    This function tests the data validation process using the upload_validator on the 'incomplete_data' variable, which is
    created by copying 'data' and then dropping the 'AGE' column. It checks whether the validation result is the expected
    error message and whether the error message is present in the captured log messages.

    Args:
        caplog (pytest.LogCaptureFixture): A Pytest fixture for capturing log messages.

    Raises:
        AssertionError: If the validation result is not the expected error message or if the error message is not
        present in the log messages.

    Note:
    - The 'incomplete_data' variable should contain the data with a missing 'AGE' column.
    - The function assumes that the validation process returns an error message when expected columns are missing.
    """

    incomplete_data = data.copy()
    incomplete_data = incomplete_data.drop('AGE', axis=1)

    validation = upload_validator.validate(incomplete_data)

    log_msg = caplog.text
    assert validation == "No column 'AGE' in uploaded data", "Unexpected behaviour"
    assert "No column 'AGE' in uploaded data" in log_msg, "Unexpected behaviour"


def test_invalid_datatype(caplog):
    """
    Test the data validation process using upload_validator with data containing an invalid data type.

    This function tests the data validation process using the upload_validator on the 'invalid_datatype' variable, which is
    created by copying 'data' and changing the data type of the 'AGE' column to 'object'. It checks whether the validation
    result contains the expected error message about the data type mismatch and whether the error message is present in
    the captured log messages.

    Args:
        caplog (pytest.LogCaptureFixture): A Pytest fixture for capturing log messages.

    Raises:
        AssertionError: If the validation result does not contain the expected error message or if the error message is not
        present in the log messages.

    Note:
    - The 'invalid_datatype' variable should contain the data with an 'AGE' column of an incorrect data type.
    - The function assumes that the validation process returns an error message when data type mismatches occur.
    """

    invalid_datatype = data.copy()
    invalid_datatype['AGE'] = invalid_datatype['AGE'].astype('object')

    validation = upload_validator.validate(invalid_datatype)

    log_msg = caplog.text
    assert "does not match required data type. Expected int64" in validation, "Unexpected behaviour"
    assert "does not match required data type. Expected int64" in log_msg, "Unexpected behaviour"