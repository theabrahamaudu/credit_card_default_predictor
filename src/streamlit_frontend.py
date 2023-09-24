"""
This module houses the Streamlit frontend for the app.

It allows the user to upload sample credit card data as .csv file and choose which prediction model to use
from a range of pretrained models.

The result is presented as percentage probability to default on credit card payment
"""

import sys
import os
sys.path.append(f"{os.getcwd()}")
import pandas as pd
import streamlit as st
import requests
from models import models_dict
from utils.frontend_log_config import frontend as logger
from utils.upload_validator import validate


def run():
    """
    Streamlit configuration for Credit Card Default Prediction web user interface

    - Allows user to upload sample credit card data
    - Allows user to select prediction model to be used
    - Sends request to backend API for prediction and then displays result
    """
    logger.info("Session started")
    st.set_page_config(page_title="Credit Card Default App",
                        page_icon="💳")
    st.image("https://www.canstar.com.au/wp-content/uploads/2017/09/Credit-card-default-1.jpg")
    st.title(" Customer Credit Card Default Predictor")
    st.subheader("Predict the probability of a customer defaulting on Credit Card payment\n")

    st.sidebar.text("Steps:\n"
                    "1. Upload customer data\n"
                    "2. Choosel ML model\n"
                    "3. Get prediction")

    file = st.file_uploader("Upload customer data (CSV)", type=['csv'])
    valid = False

    if file is not None:
        logger.info("User data uploaded")
        customer_data = pd.read_csv(file)

        try:
            msg = validate(customer_data)

            if msg is "validated":
                logger.info("Uploaded data validated")
                valid = True
            else:
                st.error(f"Invalid data: {msg}")
        except Exception as e:
            st.error(f"Error validating data: {e}")

    if valid:
        customer_data = customer_data.to_dict()
        model = st.selectbox("Choose Prediction Model", sorted(models_dict.values()))

        data = {
            "customer_data": customer_data,
            "model": model
            }

        if st.button("Predict default probability"):
            try:
                logger.info("Attempting API call")
                import time
                start_time = time.perf_counter()

                response = requests.post("http://127.0.0.1:8000/predict", json=data).json()

                prediction = response["result"]
                pred_time = response["time"]

                elapsed_time = time.perf_counter() - start_time

                st.success(f"Probability of Credit Card default: {float(prediction):.2f}%")
                st.success(f"Model prediction time: {pred_time:.3f}s\n")
                st.success(f"Overall run time: {elapsed_time:.3f}s")

                logger.info("Prediction successful, \n"
                            f"Model used: {model}, \n"
                            f"Credit Card Default probability: {prediction}, \n"
                            f"Model pred time: {pred_time}, \n"
                            f"total run time: {elapsed_time}")
            except:
                st.error("Error: Please check the file or your network connection")
                logger.exception("An error occurred whilst attempting to call API")


if __name__ == "__main__":
        run()