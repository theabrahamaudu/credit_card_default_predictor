"""
This module houses the Streamlit frontend for the app.

It allows the user to upload sample credit card data as .csv file and choose which prediction model to use
from a range of pretrained models.

The result is presented as percentage probability to default on credit card payment
"""

import pandas as pd
import streamlit as st
import requests
from models import models_dict


def run():
    """
    Streamlit configuration for Credit Card Default Prediction web user interface

    - Allows user to upload sample credit card data
    - Allows user to select prediction model to be used
    - Sends request to backend API for prediction and then displays result
    """
    st.image("https://www.canstar.com.au/wp-content/uploads/2017/09/Credit-card-default-1.jpg")
    st.title(" Customer Credit Card Default Predictor")
    st.subheader("Predict the probability of a customer defaulting on Credit Card payment\n")

    st.sidebar.text("Steps:\n"
                    "1. Upload customer data\n"
                    "2. Choosel ML model\n"
                    "3. Get prediction")

    file = st.file_uploader("Upload customer data (CSV)", type=['csv'])

    if file is not None:
        customer_data = pd.read_csv(file)
        customer_data = customer_data.to_dict()
        model = st.selectbox("Choose Prediction Model", sorted(models_dict.values()))

        data = {
            "customer_data": customer_data,
            "model": model
            }

        if st.button("Predict default probability"):
            response = requests.post("http://127.0.0.1:8000/predict", json=data)

            prediction = response.text.strip()

            st.success(f"Probability of Credit Card default:\n"
                       f"{float(prediction):.2f}%")


if __name__=="__main__":
        run()