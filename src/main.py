"""
This module contains the API service for predicting credit card default based on data supplied by user on web UI
"""

import sys
import os
sys.path.append(f"{os.getcwd()}")
import joblib
from fastapi import FastAPI
import uvicorn
import pandas as pd
import numpy as np
from pydantic import BaseModel
import time
from utils.preprocess import preprocess_inference_input
from utils.backend_log_config import backend as logger


logger.info("API service running")

# Initialize FastAPI
app = FastAPI(title='Credit Card Default Predictor',
              version='1.0',
              description='Multiple models are used for prediction'
              )


# Data Validation
class Data(BaseModel):
    model: str


# API home endpoint
@app.get('/')
@app.get('/home')
def read_home():
    """
    Home endpoint which can be used to test the availability of the application.
    """
    logger.info("API service tested")
    return {'message': 'System is healthy'}


# Prediction endpoint
@app.post("/predict")
def predict(data: dict):
    """
    Takes dictionary containing specification of desired prediction model and user credit card data as input and
    returns prediction result as output.
    Args:
        data(dict): data from web UI

    Returns:
        result(float): float value of percentage default probability
    """
    logger.info("prediction request received")
    logger.info("detecting selected model")
    model_str = data['model']
    try:
        logger.info("loading selected model")
        if model_str == 'Logistic Regression':
            model = joblib.load('./models/Logistic_Regression.pkl')
        elif model_str == 'Support Vector Machine':
            model = joblib.load('./models/C_Support_Vector_Classification.pkl')
        elif model_str == 'Neural Network':
            model = joblib.load('./models/Neural_Network_(Multi_layer_Perceptron_classifier).pkl')
        elif model_str == 'Random Forest':
            model = joblib.load('./models/Random_Forest.pkl')
        logger.info("model loaded successfully")
    except Exception as e:
        logger.error(f"error loading model: {e}")

    logger.info("loading and preprocessing web UI data")

    try:
        data = pd.DataFrame(data["customer_data"])
        data = preprocess_inference_input(data)
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")

    try:
        if model_str == 'Support Vector Machine':
            start_time = time.perf_counter()
            result: float = float(model.predict(data)[0])
            elapsed_time = time.perf_counter()-start_time
            packet = {"result": result, "time": elapsed_time}
        else:
            start_time = time.perf_counter()
            result: float = float(model.predict_proba(data)[0][1] * 100)
            elapsed_time = time.perf_counter() - start_time
            packet = {"result": result, "time": elapsed_time}

        logger.info("sending result to frontend")
        return packet
    except Exception as e:
        logger.exception(f"Error generating prediction: {e}")
        return {"result": None, "time": None}


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)