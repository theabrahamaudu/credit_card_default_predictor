"""
This module contains the API service for predicting credit card default based on data supplied by user on web UI
"""

import joblib
from fastapi import FastAPI
import uvicorn
import pandas as pd
from pydantic import BaseModel
import time
from utils.preprocess import preprocess_website_input
from logs.backend_log_config import backend as logger


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

    logger.info("loading selected model")
    if model_str == 'Logistic Regression':
        model = joblib.load('../models/LogisticRegression().pkl')
    elif model_str == 'Support Vector Machine':
        model = joblib.load('../models/SVC().pkl')
    elif model_str == 'Neural Network':
        model = joblib.load('../models/MLPClassifier().pkl')
    elif model_str == 'Random Forest':
        model = joblib.load('../models/RandomForestClassifier().pkl')
    logger.info("model loaded successfully")

    logger.info("loading and preprocessing web UI data")

    try:
        data = pd.DataFrame(data["customer_data"])
        data = preprocess_website_input(data)
    except:
        logger.exception("Error preprocessing data")

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
    except:
        logger.exception("Error generating prediction")
    return packet


if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)