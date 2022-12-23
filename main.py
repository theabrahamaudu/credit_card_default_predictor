import joblib
from fastapi import FastAPI
import uvicorn
import pandas as pd
from pydantic import BaseModel
from preprocess import preprocess_website_input

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
    Home endpoint which can be used to test the availability of the    application.
    """
    return {'message': 'System is healthy'}


# Prediction endpoint
@app.post("/predict")
def predict(data: dict):
    model_str = data['model']

    if model_str == 'Logistic Regression':
        model = joblib.load('LogisticRegression().pkl')
    elif model_str == 'Support Vector Machine':
        model = joblib.load('SVC().pkl')
    elif model_str == 'Neural Network':
        model = joblib.load('MLPClassifier().pkl')
    elif model_str == 'Random Forest':
        model = joblib.load('RandomForestClassifier().pkl')

    data = pd.DataFrame(data["customer_data"])
    data = preprocess_website_input(data)
    if model_str == 'Support Vector Machine':
        result = float(model.predict(data)[0])
    else:
        result = model.predict_proba(data)[0][1] * 100
    return result


if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)