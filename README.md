# Credit Card Default Predictor

This app is able to take user credit card data and make a prediction on whether the user will default
on their next credit card payment or not.
It can be used by interacting with the web UI.

Try it here: https://creditcarddefaultapp.streamlit.app/ [Not always up]


Backend server deployed here: https://credit-card-predictor-api.onrender.com [Not always up]

## Description

The prediction infrastructure of this project was built using a [credit card dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) from kaggle.
The dataset was then preprocessed to deal with categorical variables, under-sampled to deal with data imbalance, as well as scaling, to bring all the features within a narrow range for better performance.
Four models were trained on the dataset:

* Logistic Regression
* Neural Network
* Random Forest
* Support Vector Machine

## Model Performance

- [ ] Update: After refining preprocessing pipeline:
  Date: 13/07/2023
~~~
2023-07-13 17:24:33,606:models.py:test_models:INFO:Logistic_Regression:
accuracy --> 0.6923076923076923 
MCC --> 0.40260152462276916 
f1_score --> 0.6745762711864408
:
2023-07-13 17:24:44,284:models.py:test_models:INFO:C_Support_Vector_Classification:
accuracy --> 0.6963141025641025 
MCC --> 0.41053255003272965 
f1_score --> 0.6790855207451312
:
2023-07-13 17:24:44,332:models.py:test_models:INFO:Neural_Network_(Multi_layer_Perceptron_classifier):
accuracy --> 0.6782852564102564 
MCC --> 0.3543679248291287 
f1_score --> 0.6961785849413544
:
2023-07-13 17:24:44,988:models.py:test_models:INFO:Random_Forest:
accuracy --> 0.702323717948718 
MCC --> 0.4112599202179095 
f1_score --> 0.7010060362173037
~~~
All the models as well as the ecoder and scaler were saved to memory so as to enable the transformation of new data from end users before prediction on the data.

The app has a prediction API built with FastAPI and a frontend built with Streamlit

## Basic Workflow

- Perform EDA
- Preprocess raw data and save encoder and scaler
- Train models and save models
- Test models
- Setup API service
- Design frontend

## API Structure

The API service has two endpoints:

* Home: Used to test that the API is running
* Predict: API service has a prediction endpoint which:

  - Takes in data from the web UI
  - Load prediction model selected by user
  - Convert user credit card JSON data to DataFrame
  - Make prediction based on credit card data
  - Return predicted outcome as float

## Frontend Structure

The frontend is built with Streamlit.How it works:

- Promts the user to upload customer credit card data as .csv file
- Validates data and displays error message if uploaded data does not match model requirements
- Allows the user to select a preferred prediction model only if the right file type has been uploaded
- Makes a post request to the prediction endpoint with data provided by user
- Displays a formatted version of the request response

## Getting Started

### Dependencies

* Windows 10
* Python 3.10
* fastapi==0.88.0
* joblib==1.2.0
* numpy==1.23.5
* pandas==1.5.2
* pydantic==1.10.2
* scikit-learn==1.2.0
* streamlit==1.16.0
* uvicorn==0.20.0

### Installing

To test this project on your local machine:

* Create a new project
* Clone this repo by running:

```
git clone https://github.com/theabrahamaudu/credit_card_default_predictor.git
```

* install the requirements by running:

```
python -m pip install -r requirements.txt
```

### Executing program

* Start the API service by running:

```
python -m src/main
```

* Run the Streamlit frontend:

```
streamlit run src/streamlit_frontend.py
```

* To test the app, use the "sample_user_data.csv" in the data directory as customer data

## Help

Feel free to reach out to me if you have any issues using the platform

## Improvements in the Works

- [ ] Deployment to cloud (Blocked)

## Authors

Contributors names and contact info

*Abraham Audu*

* GitHub - [@the_abrahamaudu](https://github.com/theabrahamaudu)
* LinkedIn - [@theabrahamaudu](https://www.linkedin.com/in/theabrahamaudu/)
* Instagram - [@the_abrahamaudu](https://www.instagram.com/the_abrahamaudu/)

## Version History

* See [commit change](https://github.com/theabrahamaudu/credit_card_default_predictor/commits/main)
* See [release history](https://github.com/theabrahamaudu/credit_card_default_predictor/releases)

## Acknowledgments

* This project was inspired by the iNeuron internship portal
* This [video](https://www.youtube.com/watch?v=kn5hVBR40eo) gave me a lot of guidance in building the models
* Streamlit official documentation was handdy in setting up the file uploader
* I got a lot of help from different websites in debugging at different stages (StackOverflow, etc)
