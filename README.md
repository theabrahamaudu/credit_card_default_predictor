# Credit Card Default Predictor

This app is able to take user credit card data and make a prediction on whether the user will default
on their next credit card payment or not. 
It can be used by interacting with the web UI.

## Description

The prediction infrastructure of this project was built using a [credit card dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) from kaggle.
The dataset was then preprocessed to deal with categorical variables, over-sampled to deal with data imbalance, as well as scaling, to bring all the features within a narrow range for better performance.
Four models were trained on the dataset:
* Logistic Regression
* Neural Network
* Decision Tree
* Support Vector Machine

## Model Performance
- [ ] Update: After implementing Matthews Correlation Coefficient and F1 Score:  
Date: 09/01/2023
  
        2023-01-09 23:50:55,687:models.py:test_models:INFO:Logistic_Regression: 
        accuracy --> 0.6992222222222222 
        MCC --> 0.29831296922282063 
        f1_score --> 0.4795231686214189
        :
        2023-01-09 23:51:54,163:models.py:test_models:INFO:C_Support_Vector_Classification: 
        accuracy --> 0.7768888888888889 
        MCC --> 0.3849156935678977 
        f1_score --> 0.5293014533520862
        :
        2023-01-09 23:51:54,211:models.py:test_models:INFO:Neural_Network_(Multi_layer_Perceptron_classifier): 
        accuracy --> 0.732 
        MCC --> 0.3252159123373819 
        f1_score --> 0.49391523289970624
        :
        2023-01-09 23:51:55,024:models.py:test_models:INFO:Random_Forest: 
        accuracy --> 0.8132222222222222 
        MCC --> 0.40309391623575275 
        f1_score --> 0.5022209061297009


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
The frontend is built with Streamlit.  
How it works:
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
cd src
```
```
python -m main
```
* Run the Streamlit frontend:
```
streamlit run streamlit_frontend.py
```
* To test the app, use the "sample_user_data.csv" in the repository as customer data

## Help

Feel free to reach out to me if you have any issues using the platform

## Improvements in the Works
- [ ] Better feature engineering/selection
- [ ] Unit tests
- [ ] Deployment to cloud
- [ ] Full documentation

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
