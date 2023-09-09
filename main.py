
#Import Libraries
import pandas as pd
import numpy as np


import yaml

#read yaml file
with open('config.yaml') as file:
  config= yaml.safe_load(file)
  
from processing.data_management import load_dataset, save_pipeline, load_pipeline
import processing.preprocessors as pp
import pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from functions import calculate_roc_auc
from predict import make_prediction


def run_training():
    """Train the model"""

    #Read Data
    train = load_dataset('titanic.csv')
    
    X_train, X_test, y_train, y_test = train_test_split(train.drop(columns=config['TARGET']), train[config['TARGET']], 
                                                    test_size=.2, random_state=config['SEED'], 
                                                   stratify=train[config['TARGET']])
    pipeline.pipe.fit(X_train, y_train)
    print(f"Train ROC-AUC: {calculate_roc_auc(pipeline.pipe, X_train, y_train):.4f}")
    print(f"Test ROC-AUC: {calculate_roc_auc(pipeline.pipe, X_test, y_test):.4f}")
    save_pipeline(pipeline_to_save=pipeline.pipe)
    
if __name__=='__main__':
    run_training()
    # Test Prediction
    test_data = load_dataset(file_name='titanic.csv')
    make_prediction(test_data)
    print("Prediction Done")



