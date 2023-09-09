import pandas as pd
import joblib
import os
import pickle
import yaml

#read yaml file
with open('config.yaml') as file:
  config= yaml.safe_load(file)

def load_dataset(file_name):
    _data = pd.read_csv(file_name)
    return _data

def save_pipeline(pipeline_to_save):
    save_file_name = 'logistic_regression.pkl'
    loaded_model = pickle.dump(pipeline_to_save , open(save_file_name , 'wb'))
    print("Saved Pipeline : ",save_file_name)


def load_pipeline(pipeline_to_load):
    trained_model = pd.read_pickle(pipeline_to_load)
    return trained_model