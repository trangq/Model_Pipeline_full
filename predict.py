import numpy as np
import pandas as pd 

import yaml

#read yaml file
with open('config.yaml') as file:
  config= yaml.safe_load(file)
from processing.data_management import load_pipeline

pipeline_file_name = 'logistic_regression.pkl'

_price_pipe = load_pipeline(pipeline_file_name)

def make_prediction(input_data):
    data = pd.DataFrame(input_data)
    prediction = _price_pipe.predict(data[config['FEATURES']])
    result = pd.concat([data[config['FEATURES']],pd.DataFrame(prediction)], axis=1)
    result = result.rename(columns={0: 'Prediction'})
    result.to_csv('out.csv')
   
    
    # output = prediction

    # results = {
    #     'prediction': output,
    #     'model_name': pipeline_file_name,
    #     'version':'version1'
    # }

    # return prediction

