from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier


import processing.preprocessors as pp
import yaml

#read yaml file
with open('config.yaml') as file:
  config= yaml.safe_load(file)


pipe = Pipeline([
    ('num_imputer', pp.Imputer(config['NUMERICAL_FEATURES'], method='mean')),
    ('scaler', pp.Scaler(config['NUMERICAL_FEATURES'])),
    ('cat_imputer', pp.Imputer(config['CATEGORICAL_FEATURES'])),
    ('encoder', pp.Encoder(config['CATEGORICAL_FEATURES'])),
    ('model', pp.LogisticRegression())
])

