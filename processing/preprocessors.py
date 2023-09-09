import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn import set_config


#Local Files

class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, features, method='constant', value='missing'):
        self.features = features
        self.method = method
        self.value = value
    
    def fit(self, X, y=None):
        if self.method=='mean':
            self.value = X[self.features].mean()
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.features] = X[self.features].fillna(self.value)
        return X_transformed
    
class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features
    
    def fit(self, X, y=None):
        self.min = X[self.features].min()
        self.range = X[self.features].max()-self.min
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.features] = (X[self.features]-self.min)/self.range
        return X_transformed
  
class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, features, drop='first'):
        self.features = features
        self.drop = drop
    
    def fit(self, X, y=None):
        self.encoder = OneHotEncoder(sparse=False, drop=self.drop)
        self.encoder.fit(X[self.features])
        return self
    
    def transform(self, X):
        X_transformed = pd.concat([X.drop(columns=self.features).reset_index(drop=True), 
                                   pd.DataFrame(self.encoder.transform(X[self.features]), 
                                                columns=self.encoder.get_feature_names_out(self.features))],
                                  axis=1)
        return X_transformed




#Categorical Encoder
class CategoricalEncoder(BaseEstimator,TransformerMixin):
    def __init__(self, variables=None):
        self.variables=variables
    
    def fit(self, X,y):
        self.encoder_dict_ = {}
        for var in self.variables:
            t = X[var].value_counts().sort_values(ascending=True).index 
            self.encoder_dict_[var] = {k:i for i,k in enumerate(t,0)}
        return self
    
    def transform(self,X):
        X=X.copy()
        ##This part assumes that categorical encoder does not intorduce and NANs
        ##In that case, a check needs to be done and code should break
        for feature in self.variables:
            X[feature] = X[feature].map(self.encoder_dict_[feature])
        return X

# #Temporal Variables
# class TemporalVariableEstimator(BaseEstimator,TransformerMixin):
#     def __init__(self, variables=None, reference_variable = None):
#         self.variables=variables
#         self.reference_variable = reference_variable
    
#     def fit(self, X,y=None):
#         #No need to put anything, needed for Sklearn Pipeline
#         return self
    
#     def transform(self, X):
#         X=X.copy()
#         for var in self.variables:
#             X[var] = X[var]-X[self.reference_variable]
#         return X 



    
# # # Log Transformations
# class LogTransformation(BaseEstimator, TransformerMixin):
#     def __init__(self, variables=None):
#         self.variables = variables
    
#     def fit(self, X,y):
#         return self

#     ### Need to check in advance if the features are all non negative >0
#     ### If yes, needs to be transformed properly
#     def transform(self,X):
#         X=X.copy()
#         for var in self.variables:
#             X[var] = np.log(X[var])
#         return X


# # # Drop Features
# class DropFeatures(BaseEstimator, TransformerMixin):
#     def __init__(self, variables_to_drop=None):
#         self.variables_to_drop = variables_to_drop
    
#     def fit(self, X,y=None):
#         return self 
    
#     def transform(self, X):
#         X=X.copy()
#         X= X.drop(self.variables_to_drop, axis=1)
#         return X
    
    
#Rare label Categorical Encoder

# class RareLabelCategoricalImputer(BaseEstimator,TransformerMixin):
#     def __init__(self, tol=0.05, variables=None):
#         self.tol=tol
#         self.variables=variables
    
#     def fit(self, X, y=None):
#         self.encoder_dict_={}
#         for var in self.variables:
#             # the encoder will learn the most frequent categories
#             t = pd.Series(X[var].value_counts() / np.float(len(X)))
#             # frequent labels:
#             self.encoder_dict_[var] = list(t[t >= self.tol].index)
#         return self

#     def transform(self, X):
#         X=X.copy()
#         for feature in self.variables:
#             X[feature] = np.where(X[feature].isin(self.encoder_dict_[feature]), X[feature], 'Rare')
#         return X

