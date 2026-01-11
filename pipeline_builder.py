import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Transformer for Feature Engineering.
    Creates new features and transforms existing ones.
    """
    def __init__(self, add_family_size=True):
        self.add_family_size = add_family_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.add_family_size:
            if 'sibsp' in X_copy.columns and 'parch' in X_copy.columns:
                X_copy['family_size'] = X_copy['sibsp'] + X_copy['parch'] + 1
        if 'fare' in X_copy.columns:
            X_copy['fare'] = np.log1p(X_copy['fare'])
            
        return X_copy

def get_preprocessing_pipeline(config):
    """
    Creates the core preprocessing (Imputing, Scaling, Encoding).
    """
    scaler = StandardScaler() if config.get("scaling_method") == "standard" else MinMaxScaler()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=config.get("num_impute_strategy", "median"))),
        ('scaler', scaler)
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, config["numeric_features"]),
            ('cat', categorical_transformer, config["categorical_features"])
        ],
        remainder='drop' 
    )
    
    return preprocessor

def build_full_model_pipeline(config):
    """
    Assembles the complete end-to-end pipeline: 
    Feature Engineering -> Preprocessing -> Model
    """
    
    full_pipeline = Pipeline(steps=[
        ('engineer', FeatureEngineer()), 
        ('preprocessor', get_preprocessing_pipeline(config)), 
        ('classifier', LogisticRegression()) 
    ])
    
    return full_pipeline