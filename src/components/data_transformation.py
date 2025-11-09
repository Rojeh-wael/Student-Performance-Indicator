import sys
from dataclasses import dataclass
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Data Transformation initiated")

            numerical_columns = ['math score', 'reading score', 'writing score']
            categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler())
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_pipeline, numerical_columns),
                    ('cat', categorical_pipeline, categorical_columns)
                ]
            )

            logging.info("Data Transformation completed")
            return preprocessor
        except Exception as e:
            logging.info("Exception occurred in Data Transformation")
            raise CustomException(e, sys)
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Initiating data transformation")
            preprocessor = self.get_data_transformer_object()

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            X_train = train_df.drop(columns=['math score', 'reading score', 'writing score'])
            y_train = train_df[['math score', 'reading score', 'writing score']]

            X_test = test_df.drop(columns=['math score', 'reading score', 'writing score'])
            y_test = test_df[['math score', 'reading score', 'writing score']]

            logging.info("Data transformation completed")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            return X_train, y_train, X_test, y_test
        except Exception as e:
            logging.info("Exception occurred in data transformation")
            raise CustomException(e, sys)