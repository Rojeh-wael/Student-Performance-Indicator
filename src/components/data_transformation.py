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

            numerical_columns = []
            categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
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
            # fit the preprocessor on training features and transform both train and test
            input_feature_train_arr = preprocessor.fit_transform(X_train)
            input_feature_test_arr = preprocessor.transform(X_test)

            # concatenate features and targets so downstream expects arrays with target as last column
            train_array = np.c_[input_feature_train_arr, y_train.values]
            test_array = np.c_[input_feature_test_arr, y_test.values]

            # save the fitted preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            return train_array, test_array
        except Exception as e:
            logging.info("Exception occurred in data transformation")
            raise CustomException(e, sys)