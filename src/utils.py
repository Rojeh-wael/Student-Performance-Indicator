import os  
import sys

from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.info("Exception occurred while saving object")
        raise CustomException(e, sys)
def evaluate_models(X_train, y_train, X_test, y_test, models, param_distributions):
    try:
        report = {}

        for model_name, model in models.items():
            params = param_distributions.get(model_name, {})
            
            # Use GridSearchCV if params available, otherwise just fit the model
            if params:
                gscv = GridSearchCV(model, params, cv=5, n_jobs=-1)
                gscv.fit(X_train, y_train)
                best_model = gscv.best_estimator_
            else:
                model.fit(X_train, y_train)
                best_model = model
            
            y_test_pred = best_model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score

        return report

    except Exception as e:
        logging.info("Exception occurred while evaluating models")
        raise CustomException(e, sys)