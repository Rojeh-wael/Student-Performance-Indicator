import os  
import sys
from src.exception import CustomException
from src.logger import logging
import pickle
import numpy as np
import pandas as pd
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.info("Exception occurred while saving object")
        raise CustomException(e, sys)