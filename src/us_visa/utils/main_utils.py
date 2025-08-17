import os 
import sys 

import numpy as np
import dill
import yaml
from pandas import DataFrame
import pandas as pd 
from src.us_visa.exception import UsVisaException
from src.us_visa.logger import logging


def read_csv(file_path: str) -> DataFrame:
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise UsVisaException(e , sys)

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise UsVisaException(e, sys) from e
    

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise UsVisaException(e, sys) from e
    

def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of utils")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited the save_object method of utils")

    except Exception as e:
        raise UsVisaException(e, sys) from e


    
def load_object(file_path: str) -> object:
    logging.info("Entered the load_object method of utils")

    try:

        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)

        logging.info("Exited the load_object method of utils")

        return obj

    except Exception as e:
        raise UsVisaException(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
           os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise UsVisaException(e, sys) from e
    



def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise UsVisaException(e, sys) from e



def drop_columns(df: DataFrame, cols: list)-> DataFrame:

    """
    drop the columns form a pandas DataFrame
    df: pandas DataFrame
    cols: list of columns to be dropped
    """
    logging.info("Entered drop_columns methon of utils")

    try:
        df = df.drop(columns = cols , axis = 1)
        logging.info(f"Dropped columns: {cols}")
        logging.info("Exited the drop_columns method of utils")
        return df
    except Exception as e:
        raise UsVisaException(e, sys) from e

def save_yaml_file(file_path: str, data: dict):
    """
    Save dictionary data to a YAML file.

    Args:
        file_path (str): Location of the YAML file to save.
        data (dict): Dictionary data to save in YAML format.
    """
    try:
        # Ensure the directory exists
        dir_path = os.path.dirname(file_path)
        if dir_path:
           os.makedirs(dir_path, exist_ok=True)
        
        # Write the dictionary to YAML
        with open(file_path, 'w') as file_obj:
            yaml.dump(data, file_obj)

    except Exception as e:
        raise UsVisaException(e, sys) from e