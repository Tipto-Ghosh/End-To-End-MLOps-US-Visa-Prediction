import sys 
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.us_visa.exception import UsVisaException
from src.us_visa.logger import logging


class TargetValueMapping:
    def __init__(self):
        self.Certified:int = 0
        self.Denied:int = 1
    
    def _asdict(self): 
        return self.__dict__
    
    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values() , mapping_response.keys()))



class UsVisaModel:
    def __init__(self , preprocessing_object: Pipeline, trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object
    
    
    def predict(self , dataframe : DataFrame) -> DataFrame:
        """
        Function accepts raw inputs and then transformed raw input using preprocessing_object
        which guarantees that the inputs are in the same format as the training data
        At last it performs prediction on transformed features
        """
        
        logging.info("Entered into predict method of UsVisaModel class")
        
        try:
           logging.info("Using the preprocessor to transform the dataframe")
           transformed_feature = self.preprocessing_object.transform(dataframe)
           logging.info(f"data transformed in predict method. transformed_feature shape({transformed_feature.shape})")
           
           logging.info("Using the trained model to get predictions")
           predictions = self.trained_model_object(transformed_feature)
           logging.info(f"predictions array shape ({predictions.shape})")
           logging.info("Exiting from predict method of UsVisaModel class")
           return predictions
       
        except Exception as e:
            raise UsVisaException(e , sys) from e 
    
    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"
    
    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"