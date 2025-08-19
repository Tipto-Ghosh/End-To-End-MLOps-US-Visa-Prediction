import os 
import sys 

import numpy as np
import pandas as pd 

from src.us_visa.entity.config_entity import UsVisaPredictorConfig
from src.us_visa.entity.s3_estimator import USvisaEstimator
from src.us_visa.exception import UsVisaException
from src.us_visa.logger import logging
from src.us_visa.utils.main_utils import read_yaml_file


# class to get the data from user
class UsVisaData:
    """ 
    
    """
    def __init__(self,
        continent , education_of_employee , has_job_experience , requires_job_training,
        no_of_employees , region_of_employment , prevailing_wage , unit_of_wage,
        full_time_position , company_age
    ):
        """
        Usvisa Data constructor
        Input: all features of the trained model for prediction
        """
        
        try:
            self.continent = continent
            self.education_of_employee = education_of_employee
            self.has_job_experience = has_job_experience
            self.requires_job_training = requires_job_training
            self.no_of_employees = no_of_employees
            self.region_of_employment = region_of_employment
            self.prevailing_wage = prevailing_wage
            self.unit_of_wage = unit_of_wage
            self.full_time_position = full_time_position
            self.company_age = company_age
            
        except Exception as e:
            raise UsVisaException(e , sys) from e 
    
    def get_usvisa_data_as_dict(self): 
        """
        This function returns a dictionary from USvisaData class input 
        """
    
        try:
            logging.info("Entered get_usvisa_data_as_dict method as USvisaData class")
           
            input_data = {
               "continent": [self.continent],
                "education_of_employee": [self.education_of_employee],
                "has_job_experience": [self.has_job_experience],
                "requires_job_training": [self.requires_job_training],
                "no_of_employees": [self.no_of_employees],
                "region_of_employment": [self.region_of_employment],
                "prevailing_wage": [self.prevailing_wage],
                "unit_of_wage": [self.unit_of_wage],
                "full_time_position": [self.full_time_position],
                "company_age": [self.company_age],
            }       
            
            logging.info("Created usvisa data dict")
            logging.info("Exited get_usvisa_data_as_dict method as USvisaData class")
            
            return input_data
        except Exception as e:
            raise UsVisaException(e , sys) from e 
        
    def get_usvisa_input_data_frame(self)-> pd.DataFrame:
        """ 
        converts UsVisaData(Dict) into dataframe and return it
        """
        try:
           usvisa_input_dict = self.get_usvisa_data_as_dict()
           return  usvisa_input_dict
        except Exception as e:
            raise UsVisaException(e , sys) from e 



class UsVisaClassifier:
    def __init__(self , prediction_pipeline_config : UsVisaPredictorConfig = UsVisaPredictorConfig() , ) -> None:
        try:
            
            self.prediction_pipeline_config = prediction_pipeline_config 
        except Exception as e:
            raise UsVisaException(e , sys) from e 
    
    def predict(self , dataframe) -> str:
        """ 
        This method used to do the prediction
        Returns: Prediction as a string
        """
        
        try:
            logging.info("Entered predict method of USvisaClassifier class")
        
            model = USvisaEstimator(
                bucket_name = self.prediction_pipeline_config.model_bucket_name,
                model_path = self.prediction_pipeline_config.model_file_path
            ) 
            
            result = model.predict(dataframe = dataframe)
            
            return result
        except Exception as e:
            raise UsVisaException(e , sys) from e 