import sys 
from pandas import DataFrame

from src.us_visa.cloud_storage.aws_storage import StorageService
from src.us_visa.entity.estimator import UsVisaModel
from src.us_visa.exception import UsVisaException
from src.us_visa.logger import logging

class USvisaEstimator:
    """
    This class is used to save and retrieve us_visas model in s3 bucket and to do prediction
    """
    
    def __init__(self , bucket_name , model_path , ):
        """
        :param bucket_name: Name of your model bucket
        :param model_path: Location of your model in bucket
        """
        self.bucket_name = bucket_name
        self.s3 = StorageService()
        self.model_path = model_path
        self.loaded_model : UsVisaModel = None
    
    def is_model_present_in_bucket(self , model_path) -> bool:
        try:
            return self.s3.s3_key_path_available(
                bucket_name = self.bucket_name , s3_key = model_path 
            )
        except UsVisaException as e:
            logging.info(e)
    
    def load_model_object(self , ) -> UsVisaModel:
        """ 
        Load the model from the model_path
        """
        return self.s3.load_model(self.model_path , bucket_name = self.bucket_name)
    
    
    def save_model(self,from_file,remove:bool=False)->None:
        """
        Save the model to the model_path
        :param from_file: Your local system model path
        :param remove: By default it is false that mean you will have your model locally available in your system folder
        :return:
        """
        try:
            self.s3.upload_file(
                from_file,
                to_filename = self.model_path,
                bucket_name = self.bucket_name,
                remove = remove
            )
        except Exception as e:
            raise UsVisaException(e , sys)
    
    
    def predict(self,dataframe:DataFrame):
        """
        :param dataframe:
        :return: prediction for the dataframe(test data)
        """
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model_object()
            
            return self.loaded_model.predict(dataframe=dataframe)
        
        except Exception as e:
            raise UsVisaException(e , sys) from e 
    
    