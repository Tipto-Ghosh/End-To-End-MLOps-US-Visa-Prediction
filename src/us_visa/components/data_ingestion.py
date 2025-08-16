import sys
import os 

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.us_visa.entity.config_entity import DataIngestionConfig
from src.us_visa.entity.artifact_entity import DataIngestionArtifact
from src.us_visa.exception import UsVisaException
from src.us_visa.logger import logging
from src.us_visa.data_access.usvisa_data import USvisaData


class DataIngestion:
    # take data ingestion config as parameter
    def __init__(self , data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise UsVisaException(e , sys)
    
    def export_data_into_feature_store(self) -> DataFrame:
        """ 
        This method exports data from mongodb to csv file
        """
        try:
           logging.info(f"Getting data from mongodb")
           
           usvisa_data_obj = USvisaData()
           dataframe = usvisa_data_obj.export_collection_data_as_dataframe(
               collection_name = self.data_ingestion_config.collection_name
            )
           
           logging.info("Data conversion from Collection to DataFrame successful")
           logging.info(f"Shape of the dataframe: {dataframe.shape}")
           
           # save the full dataset as the feature_store
           feature_store_file_path = self.data_ingestion_config.feature_store_file_path
           feature_store_dir_path = os.path.dirname(feature_store_file_path)
           # make the directory
           os.makedirs(feature_store_dir_path , exist_ok = True)
           
           logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")
           dataframe.to_csv(feature_store_file_path , index = False , header = True)
           
           return dataframe
           
        except Exception as e:
            raise UsVisaException(e , sys)
    
    
    # Now do the train test split
    def split_data_as_train_test(self , dataframe: DataFrame) -> None:
        """ 
        This method splits the dataframe into train set and test set based on split ratio.
        Then save the train and test set as csv file
        """
        
        logging.info("Entered split_data_as_train_test method of data_ingestion")
        
        try:
            train_set , test_set = train_test_split(
                dataframe , 
                test_size = self.data_ingestion_config.train_test_split_ratio,
                random_state = 42
            )
            
            # Now make the directory where train and test data will be saved
            train_file_path = self.data_ingestion_config.training_file_path
            test_file_path = self.data_ingestion_config.testing_file_path
            dir_path = os.path.dirname(train_file_path)
            
            # Create the directory
            os.makedirs(dir_path , exist_ok = True)
            
            # Now save the train and test set as csv
            train_set.to_csv(train_file_path , index = False , header = True)
            test_set.to_csv(test_file_path , index = False , header = True)
            logging.info("Saved train and test data as csv file")
            
        except Exception as e:
            raise UsVisaException(e , sys)
    
    # now chain the data export and train test split method
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """ 
        This method initiates the data ingestion components of training pipeline 
        And returns DataIngestionArtifact as output
        """ 
        
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")
        
        try:
            # get the dataframe
            dataframe = self.export_data_into_feature_store()
            logging.info("from initiate_data_ingestion: Got the data from mongodb as Dataframe")
            
            # Do the train test split
            self.split_data_as_train_test(dataframe)
            
            logging.info("Performed train test split on the dataset")
            
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path = self.data_ingestion_config.training_file_path, 
                test_file_path = self.data_ingestion_config.testing_file_path
            ) 
            
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        
        except Exception as e:
            raise UsVisaException(e , sys)