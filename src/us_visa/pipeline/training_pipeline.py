import sys 
from src.us_visa.logger import logging
from src.us_visa.exception import UsVisaException
from src.us_visa.components.data_ingestion import DataIngestion
from src.us_visa.entity.config_entity import DataIngestionConfig
from src.us_visa.entity.artifact_entity import DataIngestionArtifact



class TrainingPipeline:
    def __init__(self):
        # Do the data ingestion
        self.data_ingestion_config = DataIngestionConfig()
    
    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        """ 
        This will start the data ingestion component and return DataIngestionArtifact
        """
        
        try:
            logging.info("Entered start_data_ingestion method from TrainingPipeline class")
            
            data_ingestion = DataIngestion(data_ingestion_config = self.data_ingestion_config)
            
            logging.info("Calling initiate data ingestion from start_data_ingestion method of TrainingPipeline class")
            
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion() 
            logging.info(f"data_ingestion_artifact is received from start_data_ingestion method. {data_ingestion_artifact}")
            
            return data_ingestion_artifact
        except Exception as e:
            raise UsVisaException(e , sys)
    
    
    def run_training_pipeline(self , ) -> None:
        """ 
        This method of TrainingPipeline class is responsible for running complete training pipeline
        """
        
        try:
          # 1. Run the data ingestion  
          data_ingestion_artifact = self.start_data_ingestion()
          logging.info("Data Ingestion is Done!!")
             
        except Exception as e:
            raise UsVisaException(e , sys)