import sys 

from src.us_visa.cloud_storage.aws_storage import StorageService
from src.us_visa.exception import UsVisaException
from src.us_visa.logger import logging
from src.us_visa.entity.artifact_entity import ModelEvaluationArtifact , ModelPusherArtifact
from src.us_visa.entity.config_entity import ModelPusherConfig
from src.us_visa.entity.s3_estimator import USvisaEstimator


class ModelPusher:
    def __init__(self , model_evaluation_artifact : ModelEvaluationArtifact,model_pusher_config : ModelPusherConfig):
        
        self.s3 = StorageService()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.usvisa_estimator = USvisaEstimator(
            bucket_name = model_pusher_config.bucket_name,
            model_path = model_pusher_config.s3_model_key_path
        )
    
    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        This function is used to initiate all steps of the model pusher 
 
        Returns:
            Returns model evaluation artifact
        """
        
        logging.info("Entered initiate_model_pusher method of ModelTrainer class")
        
        try:
            logging.info("Uploading artifacts folder to s3 bucket")
            self.usvisa_estimator.save_model(
                from_file = self.model_evaluation_artifact.trained_model_path
            )
            
            model_pusher_artifact = ModelPusherArtifact(
                bucket_name = self.model_pusher_config.bucket_name,
                s3_model_path = self.model_pusher_config.s3_model_key_path
            )
            
            logging.info("Uploaded artifacts folder to s3 bucket")
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelTrainer class")
            
            return model_pusher_artifact
             
        except Exception as e:
            raise UsVisaException(e , sys)
        
        
        