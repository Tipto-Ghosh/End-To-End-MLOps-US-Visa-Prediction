import sys 
from src.us_visa.logger import logging
from src.us_visa.exception import UsVisaException


from src.us_visa.components.data_ingestion import DataIngestion
from src.us_visa.components.data_validation import DataValidation
from src.us_visa.components.data_transformation import DataTransformation
from src.us_visa.components.model_trainer import ModelTrainer
from src.us_visa.components.model_evaluation import ModelEvaluation
from src.us_visa.components.model_pusher import ModelPusher

from src.us_visa.entity.config_entity import(
    DataIngestionConfig , DataValidationConfig, 
    DataTransformationConfig , ModelTrainerConfig, 
    ModelEvaluationConfig , ModelPusherConfig
)
from src.us_visa.entity.artifact_entity import (
    DataIngestionArtifact , DataValidationArtifact, 
    DataTransformationArtifact , ModelTrainerArtifact, 
    ModelEvaluationArtifact , ModelPusherArtifact
)



class TrainingPipeline:
    def __init__(self):
        # Do the data ingestion
        self.data_ingestion_config = DataIngestionConfig()
        
        # Do the data validation
        self.data_validation_config = DataValidationConfig()
        
        # Do the data transformation
        self.data_transformation_config = DataTransformationConfig()
        
        # Do the model training
        self.model_trainer_config = ModelTrainerConfig()
        
        # Do the model evaluation
        self.model_evaluation_config = ModelEvaluationConfig()
        
        # Do the model pushing to s3
        self.model_pusher_config = ModelPusherConfig()
               
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
    
    
    def start_data_validation(self , data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """ 
        Do the data validation and return the DataValidationArtifact
        """
        
        try:
            logging.info("Entered start_data_validation method from TrainingPipeline class")
            
            data_validation = DataValidation(
                data_validation_config = self.data_validation_config,
                data_ingestion_artifact = data_ingestion_artifact
            )
            
            # initiate the data validation
            data_validation_artifact = data_validation.initiate_data_validation()
            
            logging.info("Performed the data validation operation")
            return data_validation_artifact
        except Exception as e:
            raise UsVisaException(e , sys)
        
        
    def start_data_transformation(self , data_ingestion_artifact: DataIngestionArtifact , data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        """
        This method of TrainPipeline class is responsible for starting data transformation component
        """
        try:
            data_transformation = DataTransformation(
                data_transformation_config = self.data_transformation_config,
                data_ingestion_artifact = data_ingestion_artifact,
                data_validation_artifact = data_validation_artifact
            )
            
            # initiate  data transformation
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise UsVisaException(e , sys)
        
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """
        This method of TrainPipeline class is responsible for starting model training
        """
        try:
            model_trainer = ModelTrainer(
                data_transformation_artifact = data_transformation_artifact,
                model_trainer_config = self.model_trainer_config
            )
            
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact

        except Exception as e:
            raise UsVisaException(e, sys)
        
    
    def start_model_evaluation(self , data_ingestion_artifact : DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        
        """ 
        This method of TrainingPipeline class is responsible for starting model evaluation
        """
        try:
            logging.info("Entered into start_model_evaluation from training pipeline")
            
            model_evaluation = ModelEvaluation(
                model_evaluation_config = self.model_evaluation_config,
                data_ingestion_artifact = data_ingestion_artifact,
                model_trainer_artifact = model_trainer_artifact
            ) 
            
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            return model_evaluation_artifact
        
        except Exception as e:
            raise UsVisaException(e , sys) from e 
        
    def start_model_pusher(self , model_evaluation_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        
        """ 
        This method of TrainPipeline class is responsible for starting model pushing
        """
        try:
            logging.info("Entered into run_model_pusher")
            model_pusher = ModelPusher(
                model_evaluation_artifact = model_evaluation_artifact
            )
            
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            logging.info("Exiting from run_model_pusher from training_pipeline class")
            
            return model_pusher_artifact
        except Exception as e:
            raise UsVisaException(e , sys) from e 
        
        
    def run_training_pipeline(self , ) -> None:
        """ 
        This method of TrainingPipeline class is responsible for running complete training pipeline
        """
        
        try:
            # 1. Run the data ingestion  
            data_ingestion_artifact = self.start_data_ingestion()
            logging.info("Data Ingestion is Done!!")
            
            # 2. Run the data validation
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact = data_ingestion_artifact)
            #print(f"data_validation_artifact: {data_validation_artifact}")
            
            # 3. Run the data transformation
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact = data_ingestion_artifact,
                data_validation_artifact = data_validation_artifact
            )
            
            # 4. Run the model trainer
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact = data_transformation_artifact
            )
            
            # 5. Run the model evaluation
            model_evaluation_artifact = self.start_model_evaluation(
                data_ingestion_artifact = data_ingestion_artifact, 
                model_trainer_artifact = model_trainer_artifact
            )
            
            # check current model is accepted or not
            if not model_evaluation_artifact.is_model_accepted:
                logging.info("Model is not accepted. So cloud model remain same")
                return None 
                
            # 6. Run the model pusher   
            model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact = model_trainer_artifact)
            logging.info("End of run_training pipeline")
            logging.info(model_pusher_artifact.bucket_name)
          
        except Exception as e:
            raise UsVisaException(e , sys)