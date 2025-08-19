import os , sys 
import pandas as pd 
from typing import Optional
from dataclasses import dataclass
from sklearn.metrics import f1_score # we will evaluate models based one f1_score

from src.us_visa.exception import UsVisaException
from src.us_visa.logger import logging
from src.us_visa.constants import TARGET_COLUMN , CURRENT_YEAR
from src.us_visa.entity.s3_estimator import USvisaEstimator
from src.us_visa.entity.estimator import UsVisaModel
from src.us_visa.entity.estimator import TargetValueMapping
from us_visa.entity.artifact_entity import DataIngestionArtifact, ModelEvaluationArtifact, ModelTrainerArtifact
from us_visa.entity.config_entity import ModelEvaluationConfig



@dataclass
class EvaluateModelResponse:
    trained_model_f1_score : float
    cloud_model_f1_score : float
    is_model_accepted : bool
    f1_score_difference : float



class ModelEvaluation:
    
    def __init__(self, model_evaluation_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise UsVisaException(e , sys) from e

    
    def get_best_model(self) -> Optional[USvisaEstimator]:
        """
        Description :   This function is used to get model in production
        
        Output      :   Returns model object if available in s3 storage
        """
        
        try:
            bucket_name = self.model_evaluation_config.bucket_name 
            model_path = self.model_evaluation_config.s3_model_key_path
            
            usvisa_estimator = USvisaEstimator(
                bucket_name = bucket_name , model_path = model_path
            )
            
            if usvisa_estimator.is_model_present_in_bucket(model_path = model_path):
                return usvisa_estimator
            
            return None 
            
        except Exception as e:
            raise UsVisaException(e , sys)
    
    
    def evaluate_model(self) -> EvaluateModelResponse:
        """ 
        Description :   This function is used to evaluate trained model with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        """
        
        try:
            # load the test dataframe
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
            # drop any row if needed[check feature engineering]
            # 2. remove these rows where no_of_employees are 0 or -ve
            count_rows_removed = test_df.shape[0] 
            test_df = test_df[test_df['no_of_employees'] > 0]
            count_rows_removed -= test_df.shape[0]
            logging.info(f"from test_df data: rows removed for no_of_employees 0/-ve values: {count_rows_removed}")
            
            # Find the company age
            test_df['company_age'] = CURRENT_YEAR - test_df['yr_of_estab']
            # drop columns 
            test_df = test_df.drop(columns = ['yr_of_estab'] , axis = 1)
            
            # seperate feature and target
            logging.info("seperate input feature and target from test data")
            X_test , y_test = test_df[ : , : -1] , test_df[ : , -1]
            
            #  do the target value mapping
            y_test = y_test.replace(TargetValueMapping._asdict())
            
            # get the f1_score of trained model
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            logging.info("got the f1_score of trained model from evaluate method of class ModelEvaluation")
            
            
            best_model_f1_score = None 
            best_model = self.get_best_model()
            
            # if cloud has a model then do prediction using the model
            if best_model is not None: 
                y_pred_best_model = best_model.predict(X_test)
                best_model_f1_score = f1_score(y_test , y_pred_best_model)
                
            
            temp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            
            is_model_accepted = best_model_f1_score > trained_model_f1_score
            f1_score_difference = trained_model_f1_score - best_model_f1_score
            
            # make the EvaluateModelResponse object to model pusher
            result = EvaluateModelResponse(
                trained_model_f1_score = temp_best_model_score,
                cloud_model_f1_score = best_model_f1_score,
                is_model_accepted = is_model_accepted , 
                f1_score_difference = f1_score_difference
            )
            logging.info(f"result: {result}")
            logging.info("Exiting from evaluate_model method of ModelEvaluation class")
            return result
        
        except Exception as e:
            raise UsVisaException(e , sys)
    
    
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """ 
        This function is used to initiate all steps of the model evaluation
        """
        
        try:
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_evaluation_config.s3_model_key_path
            
            model_evaluation_artifact = ModelEvaluationArtifact(
               is_model_accepted = evaluate_model_response.is_model_accepted,
               s3_model_path = s3_model_path , 
               trained_model_path = self.model_trainer_artifact.trained_model_file_path,
               changed_accuracy = evaluate_model_response.f1_score_difference
            )
            
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
             
        except Exception as e:
            raise UsVisaException(e , sys)