import sys
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score , f1_score , precision_score , recall_score

from src.us_visa.utils.model_factory import ModelFactory

from src.us_visa.exception import UsVisaException
from src.us_visa.logger import logging

from src.us_visa.utils.main_utils import load_numpy_array_data , read_yaml_file , load_object , save_object
from src.us_visa.entity.config_entity import ModelTrainerConfig
from src.us_visa.entity.artifact_entity import DataTransformationArtifact , ModelTrainerArtifact , ClassificationMetricArtifact
from src.us_visa.entity.estimator import UsVisaModel



class ModelTrainer:
    def __init__(self , data_transformation_artifact: DataTransformationArtifact , model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
    
    def get_model_object_and_report(self , train: np.array , test: np.array) -> Tuple[object , object]:
        """ 
        Description :   This function uses ModelFactory to get the best model object and report of the best model.
        Returns metric artifact object and best model object
        """
        
        try:
            logging.info("Entered into get_model_object_and_report method of ModelTrainer class")
            
            # 1. seperate input feature and target from train data
            logging.info("1. seperate input feature and target from train data")
            X_train , y_train = train[ : , : -1] , train[ : , -1]
            
            # 2. seperate input feature and target from test data
            logging.info("2. seperate input feature and target from test data")
            X_test , y_test = test[ : , : -1] , test[ : , -1]
            
            # 3. use ModelFactory to get the best model object
            
            model_factory = ModelFactory(
                model_config_path = self.model_trainer_config.model_config_file_path,
                tuned_model_report_path = self.model_trainer_config.tuned_model_report_file_path
            )
            
            # tune all the models from model.yaml file
            logging.info("calling model_factory.get_best_model method from get_model_object_and_report")
            best_model_detail = model_factory.get_best_model(
                X_train = X_train , y_train = y_train , base_accuracy = self.model_trainer_config.expected_accuracy
            ) 
            
            # get the best model_object
            model_obj = best_model_detail.best_model
            logging.info("Got best model from best_model_detail.best_model")
            
            # 4. do the prediction using on test data with the best model
            logging.info("started the prediction using on test data with the best model")
            y_pred = model_obj.predict(X_test)
            logging.info(f"prediction done with best model object. y_pred shape: ({y_pred.shape})")
            
            # 5. find the classification metrices for test data
            logging.info("finding the classification metrices for test data") 
            accuracy = accuracy_score(y_test, y_pred) 
            f1 = f1_score(y_test, y_pred)  
            precision = precision_score(y_test, y_pred)  
            recall = recall_score(y_test, y_pred)
            
            # 6. make the ClassificationMetricArtifact
            logging.info("making the ClassificationMetricArtifact")
            metric_artifact = ClassificationMetricArtifact(
                accuracy_score = accuracy, f1_score = f1 , 
                precision_score = precision , recall_score = recall
            )
            
            # 7. return the best_model details and ClassificationMetricArtifact
            logging.info("Exiting from get_model_object_and_report method")
            return metric_artifact
        
        except Exception as e:
            raise UsVisaException(e , sys)
    
    
    def initiate_model_trainer(self , ) -> ModelTrainerArtifact:
        """ 
        This function initiates a model trainer steps for training pipeline
        """
        
        try:
            logging.info("Entered initiate_model_trainer method of ModelTrainer class")
            
            # 1. load the train and test numpy array
            train_arr = load_numpy_array_data(
                file_path = self.data_transformation_artifact.transformed_train_data_file_path
            )
            test_arr = load_numpy_array_data(
                file_path = self.data_transformation_artifact.transformed_test_data_file_path
            )
            
            # 2. call  get_model_object_and_report to get the best model
            best_model_detail , metric_artifact = self.get_model_object_and_report(
                train = train_arr , test = test_arr
            )
            
            # 3. check best model accepted or not based on expected_accuracy_score
            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                logging.info("No best model found with score more than expected_accuracy score")
                raise Exception("No best model found with score more than expected_accuracy score")
            
            # 4. Call UsVisaModel to combine preproccessor and best model
            # load the preprocessing object
            preprocessing_object = load_object(self.data_transformation_artifact.transformed_object_file_path)
            usvisa_model = UsVisaModel(
                preprocessing_object = preprocessing_object , trained_model_object = best_model_detail.best_model
            )
            # 5. save the usvisa_model(object) to trained_model_file_path
            logging.info("saving usvisa_model(preprocessing_object + best_model_detail.best_model)")
            save_object(file_path = self.model_trainer_config.trained_model_file_path)
            logging.info("Created usvisa model object with preprocessor and model")
            
            
            # 6. construct the model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path = self.model_trainer_config.trained_model_file_path,
                metric_artifact = metric_artifact
            )
            # 7. return the model trainer artifact
            return model_trainer_artifact
        
        except Exception as e:
            raise UsVisaException(e , sys)