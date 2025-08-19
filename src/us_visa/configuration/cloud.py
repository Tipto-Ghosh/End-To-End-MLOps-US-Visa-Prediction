import os
import sys 
import joblib
import wandb
import shutil


import mlflow
import mlflow.sklearn
from urllib.parse import urlparse

from src.us_visa.exception import UsVisaException
from src.us_visa.logger import logging
from src.us_visa.utils.main_utils import load_object , save_object
from src.us_visa.entity.config_entity import CloudModelConfig
from src.us_visa.entity.artifact_entity import CloudModelArtifact
from src.us_visa.constants import WANDB_PROJECT_NAME


class wandb_cloud:
    def __init__(self , cloud_config : CloudModelConfig = CloudModelConfig() , project_name: str = WANDB_PROJECT_NAME):
        self.cloud_config = cloud_config
        self.project_name = project_name
    
    def get_latest_model_from_cloud(self) -> CloudModelArtifact:
        """ 
        Fetches the latest model from W&B, saves it in cloud_model_dir,
        and returns CloudModelArtifact object
        """
        
        try:
            logging.info("Entered into get_latest_model method class name[wandb_cloud]")
            api = wandb.Api()
            
            # get all artifacts of type 'model' from the project
            # artifacts = api.artifacts(
            #     self.cloud_config.artifacts_path
            # )
            
            project = "Tipto_Ghosh/usvisa-prediction"
            artifact_type = "model"
            artifacts = api.artifact_versions(entity = "Tipto_Ghosh", project = "usvisa-prediction", type = artifact_type)

            
            logging.info(f"Got the artifacts. artifacts len: {len(artifacts)}")
             
            # sort the artifacts based on lastest creation
            artifacts = sorted(
                artifacts , key = lambda x : x.created_at , reverse = True
            )
            
            # if we dont have any artifacts
            if not artifacts:
                logging.info(
                    "No artifacts found in cloud. Exiting from get_latest_model method class name[wandb_cloud]"
                )
                return CloudModelArtifact(
                    cloud_model_object_file_path = None
                )
            
            # if we have artifacts then take the first one[latest]
            latest_artifact = artifacts[0]
            logging.info(f"Got the lastest artifacts. created at: {latest_artifact.created_at}")
            
            artifact_dir = latest_artifact.download()
            logging.info(
                f"latest_artifact downloaded at: [{artifact_dir}]"
            )
            
            # Find the model
            model_files = [file for file in os.listdir(artifact_dir) if file.endswith(".pkl")]
            
            if not model_files:
                logging.info("No .pkl model file found in the latest artifact.")
                logging.info("Exiting from get_latest_model method class name[wandb_cloud]")
                return CloudModelArtifact(
                    cloud_model_object_file_path = None
                )
            
            # model source path
            model_src_path = os.path.join(artifact_dir, model_files[0])
            # model destination path
            model_dst_path = os.path.join(self.cloud_config.cloud_model_dir, model_files[0])
            
            # now load the model
            model_obj = load_object(model_src_path)
            logging.info(f"cloud latest model object loaded. object type:{type(model_obj)}")
            
            # save the model
            save_object(model_dst_path , model_obj)
            logging.info(f"cloud latest model saved at: [{model_dst_path}]")
            
            cloud_model_artifact = CloudModelArtifact(
                cloud_model_object_file_path = model_dst_path
            )
            logging.info("Exiting from get_latest_model method class name[wandb_cloud]")
            return cloud_model_artifact
                       
        except Exception as e:
            logging.info("Error while fetching cloud model")
            raise UsVisaException(e , sys)



class MlFlowConfiguration:
    def __init__(self):
        pass