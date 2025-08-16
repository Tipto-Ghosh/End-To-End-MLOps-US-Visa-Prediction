import sys , os , json

import pandas as pd 
from pandas import DataFrame

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection

from src.us_visa.exception import UsVisaException
from src.us_visa.logger import logging
from src.us_visa.utils.main_utils import read_yaml_file , write_yaml_file , read_csv
from src.us_visa.entity.config_entity import DataValidationConfig
from src.us_visa.entity.artifact_entity import DataIngestionArtifact , DataValidationArtifact
from src.us_visa.constants import SCHEMA_FILE_PATH
import warnings
warnings.filterwarnings("ignore")



class DataValidation:
    def __init__(self , data_ingestion_artifact : DataIngestionArtifact , data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            # read the schema file
            logging.info("reading the schema file from DataValidation")
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            logging.info("reading the schema file from DataValidation is done")
        except Exception as e:
            raise UsVisaException(e , sys)     
    
    
    # validatate number of columns
    def validate_number_of_columns(self , dataframe: DataFrame) -> bool:
        try:
            total_columns_in_schema = len(self._schema_config['columns'])
            total_columns_in_dataframe = len(dataframe.columns)
            
            is_all_column_present = total_columns_in_schema == total_columns_in_dataframe
            logging.info(f"All Columns are present status: [{is_all_column_present}]")
            return is_all_column_present 
        except Exception as e:
            logging.info("Exception in validate_number_of_columns")
            raise UsVisaException(e , sys)
    
    
    # Validate all numerical columns exists
    def validate_all_numerical_column_exists(self , dataframe: DataFrame) -> bool:
        dataframe_columns = dataframe.columns.to_list()
        
        schema_numerical_columns = self._schema_config['numerical_columns']
        
        missing_numerical_columns = []
        
        for column in schema_numerical_columns:
            if column not in dataframe_columns:
                missing_numerical_columns.append(column)
        
        if len(missing_numerical_columns) > 0:
            logging.info(f"Missing Numerical Column: {missing_numerical_columns}")
        
        return False if len(missing_numerical_columns) > 0 else True
    
    
    
    # Validate all categorical columns exists
    def validate_all_categorical_column_exists(self , dataframe: DataFrame) -> bool:
        dataframe_columns = dataframe.columns.to_list()
        schema_categorical_columns = self._schema_config['categorical_columns']
        
        missing_categorical_columns = []
        
        for column in schema_categorical_columns:
            if column not in dataframe_columns:
                missing_categorical_columns.append(column)
        
        if len(missing_categorical_columns) > 0:
            logging.info(f"Missing categorical Column: {missing_categorical_columns}")
        
        return False if len(missing_categorical_columns) > 0 else True
    
    
    def initiate_column_validation(self , dataframe: DataFrame) -> bool:
        """ 
        chain the 3 steps of column validation
        """
        
        try:
            logging.info("Entered into initiate_column_validation")
            
            count_check = self.validate_number_of_columns(dataframe)
            numerical_check = self.validate_all_numerical_column_exists(dataframe)
            categorical_check = self.validate_all_categorical_column_exists(dataframe)
            
            status = count_check and numerical_check and categorical_check
            logging.info(f"Column validation status is [{status}]")
            
            return status
        except Exception as e:
            raise UsVisaException(e , sys)
        
        
    # now detect data drift
    def detect_dataset_drift(self , reference_df: DataFrame , current_df: DataFrame) -> bool:
        """ 
        This method validates if drift is detected or not.
        Returns bool value based on validation results.
        """
        
        try:
            # create the profile
            data_drift_profile = Profile(sections = [DataDriftProfileSection()]) 
            data_drift_profile.calculate(reference_data = reference_df , current_data = current_df)
            
            # make the report as json
            report = data_drift_profile.json()
            json_report = json.loads(report)
            
            # make the directory where report will be saved
            validation_dir = os.path.dirname(self.data_validation_config.data_drift_report_file_path)
            
            logging.info(f"validation_dir: [{validation_dir}]")
            os.makedirs(validation_dir , exist_ok = True) 
            # save the json_report as a yaml file
            write_yaml_file(
                file_path = self.data_validation_config.data_drift_report_file_path,
                content = json_report
            )
            
            n_features = json_report["data_drift"]["data"]["metrics"]["n_features"]
            n_drifted_features = json_report["data_drift"]["data"]["metrics"]["n_drifted_features"]
            data_drift_percentage = (n_drifted_features / n_features) * 100
            logging.info(f"Out of {n_features} columns data drift detected in {n_drifted_features} columns")
            logging.info(f"Data Drift percentage: {data_drift_percentage} %")
            drift_status = json_report["data_drift"]["data"]["metrics"]["dataset_drift"]
            
            # drift_status == False if no data drift -> everthing is okay
            # drift_status == True -> Data Drift detected 
            return drift_status
        except Exception as e:
            raise UsVisaException(e , sys)
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        """ 
        This method initiates the data validation component for the pipeline
        """
        
        try:
            logging.info("Starting data validation")
            validation_error_msg = ""
            
            # get the train and test data
            train_df = read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = read_csv(self.data_ingestion_artifact.test_file_path)
            
            logging.info(f"train_df data shape: {train_df.shape}")
            logging.info(f"test_df data shape: {test_df.shape}")
            
            # check column validity
            train_data_column_validity = self.initiate_column_validation(train_df)
            
            if train_data_column_validity == False:
                validation_error_msg += "[Columns are missing in training dataframe.]"

            test_data_column_validity = self.initiate_column_validation(test_df)
            if test_data_column_validity == False:
                validation_error_msg += "[Columns are missing in test dataframe.]"
            
            # column validation status
            column_validation_status = train_data_column_validity and test_data_column_validity
            
            # if column_validation_status is OK then start data drift
            if column_validation_status:
                drift_status = self.detect_dataset_drift(
                    reference_df = train_df , current_df = test_df
                )
                
                # if data drift detected 
                if drift_status:
                    logging.info("Data Drift Detected")
                    validation_error_msg = "Data Drift Detected"
                    data_validation_status = False
                else:
                    logging.info("Data Drift not Detected")
                    validation_error_msg = "Data Drift not Detected"
                    data_validation_status = True
            else:
                logging.info(f"Data Validation error: {validation_error_msg}") 
                data_validation_status = False
            
            
            # now make the artifact for data validation
            data_validation_artifact = DataValidationArtifact(
                validation_status = data_validation_status,
                message = validation_error_msg,
                drift_report_file_path = self.data_validation_config.data_drift_report_file_path
            )
            
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        
        except Exception as e:
            raise UsVisaException(e , sys)