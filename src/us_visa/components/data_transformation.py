import sys
import os 

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

from src.us_visa.constants import TARGET_COLUMN , SCHEMA_FILE_PATH , CURRENT_YEAR
from src.us_visa.entity.config_entity import DataIngestionConfig , DataTransformationConfig , DataValidationConfig
from src.us_visa.entity.artifact_entity import DataIngestionArtifact , DataValidationArtifact , DataTransformationArtifact

from src.us_visa.exception import UsVisaException
from src.us_visa.logger import logging

from src.us_visa.utils.main_utils import save_object , save_numpy_array_data , read_yaml_file , drop_columns , read_csv
from src.us_visa.entity.estimator import TargetValueMapping



class DataTransformation:
    def __init__(self , data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifact, 
                 data_validation_artifact: DataValidationArtifact
                ):
        
        try:
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            
            # read the schema file cause we need to know the drop and transformation col names
            self._schema_config = read_yaml_file(file_path = SCHEMA_FILE_PATH)
        except Exception as e:
            raise UsVisaException(e , sys)
    
    
    def get_data_transformation_object(self) -> Pipeline:
        """ 
        This method creates and returns a data transformer(preprocessor) object for the data
        """
        try:
            logging.info("Entered get_data_transformer_object method of DataTransformation class")
            
            # make the transformation pipeline
            
            numerical_transformer = StandardScaler()
            oneHot_transformer = OneHotEncoder()
            ordinal_transformer = OrdinalEncoder() 
            logging.info("Initialized StandardScaler, OneHotEncoder, OrdinalEncoder")
            
            # get the columns where we have to do preprocessing
            onehot_encoding_columns = self._schema_config['oh_columns']
            ordinal_encoding_columns = self._schema_config['or_columns']
            standard_scaler_columns = self._schema_config['num_features']
            power_transformer_columns = self._schema_config['transform_columns']
            
            logging.info(
                f"Got onehot_encoding_columns[{len(onehot_encoding_columns)}] , \
                ordinal_encoding_columns[{len(ordinal_encoding_columns)}] , \
                standard_scaler_columns , power_transformer_columns[{len(standard_scaler_columns)}], \
                power_transformer_columns[{len(power_transformer_columns)}] for preprocessor."
            )
            
            # do the power transformation
            transform_pipeline = Pipeline(steps = [
                ('transformer' , PowerTransformer(method = 'yeo-johnson'))
            ])
            
            # create the preprocessor pipeline
            preprocessor = ColumnTransformer([
                ("OneHotEncoder" , oneHot_transformer , onehot_encoding_columns), 
                ("Ordinal_Encoder" , ordinal_transformer , ordinal_encoding_columns), 
                ("Transformer" , transform_pipeline , power_transformer_columns),
                ("StandardScaler" , numerical_transformer , standard_scaler_columns)
            ])
            
            logging.info("Created preprocessor object from ColumnTransformer")
            return preprocessor
            
        except Exception as e:
            raise UsVisaException(e , sys)
    
    
    def initiate_data_transformation(self , ) -> DataTransformationArtifact:
        """ 
        This method initiates the data transformation component for the pipeline
        """
        
        try:
            # check data validation
            if self.data_validation_artifact.validation_status == True:
                logging.info("Starting data transformation")
                
                # get the preprocessor object
                preprocessor = self.get_data_transformation_object()
                logging.info("Got the preprocessor object")

                # get the train and test dataframe
                train_df = read_csv(self.data_ingestion_artifact.train_file_path)
                test_df = read_csv(self.data_ingestion_artifact.test_file_path)
                logging.info(f"from initiate_data_transformation method: train_df shape [{train_df.shape}]")
                logging.info(f"from initiate_data_transformation method: test_df shape [{test_df.shape}]")   
                
                # removes rows where no_of_employees <= 0 from both train and test data
                count_rows_removed = train_df.shape[0] 
                # 2. remove these rows where no_of_employees are 0 or -ve
                train_df = train_df[train_df['no_of_employees'] > 0]
                
                count_rows_removed -= train_df.shape[0]
                logging.info(f"from train data: rows removed for no_of_employees 0/-ve values: {count_rows_removed}")
                
                # also remove from test data frame
                count_rows_removed = test_df.shape[0] 
                # 2. remove these rows where no_of_employees are 0 or -ve
                test_df = test_df[test_df['no_of_employees'] > 0]
                
                count_rows_removed -= test_df.shape[0]
                logging.info(f"from test data: rows removed for no_of_employees 0/-ve values: {count_rows_removed}")
                
                             
                # separete target columns and input features[X , y]
                input_feature_train_df = train_df.drop(columns = [TARGET_COLUMN] , axis = 1)
                target_feature_train_df = train_df[TARGET_COLUMN]
                logging.info("Separeting X and y from train dataframe")
                
                # Do the feature engineering task on train dataframe
                
                # 1. make the company age column
                input_feature_train_df['company_age'] = CURRENT_YEAR - input_feature_train_df['yr_of_estab']
                logging.info("Added company_age column to the Training dataset")
                
                
                
                # 3. drop the unecessary columns
                columns_need_to_drop = self._schema_config['drop_columns']
                logging.info(f"got columns_need_to_drop[count = {len(columns_need_to_drop)}]")
                
                input_feature_train_df = drop_columns(df = input_feature_train_df , cols = columns_need_to_drop)
                logging.info("Dropping unecessary columns from train dataframe")
                
                
                # 4. Do the target value mapping
                mapper = TargetValueMapping()
                target_feature_train_df = target_feature_train_df.replace(
                    mapper._asdict()
                )
                logging.info(f"target value mapping done for train target data")
                
                # separete target columns and input features[X , y] from test data
                input_feature_test_df = test_df.drop(columns = [TARGET_COLUMN] , axis = 1)
                target_feature_test_df = test_df[TARGET_COLUMN]
                
                # Do the feature engineering task on test dataframe
                
                # 1. make the company age column
                input_feature_test_df['company_age'] = CURRENT_YEAR - input_feature_test_df['yr_of_estab']
                logging.info("Added company_age column to the Test dataset")
                
                
                # 3. drop the unecessary columns from test data
                input_feature_test_df = drop_columns(df = input_feature_test_df , cols = columns_need_to_drop)
                logging.info("Dropping unecessary columns from test dataframe")
                
                # 4. Do the target value mapping
                target_feature_test_df = target_feature_test_df.replace(
                    mapper._asdict()
                )
                logging.info(f"target value mapping done for test target data")
                
                logging.info("Feature engineering done on both train and test data")
                
                # do the fit_transform on train data
                logging.info("Applying preprocessing object on training dataframe and testing dataframe")
                
                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
                logging.info("Used the preprocessor object to fit transform the train features")
                
                # do the transformation on test data
                input_feature_test_arr = preprocessor.transform(input_feature_test_df)
                logging.info("Used the preprocessor object to transform the test features")
                
                # Handle imbalance problem
                logging.info("Applying SMOTEENN on Training dataset")
                
                smt = SMOTEENN(sampling_strategy = "minority")
                
                input_feature_train_final, target_feature_train_final = smt.fit_resample(
                    input_feature_train_arr , target_feature_train_df
                )
                logging.info("Applied SMOTEENN on training dataset")
                
                
                logging.info("Applying SMOTEENN on Test dataset")
                input_feature_test_final, target_feature_test_final = smt.fit_resample(
                    input_feature_test_arr, target_feature_test_df
                )
                logging.info("Applied SMOTEENN on testing dataset")
                
                logging.info("Created train array and test array")
                
                logging.info("concatenating input_feature_train_final arr and target_feature_train_final arr")
                train_arr = np.c_[
                    input_feature_train_final, np.array(target_feature_train_final)
                ]
                
                logging.info("concatenating input_feature_test_final arr and target_feature_test_final arr")
                test_arr = np.c_[
                    input_feature_test_final, np.array(target_feature_test_final)
                ]
                
                # save the preprocessor object
                preprocessor_file_path = self.data_transformation_config.transformed_object_file_path
                # preprocessor_dir_name = os.path.dirname(preprocessor_file_path)
                # os.makedirs(preprocessor_dir_name , exist_ok = True)
                
                save_object(
                    file_path = preprocessor_file_path,
                    obj = preprocessor 
                )
                logging.info("saved preprocessor object")
                
                
                # save the train and test data as numpy array
                save_numpy_array_data(
                    file_path = self.data_transformation_config.transformed_train_data_file_path,
                    array = train_arr
                )
                logging.info("saved train arr")
                save_numpy_array_data(
                    file_path = self.data_transformation_config.transformed_test_data_file_path,
                    array = test_arr
                )
                logging.info("saved test arr")
                
                # make the data transformation artifact
                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path = self.data_transformation_config.transformed_object_file_path,
                    transformed_train_data_file_path = self.data_transformation_config.transformed_train_data_file_path,
                    transformed_test_data_file_path = self.data_transformation_config.transformed_test_data_file_path
                )
                logging.info("Exited initiate_data_transformation method of Data_Transformation class")
                
                return data_transformation_artifact
            else:
                logging.info("data transformation failed because of validation status")
                raise Exception(self.data_validation_artifact.message) 
        except Exception as e:
            raise UsVisaException(e , sys)