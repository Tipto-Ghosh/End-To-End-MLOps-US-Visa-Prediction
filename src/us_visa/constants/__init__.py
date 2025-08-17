import os 
from datetime import date
from dotenv import load_dotenv

load_dotenv()

DATABASE_NAME = "US_VISA"
COLLECTION_NAME = "visa_data"

FILE_NAME = "usvisa.csv" # Raw data file name

MONGODB_URL_KEY = os.getenv("mongodb_url")

PIPELINE_NAME : str = "usvisa"

ARTIFACT_DIR : str = "artifact"

TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"

PREPROCESSOR_OBJECT_FILE_NAME = "preprocessor.pkl"
MODEL_FILE_NAME = "model.pkl"


TARGET_COLUMN = "case_status"
CURRENT_YEAR = date.today().year
SCHEMA_FILE_PATH = os.path.join("config" , "schema.yaml")



# Data Ingestion Constants
DATA_INGESTION_COLLECTION_NAME : str = "visa_data"
DATA_INGESTION_DIR_NAME : str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR : str = "feature_store"
DATA_INGESTION_INGESTED_DIR : str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO : str = 0.2 # test size



# Data Validation realted contant start with DATA_VALIDATION 
DATA_VALIDATION_DIR_NAME : str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR : str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME : str = "report.yaml"




# Data Validation realted contant start with DATA_TRANSFORMATION
DATA_TRANSFORMATION_DIR_NAME : str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR : str = "transformed_data"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR : str = "transformed_object"


# Model Trainer realted contant start with MODEL_TRAINER
MODEL_TRAINER_DIR_NAME : str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR : str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME : str = MODEL_FILE_NAME
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH : str = os.path.join("config" , "model.yaml")
MODEL_TRAINER_EXPECTED_SCORE : float = 0.8 # minimal accuracy score for classification