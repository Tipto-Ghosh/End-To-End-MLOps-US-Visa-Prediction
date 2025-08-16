import os 
from datetime import datetime
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
MODEL_FILE_NAME = "model.pkl"


# Data Ingestion Constants
DATA_INGESTION_COLLECTION_NAME : str = "visa_data"
DATA_INGESTION_DIR_NAME : str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR : str = "feature_store"
DATA_INGESTION_INGESTED_DIR : str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO : str = 0.2 # test size