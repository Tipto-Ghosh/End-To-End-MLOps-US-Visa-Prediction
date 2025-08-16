from src.us_visa.utils.main_utils import read_yaml_file
from src.us_visa.constants import SCHEMA_FILE_PATH

print(SCHEMA_FILE_PATH)

print(read_yaml_file(SCHEMA_FILE_PATH)['transform_columns'])