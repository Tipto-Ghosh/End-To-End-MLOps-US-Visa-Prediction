import logging
import os 
from datetime import datetime
from from_root import from_root

# make the logs directory
logs_dir = os.path.join(from_root() , "logs")
os.makedirs(logs_dir , exist_ok = True)

# Log filename
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(logs_dir , LOG_FILE)

# create the log file format
log_format = "[%(asctime)s] Line: %(lineno)d | %(name)s - %(levelname)s - %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    filename = LOG_FILE_PATH, 
    format = log_format, 
    datefmt = date_format,
    level = logging.INFO
)