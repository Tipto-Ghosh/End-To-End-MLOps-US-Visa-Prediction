from src.us_visa.logger import logging
from src.us_visa.exception import UsVisaException
import sys 


try:
    a = 12 / 0
except Exception as e:
    raise UsVisaException(e , sys)