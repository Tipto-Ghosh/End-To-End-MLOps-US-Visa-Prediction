import os 
import sys

from src.us_visa.exception import UsVisaException
from src.us_visa.logger import logging
from src.us_visa.constants import DATABASE_NAME , MONGODB_URL_KEY

import pymongo
import certifi

ca = certifi.where()


# Class to return a mongoDB connection to get access to the database
class MongoDbClient:
    client = None # class level
    
    def __init__(self , database_name = DATABASE_NAME):
        
        try:
            if MongoDbClient.client is None: # so make the connection
                mongo_db_url = MONGODB_URL_KEY
            
                if mongo_db_url is None:
                    logging.info("Missing mongo_db_url in the .env file")
                    raise UsVisaException("Missing mongo_db_url" , sys)
                
                MongoDbClient.client = pymongo.MongoClient(mongo_db_url , tlsCAFile = ca) 
            
            self.client = MongoDbClient.client
            self.database_name = database_name
            self.database = self.client[self.database_name]
            
            logging.info("MongoDB Connection Succesfull") 
        except Exception as e:
            raise UsVisaException(e , sys)    