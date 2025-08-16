import sys 
import pandas as pd 
import numpy as np
from typing import Optional

from src.us_visa.configuration.mongo_db_connection import MongoDbClient
from src.us_visa.exception import UsVisaException


class USvisaData:
    """ 
    This class helps to export entire mongo db record as pandas dataframe
    """
    
    def __init__(self):
        # get the client
        try: 
           self.mongo_client = MongoDbClient()
        except Exception as e:
            raise UsVisaException(e , sys)
    
    def export_collection_data_as_dataframe(self , collection_name: str , database_name: Optional[str] = None) -> pd.DataFrame:
        """ 
        Get all the data from the database collection as dict
        Convert the dict into dataframe
        Drop the mongodb default _id value and return
        """
        
        try:
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]
            
            # get all the data and convert the data into dataframe
            df = pd.DataFrame(list(collection.find()))
            
            # remove the _id column if present
            if '_id' in df.columns.to_list():
                df = df.drop(columns = ['_id'] , axis = 1)
            
            # replace na with nan
            df.replace({"na" : np.nan} , inplace = True)
            
            return df 
        
        except Exception as e:
            raise UsVisaException(e , sys)