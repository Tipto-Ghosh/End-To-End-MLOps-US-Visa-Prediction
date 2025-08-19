import boto3
import os 

from src.us_visa.constants import AWS_ACCESS_KEY_ID_ENV_KEY , AWS_SECRET_ACCESS_KEY_ENV_KEY , REGION_NAME
from src.us_visa.logger import logging

class S3Client:
    
    s3_client = None 
    s3_resource = None 
    
    def __init__(self , region_name: str = REGION_NAME):
        """ 
        This Class gets aws credentials from env_variable and creates an connection with s3 bucket 
        and raise exception when environment variable is not set
        """
        
        if S3Client.s3_client is None or S3Client.s3_resource is None:
            __access_key_id = AWS_ACCESS_KEY_ID_ENV_KEY
            __secret_access_id = AWS_SECRET_ACCESS_KEY_ENV_KEY
           
            if __access_key_id is None:
               raise Exception(f"Environment variable: {AWS_ACCESS_KEY_ID_ENV_KEY} is not not set.")
            
            if __secret_access_id is None:
                raise Exception(f"Environment variable: {AWS_SECRET_ACCESS_KEY_ENV_KEY} is not not set.")
            
            # if every thing is okay then make the connection
            S3Client.s3_resource = boto3.resource(
                "s3" , aws_access_key_id = __access_key_id,
                aws_secret_access_key = __secret_access_id,
                region_name = region_name
            )
            logging.info(
                "S3Client.s3_resource connection done"
            )
            
            S3Client.s3_client = boto3.client(
                "s3" , aws_access_key_id = __access_key_id,
                aws_secret_access_key = __secret_access_id,
                region_name = region_name
            )
            logging.info(
                "S3Client.s3_client connection done"
            )
            
            self.s3_resource = S3Client.s3_resource
            self.s3_client = S3Client.s3_client
            