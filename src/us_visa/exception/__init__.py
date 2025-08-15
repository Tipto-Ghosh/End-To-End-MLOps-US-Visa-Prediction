import sys 
from src.us_visa.logger import logging

# Define the error message
def error_message_detail(error, error_details: sys):
    """
    Creates a detailed error message with file name, line number, and the actual error message.
    """
    _, _, exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    return f"Error occurred in python script name [{file_name}] at line: [{line_number}] with error message [{str(error)}]"


class UsVisaException(Exception):
    def __init__(self, error_message , error_detail):
        # Get detailed message using sys
        super().__init__(error_message)
        
        self.error_message = error_message_detail(
            error = error_message,
            error_details = error_detail
        )
        
        logging.error(self.error_message , exc_info = True)
    def __str__(self):
        return self.error_message