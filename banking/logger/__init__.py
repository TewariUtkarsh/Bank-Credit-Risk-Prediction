import logging
import os
from banking.constant import *


# Logging Constants:
LOGGING_DIR = "logs"
os.makedirs(LOGGING_DIR, exist_ok=True)

def get_log_file_name():
    """
    This class is responsible for return the 
    current timestamp value.

    Returns
    ----------
    str : str
        Returns the current timestamp in string format.
    """
    log_file_name = f"log_{get_current_time_stamp()}.log"
    return log_file_name

LOGGING_FILENAME = get_log_file_name()
LOGGING_FILEPATH = os.path.join(LOGGING_DIR, LOGGING_FILENAME)


logging.basicConfig(
    filename=LOGGING_FILEPATH,
    filemode='w',
    format='[%(asctime)s]^;%(lineno)s^;%(levelname)s^;%(filename)s^;%(funcName)s^;%(message)s',
    level=logging.INFO
)


# Get Logs Dataframe