import logging
import os, sys
from banking.constant import *
import pandas as pd

from banking.exception import BankingException


os.makedirs(LOGGING_DIR, exist_ok=True)


def get_log_file_name() -> str:
    """
    This class is responsible for returning the 
    log file name.

    Returns
    -------
    log_file_name : str
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
def get_logs_dataframe() -> pd.DataFrame:
    """
    This function is responsible for returning the dataframe for the
    log file.

    Returns
    -------
    log_df : pd.DataFrame
        Dataframe for the log file.
    """
    try:
        log_file_path = LOGGING_FILEPATH

        rows = []
        with open(log_file_path, 'r') as file_obj:
            for line in file_obj.readlines():
                rows.append(line.split('^;'))

        columns=["Time stamp","Log Level","line number","file name","function name","message"]
        log_df = pd.DataFrame(rows, columns=columns)
        log_df["log_message"] = log_df['Time stamp'].astype(str) +" :$ "+ log_df["message"]
        return log_df
    except Exception as e:
        raise BankingException(e, sys)

