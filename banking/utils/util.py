from banking.exception import BankingException
from banking.logger import logging
import yaml
import os, sys
import pandas as pd


def read_yaml_data(file_path:str) -> dict:
    """
    This function is responsible for reading the content of YAML file,
    for which the path is passed, and returns the content of the file.
    Parameters
    ----------
    file_path : str
        File path for the YAML file.
    
    Returns
    -------
    yaml_file_data : dict
        Content of the YAML file in the form dictionary.
    """    
    try: 
        with open(file=file_path, mode='rb') as yaml_file_obj:
            yaml_file_data = yaml.safe_load(stream=yaml_file_obj)
            logging.info(f"Loading content from [{file_path}]")
            return yaml_file_data
    except Exception as e:
        raise BankingException(e, sys) from e

    
def save_df_to_csv(data: pd.DataFrame(), file_path:str) -> None:
    """
    This function is responsible for saving the DataFrame passed to
    the desired location.
    Parameters:
    -----------
    data : pandas.DataFrame()
        DataFrame object of the dataset to be saved.
    file_path : str
        File path for saving Data.
    """   
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)
        data.to_csv(file_path, index=None)
        logging.info(f"Saving DataFrame at: [{file_path}]")
    except Exception as e:
        raise BankingException(e,sys) from e

