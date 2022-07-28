from banking.exception import BankingException
from banking.logger import logging
import yaml
import os, sys
import pandas as pd
import numpy as np
import dill
import shutil


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

def write_yaml_data(file_path: str, data:dict = None) -> None:
    """
    This function is responsible for writing the data content passed
    to the YAML file for which the path is specified.
    Parameters
    ----------
    file_path : str
        File path for the YAML file.
    data : dict, By default: None
        Data to be written in the YAML file.
    """    
    try: 
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file_object:
            yaml.dump(data,file_object)
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
        logging.info(f"Saving pandas DataFrame to File: [{file_path}]")
    except Exception as e:
        raise BankingException(e,sys) from e

def load_df_from_csv(file_path: str) -> pd.DataFrame:
    """
    This function is responsible for loading and return 
    dataframe from the specified file path.
    Parameters:
    -----------
    file_path : str
        File path for which the DataFrame needs to be returned.
    
    Returns:
    --------
    df : pd.DataFrame
        DataFrame extracted from the given file path.
    """   
    try:
        if os.path.exists(file_path):
            logging.info(f"Extracting DataFrame from file: [{file_path}]")
            df = pd.read_csv(file_path)
            return df
        raise Exception(f"File: [{file_path}] does not exists.")
    except Exception as e:
        raise BankingException(e, sys) from e


def save_numpy_array_to_file(data: np.array, file_path: str) -> None:
    """
    This function is responsible for saving the numpy array to
    the desired location.
    Parameters:
    -----------
    data : numpy.array()
        numpy array to be saved.
    file_path : str
        File path for saving data.
    """   
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file=file_path, arr=data)
        logging.info(f"Saving numpy Array to File: [{file_path}]")
    except Exception as e:
        raise BankingException(e, sys) from e

def load_numpy_array_from_file(file_path: str) -> np.array:
    """
    This function is responsible for loading the numpy array 
    from the desired location.
    Parameters:
    -----------
    file_path : str
        File path from where the numpy array will be loaded.
    """   
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        numpy_arrary = np.load(file=file_path)
        logging.info(f"Loading numpy Array from File: [{file_path}]")
        return numpy_arrary
    except Exception as e:
        raise BankingException(e, sys) from e


def save_model_object_to_file(model, file_path:str) -> None:
    """
    This function is responsible for saving the model object passed to
    the desired location.
    Parameters:
    -----------
    model:
        Model object to be dumped/saved.
    file_path : str
        File path for dumping/saving model object.
    """   
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj=model, file=file_obj)
        logging.info(f"Saving model object to pickle file: [{file_path}]")
    except Exception as e:
        raise BankingException(e, sys) from e

def load_model_object_from_file(file_path: str):
    """
    This function is responsible for loading and return 
    model object from the specified file path.
    Parameters:
    -----------
    file_path : str
        File path for loading the model object.
    
    Returns:
    --------
    model_object: 
        Model object loaded from the passed file path.
    """ 
    try:
        if os.path.exists(file_path):
            logging.info(f"Extracting model object from file: [{file_path}]")
            with open(file_path, 'rb') as file_obj:
                model_object = dill.load(file_obj)
                logging.info(f"Loading model object from File: [{file_path}]")
                return model_object
        raise Exception(f"File: [{file_path}] does not exists.")
    except Exception as e:
        raise BankingException(e, sys) from e


def del_existing_dir(directory:str, threshold:int=None):
    """
    This function is responsible deling the directory specified if it exists.
    Parameters:
    -----------
    directory: str
        Path for the directory to be deleted.
    threshold : str, by default=None
        Number of files to be kept in the directory
    """ 
    try:
        if os.path.exists(directory):
            files = os.listdir(directory)
            num_of_files_present = len(files)
                
            if threshold is not None:
                num_of_files_to_be_deleted = num_of_files_present - threshold

                for idx,file in zip(range(num_of_files_to_be_deleted), files):
                    os.remove(os.path.join(directory, file))
            else:
                for file in files:
                    os.remove(os.path.join(directory, file))
                    
            shutil.rmtree(directory)

            

    except Exception as e:
        raise BankingException(e,sys) from e

def get_np_array_for_df(df:pd.DataFrame) -> np.array:
    """
    This function is responsible for transforming the 
    passed pandas DataFrame into numpy array.
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to be transformed into numpy array.
    
    Returns:
    --------
    arr : np.array
        Transformed Array generated for the DataFrame passed
    """ 
    try:
        arr = np.array(df)
        return arr
    except Exception as e:
        raise BankingException(e, sys) from e