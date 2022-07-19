from datetime import datetime
import os

# Logging Constants:
LOGGING_DIR = 'logs'

def get_current_time_stamp() -> str:
    """
    This function is responsible for returning the 
    current timestamp value in string format.

    Returns
    -------
    current_time_stamp : str
        Current timestamp in string format.
    """
    current_time_stamp = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    return current_time_stamp


# Base Constants:
CURRENT_TIME_STAMP = get_current_time_stamp()
ROOT_DIR = os.getcwd()
CONFIG_DIR = "config"
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = os.path.join(
                        ROOT_DIR,
                        CONFIG_DIR,
                        CONFIG_FILE_NAME
                    )

# Training Pipeline:
TRAINING_PIPELINE_CONFIG_KEY= "training_pipeline_config"
TRAINING_PIPELINE_NAME_KEY = "pipeline_name"
TRAINING_PIPELINE_ROOT_ARTIFACT_DIR_KEY = "root_artifact_dir"
