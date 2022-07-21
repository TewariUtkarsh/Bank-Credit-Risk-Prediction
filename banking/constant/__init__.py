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


# Data Ingestion:
DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
DATA_INGESTION_DATASET_DOWNLOAD_URL_KEY = "dataset_download_url"
DATA_INGESTION_ZIP_DOWNLOAD_DIR_KEY = "zip_download_dir"
DATA_INGESTION_EXTRACTED_DATA_DIR = "extracted_data_dir"
DATA_INGESTION_RAW_DATA_DIR_KEY = "raw_data_dir"
DATA_INGESTION_INGESTED_DATA_DIR_KEY = "ingested_data_dir"
DATA_INGESTION_INGESTED_TRAIN_DIR_KEY = "ingested_train_dir"
DATA_INGESTION_INGESTED_TEST_DIR_KEY = "ingested_test_dir"
DATA_INGESTION_ARTIFACT_DIR = "data_ingestion"


# Data Validation:
DATA_VALIDATION_CONFIG_KEY = "data_validation_config"
DATA_VALIDATION_SCHEMA_FILE_DIR_KEY = "schema_file_dir"
DATA_VALIDATION_SCHEMA_FILE_NAME_KEY = "schema_file_name"
DATA_VALIDATION_REPORT_FILE_NAME_KEY = "report_file_name"
DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY = "report_page_file_name"
DATA_VALIDATION_ARTIFACT_DIR = "data_validation"

SCHEMA_COLUMN_NAME_KEY = "columns"
SCHEMA_TARGET_COLUMN_KEY = "target_column"
SCHEMA_DOMAIN_VALUE_KEY = "domain_value"

DATA_DRIFT_KEY = "data_drift"
DATA_DRIFT_DATA_KEY = "data"
DATA_DRIFT_METRICS_KEY = "metrics"
DATA_DRIFT_DATASET_DRIFT_KEY = "dataset_drift"




