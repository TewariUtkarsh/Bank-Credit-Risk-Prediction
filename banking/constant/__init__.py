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
SCHEMA_OLD_TARGET_COLUMN_KEY = "old_target_column"
SCHEMA_DOMAIN_VALUE_KEY = "domain_value"
SCHEMA_CONTINUOUS_COLUMN_KEY = "continuous_column"

DATA_DRIFT_KEY = "data_drift"
DATA_DRIFT_DATA_KEY = "data"
DATA_DRIFT_METRICS_KEY = "metrics"
DATA_DRIFT_DATASET_DRIFT_KEY = "dataset_drift"


# Data Transformation:
DATA_TRANSFORMATION_CONFIG_KEY = "data_transformation_config"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR_KEY = "transformed_data_dir"
DATA_TRANSFORMATION_TRANSFORMED_TRAIN_DIR_KEY = "transformed_train_dir"
DATA_TRANSFORMATION_TRANSFORMED_TEST_DIR_KEY = "transformed_test_dir"
DATA_TRANSFORMATION_PREPROCESSED_MODEL_DIR_KEY = "preprocessed_model_dir"
DATA_TRANSFORMATION_PREPROCESSED_MODEL_OBJECT_FILE_NAME_KEY = "preprocessed_model_object_file_name"
DATA_TRANSFORMATION_ARTIFACT_DIR = "data_transformation"


# Model Trainer:
MODEL_TRAINER_CONFIG_KEY = "model_trainer_config"
MODEL_TRAINER_TRAINED_MODEL_DIR_KEY = "trained_model_dir"
MODEL_TRAINER_TRAINED_MODEL_FILE_NAME_KEY = "trained_model_file_name"
MODEL_TRAINER_BASE_ACCURACY_KEY = "base_accuracy"
MODEL_TRAINER_MODEL_CONFIG_DIR_KEY = "model_config_dir"
MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY = "model_config_file_name"
MODEL_TRAINER_ARTIFACT_DIR = "model_trainer"


# Model Evaluation:
MODEL_EVALUATION_CONFIG_KEY = "model_evaluation_config"
MODEL_EVALUATION_MODEL_EVALUATION_REPORT_FILE_NAME_KEY = "model_evaluation_report_file_name"
MODEL_EVALUATION_ARTIFACT_DIR = "model_evaluation"
BEST_MODEL_KEY = "best_model"
FILE_PATH_KEY = "file_path"
HISTORY_KEY = "history"


# Model Pusher:
MODEL_PUSHER_CONFIG_KEY = "model_pusher_config"
MODEL_PUSHER_EXPORT_MODEL_DIR_KEY = "export_model_dir"


# Experiment:
EXPERIMENT_CONFIG_KEY = "experiment_config"
EXPERIMENT_DIR_KEY = "experiment_dir"
EXPERIMENT_FILE_NAME_KEY = "experiment_file_name"

COLUMNS = ["status",
                "duration",
                "credit_history",
                "purpose",
                "amount" ,
                "savings",
                "employment_duration",
                "installment_rate",
                "personal_status_sex",
                "other_debtors",
                "present_residence",
                "property",
                "age",
                "other_installment_plans",
                "housing",
                "number_credits",
                "job",
                "people_liable",
                "telephone",
                "foreign_worker",
                "credit_risk"]