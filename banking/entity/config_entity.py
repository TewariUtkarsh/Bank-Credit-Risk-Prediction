from collections import namedtuple


# Training Pipeline Config
TrainingPipelineConfig = namedtuple(
    "TrainingPipelineConfig",
    ["root_artifact_dir"]
)


# Data Ingestion Config
DataIngestionConfig = namedtuple(
    "DataIngestionConfig",
    ["dataset_download_url", "zip_download_dir", "extracted_data_dir","raw_data_dir", "ingested_train_dir", "ingested_test_dir"]
)


# Data Validation Config
DataValidationConfig = namedtuple(
    "DataValidationConfig",
    ["schema_file_path", "report_file_path", "report_page_file_path"]
)


# Data Transformation Config
DataTransformationConfig = namedtuple(
    "DataTransformationConfig",
    ["transformed_train_dir", "transformed_test_dir", "preprocessed_model_object_file_path"]
)
