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




