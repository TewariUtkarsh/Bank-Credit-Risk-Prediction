from collections import namedtuple



# Data Ingestion Artifact
DataIngestionArtifact = namedtuple(
    "DataIngestionArtifact",
    ["is_ingested", "message", "train_data_file_path", "test_data_file_path"]
)


# Data Validation Artifact
DataValidationArtifact = namedtuple(
    "DataValidationArtifact",
    ["is_validated", "message", "schema_file_path", "report_file_path", "report_page_file_path"]
)


# Data Transformation Artifact
DataTransformationArtifact = namedtuple(
    "DataTransformationArtifact",
    ["is_transformed", "message", "transformed_train_data_file_path", "transformed_test_data_file_path", "preprocessed_model_object_file_path"]
)

