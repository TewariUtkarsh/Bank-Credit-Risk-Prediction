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



