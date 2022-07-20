from collections import namedtuple



# Data Ingestion Artifact
DataIngestionArtifact = namedtuple(
    "DataIngestionArtifact",
    ["is_ingested", "message", "train_data_file_path", "test_data_file_path"]
)
