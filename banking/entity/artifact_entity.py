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


# Model Trainer Aritfact
ModelTrainerArtifact = namedtuple(
    "ModelTrainerArtifact",
    ["is_trained", "message", "trained_model_file_path", "train_accuracy", "test_accuracy","model_accuracy",
     "train_f1_score", "test_f1_score", "train_precision_score", "test_precision_score", "train_recall_score", "test_recall_score"]
)


# Model Evaluation Artifact
ModelEvaluationArtifact = namedtuple(
    "ModelEvaluationArtifact",
    ["is_trained_model_accepted", "message", "evaluated_model_file_path"]
)                                                
                                            

# Model Pusher Artifact
ModelPusherArtifact = namedtuple(
    "ModelPusherArtifact",
    ["export_model_file_path"]
)                                            