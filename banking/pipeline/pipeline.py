import os, sys
from banking.logger import logging
from banking.exception import BankingException
from banking.entity.config_entity import DataIngestionConfig, DataValidationConfig
from banking.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from banking.config.configuration import Configuration
from banking.component.data_ingestion import DataIngestion
from banking.component.data_validation import DataValidation

class Pipeline:

    def __init__(self) -> None:
        self.config = Configuration()
        

    def start_data_ingestion(self) -> DataIngestionArtifact:
        data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        return data_ingestion_artifact

    def start_data_validation(self, DataIngestionArtifact: DataIngestionArtifact) -> DataValidationArtifact:
        data_validation = DataValidation(DataValidationConfig=self.config.get_data_validation_config(), DataIngestionArtifact=DataIngestionArtifact)
        data_validation_artifact = data_validation.initiate_data_validation()
        return data_validation_artifact


    def run_pipeline(self):
        logging.info(f"Initiating the Pipeline.")
        data_ingestion_artifact = self.start_data_ingestion()
        data_validation_artifact = self.start_data_validation(DataIngestionArtifact=data_ingestion_artifact)

    def __del__(self):
        pass