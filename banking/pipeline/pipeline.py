import os, sys
from banking.logger import logging
from banking.exception import BankingException
from banking.entity.config_entity import DataIngestionConfig
from banking.entity.artifact_entity import DataIngestionArtifact
from banking.config.configuration import Configuration
from banking.component.data_ingestion import DataIngestion

class Pipeline:

    def __init__(self) -> None:
        self.config = Configuration()
        

    def start_data_ingestion(self) -> DataIngestionArtifact:
        
        data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        return data_ingestion_artifact


    def run_pipeline(self):
        data_ingestion_artifact = self.start_data_ingestion()

    def __del__(self):
        pass