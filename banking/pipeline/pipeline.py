import os, sys
from banking.logger import logging
from banking.exception import BankingException
from banking.entity.config_entity import DataIngestionConfig, DataValidationConfig, ModelTrainerConfig
from banking.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact
from banking.config.configuration import Configuration
from banking.component.data_ingestion import DataIngestion
from banking.component.data_validation import DataValidation
from banking.component.data_transformation import DataTransformation
from banking.component.model_trainer import ModelTrainer

class Pipeline:

    def __init__(self) -> None:
        try:
            self.config = Configuration()
        except Exception as e:
            raise BankingException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            return data_ingestion_artifact
        except Exception as e:
            raise BankingException(e,sys) from e

    def start_data_validation(self, 
            data_ingestion_artifact: DataIngestionArtifact
        ) -> DataValidationArtifact:
        try:
            data_validation = DataValidation(data_validation_config=self.config.get_data_validation_config(), data_ingestion_artifact=data_ingestion_artifact)
            data_validation_artifact = data_validation.initiate_data_validation()
            return data_validation_artifact
        except Exception as e:
            raise BankingException(e,sys) from e

    def start_data_transformation(self, 
            data_ingestion_artifact: DataIngestionArtifact,
            data_validation_artifact: DataValidationArtifact
        ) -> DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(
                data_transformation_config= self.config.get_data_transformation_config(),
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise BankingException(e,sys) from e

    def start_model_trainer(self,
            data_transformation_artifact: DataTransformationArtifact
        ) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(
                                model_trainer_config=self.config.get_model_trainer_config(),
                                data_transformation_artifact=data_transformation_artifact
                            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            raise BankingException(e, sys) from e

        

    def run_pipeline(self):
        try:
            logging.info(f"Initiating the Pipeline.")
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact, data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)

            logging.info(f"Model Trainer Artifact:\n{model_trainer_artifact}")

        except Exception as e:
            raise BankingException(e,sys) from e


    def __del__(self):
        pass