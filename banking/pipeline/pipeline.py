import os, sys
import numpy as np
import pandas as pd
import uuid
from datetime import datetime
from threading import Thread
from banking.logger import logging
from banking.exception import BankingException
from banking.entity.config_entity import DataIngestionConfig, DataValidationConfig, ModelTrainerConfig, ModelEvalutaionConfig, ModelPusherConfig, Experiment
from banking.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact, ModelPusherArtifact
from banking.config.configuration import Configuration
from banking.component.data_ingestion import DataIngestion
from banking.component.data_validation import DataValidation
from banking.component.data_transformation import DataTransformation
from banking.component.model_trainer import ModelTrainer
from banking.component.model_evaluation import ModelEvaluation
from banking.component.model_pusher import ModelPusher
from banking.constant import *

class Pipeline(Thread):

    experiment : Experiment = Experiment(*([None]*11))
    experiment_file_path = None

    def __init__(self, config=Configuration()) -> None:
        try:
            super().__init__(daemon=False, name='pipeline')

            experiment_info = config.config_file_info[EXPERIMENT_CONFIG_KEY]
            Pipeline.experiment_file_path = os.path.join(
                config.training_pipeline_config.root_artifact_dir,
                experiment_info[EXPERIMENT_DIR_KEY],
                experiment_info[EXPERIMENT_FILE_NAME_KEY]
            )
            os.makedirs(os.path.dirname(Pipeline.experiment_file_path), exist_ok=True)

            self.config = config

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

    def start_model_evaluation(self,
            data_ingestion_artifact:DataIngestionArtifact,
            data_validation_artifact:DataValidationArtifact,
            model_trainer_artifact:ModelTrainerArtifact
        ) -> ModelEvaluationArtifact:
        try:
            model_evaluation = ModelEvaluation(
                                    model_evaluation_config = self.config.get_model_evaluation_config(),
                                    data_ingestion_artifact=data_ingestion_artifact,
                                    data_validation_artifact=data_validation_artifact,
                                    model_trainer_artifact=model_trainer_artifact
                                )
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            return model_evaluation_artifact
        except Exception as e:
            raise BankingException(e, sys) from e

    

    def run_pipeline(self):
        try:
            if Pipeline.experiment.running_status:
                logging.info(f"Pipeline already running. Wait for the completing of existing Pipeline")
                print("Pipeline already runnig")
                return Pipeline.experiment
            

            experiment_id= f"{uuid.uuid4()}"
            start_time= datetime.now()
            Pipeline.experiment = Experiment(
                                    experiment_id= experiment_id,
                                    running_status= True,
                                    experiment_file_path= Pipeline.experiment_file_path,
                                    initialized_time_stamp= self.config.current_time_stamp,
                                    artifact_time_stamp= self.config.current_time_stamp,
                                    start_time= f"{start_time.strftime('%Y-%m-%d-%H-%M-%S')}",
                                    stop_time= f"-",
                                    execution_time= f"-",
                                    accuracy= f"-",
                                    is_trained_model_accepted= f"-",
                                    message= "Training Pipeline Started"
                                )

            logging.info(f"Pipeline experiment: {Pipeline.experiment}")
            
            Pipeline.save_experiment()

            logging.info(f"Initiating the Pipeline.")
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(
                                                data_ingestion_artifact=data_ingestion_artifact
                                            )
            data_transformation_artifact = self.start_data_transformation(
                                                    data_ingestion_artifact=data_ingestion_artifact, 
                                                    data_validation_artifact=data_validation_artifact
                                                )
            model_trainer_artifact = self.start_model_trainer(
                                                    data_transformation_artifact=data_transformation_artifact
                                                )
            # print(model_trainer_artifact)
            # print("\n\n")
            model_evaluation_artifact = self.start_model_evaluation(
                                                    data_ingestion_artifact=data_ingestion_artifact,
                                                    data_validation_artifact=data_validation_artifact,
                                                    model_trainer_artifact=model_trainer_artifact
                                                )
            # print(model_evaluation_artifact)
            model_pusher_artifact = None
            if model_evaluation_artifact.is_trained_model_accepted:
                model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)
                logging.info("Trained Model Accepted")
            else:
                logging.info("Trained Model Rejected")
            print(model_pusher_artifact)
            logging.info(f"Model Pusher Artifact:\n{model_pusher_artifact}")

            stop_time= datetime.now()
            execution_time = stop_time-start_time
            Pipeline.experiment = Experiment(
                                    experiment_id= Pipeline.experiment.experiment_id,
                                    running_status= False,
                                    experiment_file_path= Pipeline.experiment_file_path,
                                    initialized_time_stamp= Pipeline.experiment.initialized_time_stamp,
                                    artifact_time_stamp= Pipeline.experiment.artifact_time_stamp,
                                    start_time= Pipeline.experiment.start_time,
                                    stop_time= f"{stop_time.strftime('%Y-%m-%d-%H-%M-%S')}",
                                    execution_time= f"{execution_time.seconds}.{np.round(execution_time.microseconds,1)} seconds",
                                    accuracy= model_trainer_artifact.model_accuracy,
                                    is_trained_model_accepted= model_evaluation_artifact.is_trained_model_accepted,
                                    message= f"Training Pipeline Completed"
                                )

            logging.info(f"Pipeline experiment: {Pipeline.experiment}")
            
            Pipeline.save_experiment()

            logging.info(f"{'='*60}Pipeline Completed.{'='*60}\n\n")
            
        except Exception as e:
            raise BankingException(e,sys) from e

