from banking.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvalutaionConfig, ModelPusherConfig
from banking.constant import *
from banking.utils.util import read_yaml_data
from banking.exception import BankingException
from banking.logger import logging
import os, sys
from datetime import datetime


class Configuration:

    def __init__(self, 
        config_file_path:str = CONFIG_FILE_PATH,
        current_time_stamp:str = CURRENT_TIME_STAMP
        ) -> None:
        
        """
        This class is responsible for initiating the Confirguration of the Pipeline.
        Parameters
        ----------
        config_file_path : str
            File path for the config.yaml file.
        current_time_stamp : str
            Current time stamp in string format.
        
        Attributes
        ----------
        config_file_info : dict
            Content extracted from config file for the file path passed.
        current_time_stamp : str
            Current time stamp in string format.
        training_pipeline_config : namedtuple
            Named tuple for training pipeline configuration.
        """
        try:
            self.config_file_info = read_yaml_data(file_path=config_file_path)
            self.current_time_stamp = current_time_stamp

            self.training_pipeline_config = self.get_training_pipeline_config()
            logging.info(f"Training Pipeline Configuration: {self.training_pipeline_config}")
        except Exception as e:
            raise BankingException(e, sys) from e


    def get_training_pipeline_config(self) -> TrainingPipelineConfig:
        """
        This function is responsible for generating a named tuple for the
        training pipeline configuration.
        Returns
        -------
        training_pipeline_config : namedtuple
            Named tuple for the training pipeline configuration.
        """
        try:
            training_pipeline_config_info =  self.config_file_info[TRAINING_PIPELINE_CONFIG_KEY]
            root_artifact_dir = os.path.join(
                ROOT_DIR,
                training_pipeline_config_info[TRAINING_PIPELINE_NAME_KEY],
                training_pipeline_config_info[TRAINING_PIPELINE_ROOT_ARTIFACT_DIR_KEY]
            )
            training_pipeline_config = TrainingPipelineConfig(root_artifact_dir=root_artifact_dir)
            logging.info(f"Training Pipeline Config: {training_pipeline_config}")
            return training_pipeline_config
        except Exception as e:
            raise BankingException(e, sys) from e


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        This function is responsible for generating a named tuple for the
        data ingestion configuration.
        Returns
        -------
        data_ingestion_config : namedtuple
            Named tuple for the data ingestion configuration.
        """
        try: 
            logging.info(f"Extracting the Data Ingestion Configuration.")
            data_ingestion_config_info = self.config_file_info[DATA_INGESTION_CONFIG_KEY]
            root_artifact_dir = self.training_pipeline_config.root_artifact_dir
            data_ingestion_artifact_dir = os.path.join(
                root_artifact_dir,
                DATA_INGESTION_ARTIFACT_DIR,
                self.current_time_stamp
            )
            zip_download_dir = os.path.join(
                data_ingestion_artifact_dir,
                data_ingestion_config_info[DATA_INGESTION_ZIP_DOWNLOAD_DIR_KEY]
            )
            extracted_data_dir = os.path.join(
                data_ingestion_artifact_dir,
                data_ingestion_config_info[DATA_INGESTION_EXTRACTED_DATA_DIR]
            )
            raw_data_dir = os.path.join(
                data_ingestion_artifact_dir,
                data_ingestion_config_info[DATA_INGESTION_RAW_DATA_DIR_KEY]
            )
            ingested_data_dir = os.path.join(
                data_ingestion_artifact_dir,
                data_ingestion_config_info[DATA_INGESTION_INGESTED_DATA_DIR_KEY]
            )
            ingested_train_dir = os.path.join(
                ingested_data_dir,
                data_ingestion_config_info[DATA_INGESTION_INGESTED_TRAIN_DIR_KEY]
            )
            ingested_test_dir = os.path.join(
                ingested_data_dir,
                data_ingestion_config_info[DATA_INGESTION_INGESTED_TEST_DIR_KEY]
            )
            data_ingestion_config = DataIngestionConfig(
                dataset_download_url=data_ingestion_config_info[DATA_INGESTION_DATASET_DOWNLOAD_URL_KEY],
                zip_download_dir=zip_download_dir,
                extracted_data_dir=extracted_data_dir,
                raw_data_dir=raw_data_dir,
                ingested_train_dir=ingested_train_dir,
                ingested_test_dir=ingested_test_dir
            )
            logging.info(f"Data Ingestion Configuration Extracted Successfully: \n{data_ingestion_config}")
            return data_ingestion_config
        except Exception as e:
            raise BankingException(e, sys) from e


    def get_data_validation_config(self) -> DataValidationConfig:
        """
        This function is responsible for generating a named tuple for the
        data validation configuration.
        Returns
        -------
        data_validation_config : namedtuple
            Named tuple for the data validation configuration.
        """
        try:
            logging.info(f"Extracting the Data Validation Configuration.")
            data_validation_config_info = self.config_file_info[DATA_VALIDATION_CONFIG_KEY]
            
            data_validation_artifact_dir = os.path.join(
                self.training_pipeline_config.root_artifact_dir,
                DATA_VALIDATION_ARTIFACT_DIR,
                self.current_time_stamp
            )

            schema_file_path = os.path.join(
                data_validation_config_info[DATA_VALIDATION_SCHEMA_FILE_DIR_KEY],
                data_validation_config_info[DATA_VALIDATION_SCHEMA_FILE_NAME_KEY]
            )

            report_file_path = os.path.join(
                data_validation_artifact_dir,
                data_validation_config_info[DATA_VALIDATION_REPORT_FILE_NAME_KEY]
            )

            report_page_file_path = os.path.join(
                data_validation_artifact_dir,
                data_validation_config_info[DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY]
            )

            data_validation_config = DataValidationConfig(
                schema_file_path=schema_file_path,
                report_file_path=report_file_path,
                report_page_file_path=report_page_file_path
            )
            logging.info(f"Data Validation Configuration Extracted Successfully: \n{data_validation_config}")
            return data_validation_config

        except Exception as e:
            raise BankingException(e, sys) from e

        
    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        This function is responsible for generating a named tuple for the
        data transformation configuration.
        Returns
        -------
        data_transformation_config : namedtuple
            Named tuple for the data transformation configuration.
        """
        try:
            logging.info(f"Extracting the Data Transformation Configuration.")
            data_transformation_config_info = self.config_file_info[DATA_TRANSFORMATION_CONFIG_KEY]

            data_transformation_artifact_dir = os.path.join(
                self.training_pipeline_config.root_artifact_dir,
                DATA_TRANSFORMATION_ARTIFACT_DIR,
                self.current_time_stamp
            )

            transformed_train_dir = os.path.join(
                data_transformation_artifact_dir,
                data_transformation_config_info[DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR_KEY],
                data_transformation_config_info[DATA_TRANSFORMATION_TRANSFORMED_TRAIN_DIR_KEY]
            )

            transformed_test_dir = os.path.join(
                data_transformation_artifact_dir,
                data_transformation_config_info[DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR_KEY],
                data_transformation_config_info[DATA_TRANSFORMATION_TRANSFORMED_TEST_DIR_KEY]
            )

            preprocessed_model_object_file_path = os.path.join(
                data_transformation_artifact_dir,
                data_transformation_config_info[DATA_TRANSFORMATION_PREPROCESSED_MODEL_DIR_KEY],
                data_transformation_config_info[DATA_TRANSFORMATION_PREPROCESSED_MODEL_OBJECT_FILE_NAME_KEY]
            )

            data_transformation_config = DataTransformationConfig(
                transformed_train_dir=transformed_train_dir,
                transformed_test_dir=transformed_test_dir,
                preprocessed_model_object_file_path=preprocessed_model_object_file_path
            )

            logging.info(f"Data Transformation Configuration Extracted Successfully: \n{data_transformation_config}")
            return data_transformation_config

        except Exception as e:
            raise BankingException(e, sys) from e


    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        This function is responsible for generating a named tuple for the
        model trainer configuration.
        Returns
        -------
        model_trainer_config : namedtuple
            Named tuple for the model trainer configuration.
        """
        try:
            logging.info(f"Extracting the Model Trainer Configuration.")
            model_trainer_config_info = self.config_file_info[MODEL_TRAINER_CONFIG_KEY]

            model_trainer_artifact_dir = os.path.join(
                self.training_pipeline_config.root_artifact_dir,
                MODEL_TRAINER_ARTIFACT_DIR,
                self.current_time_stamp
            )

            trained_model_file_path = os.path.join(
                model_trainer_artifact_dir,
                model_trainer_config_info[MODEL_TRAINER_TRAINED_MODEL_DIR_KEY],
                model_trainer_config_info[MODEL_TRAINER_TRAINED_MODEL_FILE_NAME_KEY]
            )

            base_accuracy = model_trainer_config_info[MODEL_TRAINER_BASE_ACCURACY_KEY]

            model_config_file_path = os.path.join(
                model_trainer_config_info[MODEL_TRAINER_MODEL_CONFIG_DIR_KEY],
                model_trainer_config_info[MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY]
            )

            model_trainer_config = ModelTrainerConfig(
                trained_model_file_path=trained_model_file_path,
                base_accuracy=base_accuracy,
                model_config_file_path=model_config_file_path
            )
            
            logging.info(f"Model Trainer Configuration: [{model_trainer_config}].")
            return model_trainer_config
        except Exception as e:
            raise BankingException(e, sys) from e


    def get_model_evaluation_config(self) -> ModelEvalutaionConfig:
        """
        This function is responsible for generating a named tuple for the
        model evaluation configuration.
        Returns
        -------
        model_evaluation_config : namedtuple(ModelEvalutaionConfig)
            Named tuple for the model evaluation configuration.
        """
        try:
            logging.info(f"Extracting the Model Evaluation Configuration.")
            model_evaluation_config_info = self.config_file_info[MODEL_EVALUATION_CONFIG_KEY]

            # We are not creating a time stamp folder as
            # this model evaluation report is globally accessed and should be
            # up to date with the latest model details.
            # Also older models' details are already stored under the history key
            model_evaluation_artifact_dir = os.path.join(
                self.training_pipeline_config.root_artifact_dir,
                MODEL_EVALUATION_ARTIFACT_DIR
            )

            model_evaluation_report_file_path = os.path.join(
                model_evaluation_artifact_dir,
                model_evaluation_config_info[MODEL_EVALUATION_MODEL_EVALUATION_REPORT_FILE_NAME_KEY]
            )

            # We need the current time stamp as if we have to update the 
            # model_evaluation_report_file to add a model to the history path
            # then we will add a new key that is history and the time stamp 
            # at which the model was trained
            current_time_stamp = self.current_time_stamp

            model_evaluation_config = ModelEvalutaionConfig(
                model_evaluation_report_file_path=model_evaluation_report_file_path,
                current_time_stamp=current_time_stamp
            )
            logging.info(f"Model Evaluation Configuration: [{model_evaluation_config}].")
            return model_evaluation_config
        except Exception as e:
            raise BankingException(e, sys) from e
            
    
    def get_model_pusher_config(self) ->ModelPusherConfig:
        """
        This function is responsible for generating a named tuple for the
        model pusher configuration.
        Returns
        -------
        model_pusher_config : namedtuple(ModelPusherConfig)
            Named tuple for the model pusher configuration.
        """
        try:
            logging.info(f"Extracting the Model Pusher Configuration.")
            model_pusher_config_info = self.config_file_info[MODEL_PUSHER_CONFIG_KEY]
            
            current_time_stamp = f"{datetime.now().strftime('%Y%m%d%H%M%S')}"

            export_model_dir = os.path.join(
                ROOT_DIR,
                model_pusher_config_info[MODEL_PUSHER_EXPORT_MODEL_DIR_KEY],
                current_time_stamp
            )

            model_pusher_config = ModelPusherConfig(
                export_model_dir=export_model_dir
            )
            logging.info(f"Model Pusher Configuration: [{model_pusher_config}].")
            return model_pusher_config
        except Exception as e:
            raise BankingException(e, sys) from e

