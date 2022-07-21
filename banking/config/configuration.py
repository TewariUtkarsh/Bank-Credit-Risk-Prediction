from banking.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig
from banking.constant import *
from banking.utils.util import read_yaml_data
from banking.exception import BankingException
from banking.logger import logging
import os, sys


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

        



