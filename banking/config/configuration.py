import imp
from banking.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig
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



