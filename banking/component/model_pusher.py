from banking.exception import BankingException
from banking.logger import logging
from banking.entity.config_entity import ModelPusherConfig
from banking.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from banking.constant import *
import os, sys
import shutil


class ModelPusher:

    def __init__(self,
            model_pusher_config: ModelPusherConfig,
            model_evaluation_artifact: ModelEvaluationArtifact
        ) -> None:
        """
        This class is a Component and is responsible for initiating the 
        Model Evaluation phase of the Pipeline.
        Parameters
        ----------
        model_pusher_config : namedtuple
            Named tuple for Model Pusher Configuration.
        model_evaluation_artifact : namedtuple
            Named tuple for Model Evaluation Artifact.
        Attributes
        ----------
        model_pusher_config : namedtuple
            Named tuple for Model Pusher Configuration.
        model_evaluation_artifact : namedtuple
            Named tuple for Model Evaluation Artifact.
        """
        try:
            logging.info(f"{'='*60}Model Pusher Log Started.{'='*60}")
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact
        except Exception as e:
            raise BankingException(e, sys) from e

    
    def export_model(self) -> str:
        """
        This function is responsible for exporting the evaluated model from the file path specified.
        Returns:
        --------
        export_model_file_path : str
            File path of the evaluated model object.
        """
        try:
            evaluated_model_file_path = self.model_evaluation_artifact.evaluated_model_file_path

            evaluated_model_file_name = os.path.basename(evaluated_model_file_path)
            
            export_model_file_path = os.path.join(
                ROOT_DIR,
                self.model_pusher_config.export_model_dir,
                evaluated_model_file_name
            )
            logging.info(f"Exporting model file: [{export_model_file_path}]")
            os.makedirs(os.path.dirname(export_model_file_path), exist_ok=True)

            shutil.copy(evaluated_model_file_path, export_model_file_path)
            logging.info(f"Trained model: {evaluated_model_file_path} is copied in export dir:[{export_model_file_path}]")
            return export_model_file_path
        except Exception as e:
            raise BankingException(e, sys) from e

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        This function is responsible for initiating the model pusher phase in the pipeline.
        Returns:
        --------
        model_pusher_artifact : ModelPusherArtifact
            Namedtuple containing the details about the model pusher artifact.
        """
        try:
            export_model_file_path = self.export_model()  
            model_pusher_artifact = ModelPusherArtifact(
                                        export_model_file_path=export_model_file_path
                                    )      
            logging.info(f"Model Pusher Artifact: [{model_pusher_artifact}]")
            return model_pusher_artifact
        except Exception as e:
            raise BankingException(e, sys) from e


    def __del__(self) -> None:
        try:
            logging.info(f"{'='*60}Model Pusher Log Completed.{'='*60}\n\n")
        except Exception as e:
            raise BankingException(e, sys) from e