from banking.logger import logging
from banking.exception import BankingException
from banking.entity.config_entity import ModelEvalutaionConfig
from banking.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, DataValidationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact
from banking.utils.util import read_yaml_data, load_df_from_csv, load_model_object_from_file, write_yaml_data
from banking.constant import *
from banking.entity.model_factory import ModelFactory
import os, sys



class ModelEvaluation:
    
    def __init__(self,
            model_evaluation_config: ModelEvalutaionConfig,
            data_ingestion_artifact: DataIngestionArtifact,
            data_validation_artifact: DataValidationArtifact,
            model_trainer_artifact: ModelTrainerArtifact
        ) -> None:
        """
        This class is a Component and is responsible for initiating the 
        Model Evaluation phase of the Pipeline.
        Parameters
        ----------
        model_evaluation_config : namedtuple
            Named tuple for Model Evaluation Configuration.
        data_ingestion_artifact : namedtuple
            Named tuple for Data Ingestion Artifact.
        data_validation_artifact : namedtuple
            Named tuple for Data Validation Artifact.
        model_trainer_artifact : namedtuple
            Named tuple for Model Trainer Artifact.
        Attributes
        ----------
        model_evaluation_config : namedtuple
            Named tuple for Model Evaluation Configuration.
        data_ingestion_artifact : namedtuple
            Named tuple for Data Ingestion Artifact.
        data_validation_artifact : namedtuple
            Named tuple for Data Validation Artifact.
        model_trainer_artifact : namedtuple
            Named tuple for Model Trainer Artifact.
        """
        try:
            logging.info(f"{'='*60}Model Evaluation Log Started.{'='*60}")
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise BankingException(e, sys) from e

    def update_model_evaluation_report_file(self, model_evaluation_artifact: ModelEvaluationArtifact):
        """
        This function is responsible for updating the model evaluation report file.
        Parameters:
        --------
        model_evaluation_artifact: ModelEvaluationArtifact
            Named tuple consisting the details about the model evaluation artifact.
        """
        try:
            model_eval_report_file_path = self.model_evaluation_config.model_evaluation_report_file_path
            model_eval_report_info = read_yaml_data(file_path=model_eval_report_file_path)

            model_eval_report_info = dict() if model_eval_report_info is None else model_eval_report_info

            previous_best_model = None
            if BEST_MODEL_KEY in model_eval_report_info:
                previous_best_model = model_eval_report_info[BEST_MODEL_KEY]

            logging.info(f"Previous eval result: {model_eval_report_info}")
            new_eval_report_info = {
                BEST_MODEL_KEY: {
                    FILE_PATH_KEY: model_evaluation_artifact.evaluated_model_file_path
                }
            }

            if previous_best_model is not None:
                model_history = {
                    self.model_evaluation_config.current_time_stamp: previous_best_model
                }
                if HISTORY_KEY not in model_eval_report_info:
                    new_history = {
                        HISTORY_KEY: model_history
                    }
                    new_eval_report_info.update(new_history)
                else:
                    model_eval_report_info[HISTORY_KEY].update(model_history)

            model_eval_report_info.update(new_eval_report_info)
            logging.info(f"Updated eval result:{model_eval_report_info}")
            write_yaml_data(file_path=model_eval_report_file_path, data=model_eval_report_info)
        except Exception as e:
            raise BankingException(e, sys) from e

    def get_best_model(self) -> object:
        """
        This function is responsible for finding the model which is already running in production if exists.
        Returns:
        --------
        model : object
            Model object of the best model running in production.
        """
        try:
            # Assuming that model evaluation report file does not exist.
            # Also then no model in production.
            model = None
            model_evaluation_report_info = None
            
            model_evaluation_report_file_path = self.model_evaluation_config.model_evaluation_report_file_path
            
            if not os.path.exists(model_evaluation_report_file_path):
                # Creating an empty model evaluation report yaml file.
                write_yaml_data(file_path=model_evaluation_report_file_path, data=model_evaluation_report_info)
                return model

            model_evaluation_report_info = read_yaml_data(model_evaluation_report_file_path)
            # If model evaluation report file exists then
            # if content is empty so assign empty dict()
            # else load existing content
            model_evaluation_report_info =  dict() if model_evaluation_report_info is None else model_evaluation_report_info

            # Checking if there is no content in model evaluation report file
            if BEST_MODEL_KEY not in model_evaluation_report_info:
                # write_yaml_data(file_path=model_evaluation_report_file_path, data=model_evaluation_report_info)
                return model

            model = load_model_object_from_file(model_evaluation_report_info[BEST_MODEL_KEY][FILE_PATH_KEY])

            return model
        except Exception as e:
            raise BankingException(e, sys) from e


    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        This function is responsible for initiating the model evaluation phase in the pipeline.
        Returns:
        --------
        model_evaluation_artifact : ModelEvaluationArtifact
            Namedtuple containing the details about the model evaluation artifact.
        """
        try:
            train_data_file_path = self.data_ingestion_artifact.train_data_file_path
            test_data_file_path = self.data_ingestion_artifact.test_data_file_path

            logging.info(f"Loading training and testing file as dataframes")
            train_df = load_df_from_csv(train_data_file_path) 
            test_df = load_df_from_csv(test_data_file_path)

            schema_file_info = read_yaml_data(file_path=self.data_validation_artifact.schema_file_path)

            target_column = schema_file_info[SCHEMA_OLD_TARGET_COLUMN_KEY]

            logging.info(f"Extracting features and label for training and testing dataframes")
            x_train = train_df.drop(columns=target_column)
            y_train = train_df[target_column]

            x_test = test_df.drop(columns=target_column)
            y_test = test_df[target_column]

            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path

            logging.info(f"Acquiring the trained model for evaluation from: [{trained_model_file_path}]")
            trained_model_object = load_model_object_from_file(trained_model_file_path)
            production_model_object = self.get_best_model()

            if production_model_object is None:
                # No current model in production so accepting trained_model
                message = f"Accepting Trained model as no model in production."
                logging.info(message)
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_trained_model_accepted=True,
                    message=message,
                    evaluated_model_file_path=trained_model_file_path
                )
                self.update_model_evaluation_report_file(model_evaluation_artifact=model_evaluation_artifact)
                return model_evaluation_artifact
            
            
            model_list = [production_model_object, trained_model_object]
            
            metrics_info_artifact = ModelFactory.evaluate_classification_model(
                                                    model_list=model_list,
                                                    X_train=x_train,
                                                    y_train=y_train,
                                                    X_test=x_test,
                                                    y_test=y_test,
                                                    base_accuracy=self.model_trainer_artifact.model_accuracy
                                                )

            if metrics_info_artifact is None:
                # No model has accuracy better than the base accuracy
                message = f"No model available to be productionized."
                logging.info(message)
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_trained_model_accepted=False,
                    message=message,
                    evaluated_model_file_path=trained_model_file_path
                )

                return model_evaluation_artifact

            if metrics_info_artifact.model_index_number == 1:
                # Trained Model accepted
                message = "Trained model accepted as it performed better that the current model in production"
                logging.info(message)
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_trained_model_accepted=True,
                    message=message,
                    evaluated_model_file_path=trained_model_file_path
                )
                self.update_model_evaluation_report_file(model_evaluation_artifact=model_evaluation_artifact)
                
            else:
                # Trained Model did not perform better than the current model in production
                message=f"Trained model did not perform well so rejecting the model"
                logging.info(message)
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_trained_model_accepted=False,
                    message=message,
                    evaluated_model_file_path=trained_model_file_path
                )
            logging.info(f"Model Evaluation Artifact: [{model_evaluation_artifact}]")
            return model_evaluation_artifact                

        except Exception as e:
            raise BankingException(e, sys) from e

    def __del__(self) -> None:
        try:
            logging.info(f"{'='*60}Model Evaluation Log Completed.{'='*60}\n\n")
        except Exception as e:
            raise BankingException(e, sys) from e
