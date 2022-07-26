from banking.logger import logging
from banking.exception import BankingException
from banking.entity.config_entity import ModelTrainerConfig
from banking.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from banking.utils.util import load_numpy_array_from_file, load_model_object_from_file, save_model_object_to_file
from banking.entity.model_factory import GridSearchedModel, MetricsInfoArtifact, ModelFactory
import os, sys
from imblearn.over_sampling import SMOTE
import numpy as np
from typing import List




class BankingEstimator:
    def __init__(self, 
            trained_model_object, 
            preprocessed_model_object
        ) -> None:
        self.trained_model_object = trained_model_object
        self.preprocessed_model_object = preprocessed_model_object

    def predict(self, X:np.array):
        X_transformed = self.preprocessed_model_object.transform(X)
        return self.trained_model_object.predict(X_transformed)

    def __repr__(self) -> str:
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self) -> str:
        return f"{type(self.trained_model_object).__name__}()"



class ModelTrainer:

    def __init__(self, 
            model_trainer_config: ModelTrainerConfig,
            data_transformation_artifact: DataTransformationArtifact
        ) -> None:
        """
        This class is a Component and is responsible for initiating the 
        Model Trainer phase of the Pipeline.
        Parameters
        ----------
        model_trainer_config : namedtuple
            Named tuple for Model Trainer Configuration.
        
        Attributes
        ----------
        data_transformation_artifact : dict
            Named tuple for Data Transformation Artifact.
        """
        try:
            logging.info(f"{'='*60}Model Trainer Log Started.{'='*60} ")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise BankingException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            """
            This function is responsible for initiating the model trainer phase in the
            pipeline.
            Returns:
            --------
            ModelTrainerArtifact : namedtuple
                Named tuple consisting the artifact related details of Model Trainer Phase.
            """
            model_config_file_path = self.model_trainer_config.model_config_file_path
            # model_config_file_info = read_yaml_data(model_config_file_path)
            base_accuracy = self.model_trainer_config.base_accuracy
            
            logging.info(f"Creating a Directory to store our trained model.")
            trained_model_file_path = self.model_trainer_config.trained_model_file_path
            os.makedirs(os.path.dirname(trained_model_file_path), exist_ok=True)

            train_data_file_path = self.data_transformation_artifact.transformed_train_data_file_path
            test_data_file_path = self.data_transformation_artifact.transformed_test_data_file_path

            logging.info(f"Loading training and testing numpy data.")
            train_arr = load_numpy_array_from_file(train_data_file_path)
            test_arr = load_numpy_array_from_file(test_data_file_path)


            logging.info(f"Creating features and labels for training and testing data.")
            train_arr_features = train_arr[:, :-1]
            test_arr_features = test_arr[:, :-1]

            train_arr_labels = train_arr[:,-1]
            test_arr_labels = test_arr[:, -1]

            # SMOTE for imbalanced dataset
            logging.info(f"Resampling our training and testing data using SMOTE into training and testing features and labels.")
            smote = SMOTE(random_state=42)
            
            X_train, y_train = smote.fit_resample(train_arr_features, train_arr_labels)
            X_test, y_test = smote.fit_resample(test_arr_features, test_arr_labels)

            # Model Building:
            model_factory = ModelFactory(model_config_file_path=model_config_file_path)

            
            best_model = model_factory.get_best_model(
                                            X_train=X_train, y_train=y_train,
                                            base_accuracy=base_accuracy
                                        )
            logging.info(f"Training and acquiring the best model from training data: [{best_model}].")

            grid_searched_model_list:List[GridSearchedModel] = model_factory.grid_searched_model_list

            model_list:List[object] = [model.best_model for model in grid_searched_model_list]

            metric_info_artifact:MetricsInfoArtifact = ModelFactory.evaluate_classification_model(
                                                                    model_list=model_list,
                                                                    X_train=X_train,
                                                                    y_train=y_train,
                                                                    X_test=X_test,
                                                                    y_test=y_test,
                                                                    base_accuracy=base_accuracy
                                                                )

            logging.info(f"Metrics info about the best model: [{metric_info_artifact}]")
            # model_list = grid_search
            # metrics = evalute(model_list)
            # trained_model_object = metrics.model along with accuracies and other stuffs

            trained_model_object = metric_info_artifact.model_object

            
            preprocessed_model_object = load_model_object_from_file(self.data_transformation_artifact.preprocessed_model_object_file_path)
            logging.info(f"Loading the preprocessed model object: [{preprocessed_model_object}].")

            banking_estimator_model_object = BankingEstimator(trained_model_object=trained_model_object, preprocessed_model_object=preprocessed_model_object)

            # Saving Model
            logging.info(f"Saving the final model object: [{banking_estimator_model_object}].")
            save_model_object_to_file(model=banking_estimator_model_object, file_path=trained_model_file_path)

            is_trained = True
            message = f"Model Trainer Phase Completed Successfully."

            model_trainer_artifact = ModelTrainerArtifact(
                is_trained=is_trained,
                message=message,
                trained_model_file_path=trained_model_file_path,
                train_accuracy=metric_info_artifact.train_accuracy,
                test_accuracy=metric_info_artifact.test_accuracy,
                model_accuracy=metric_info_artifact.model_accuracy,
                train_f1_score=metric_info_artifact.train_f1_score,
                test_f1_score=metric_info_artifact.test_f1_score,
                train_precision_score=metric_info_artifact.train_precision_score,
                test_precision_score=metric_info_artifact.test_precision_score,
                train_recall_score=metric_info_artifact.train_recall_score,
                test_recall_score=metric_info_artifact.test_recall_score
            )
            logging.info(f"Model Trainer Artifact: [{model_trainer_artifact}].")
            return model_trainer_artifact
        except Exception as e:
            raise BankingException(e, sys) from e
    
    def __del__(self) -> None:
        try:
            logging.info(f"{'='*60}Model Trainer Log Completed.{'='*60}\n\n")
        except Exception as e:
            raise BankingException(e, sys) from e