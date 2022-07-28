from banking.logger import logging
from banking.exception import BankingException
from banking.utils.util import read_yaml_data
import os, sys
import numpy as np
from collections import namedtuple
from typing import List
import importlib
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score



# Constants for model config
GRID_SEARCH_KEY = "grid_search"
CLASS_KEY = "class"
MODULE_KEY = "module"
PARAMS_KEY = "params"
MODEL_SELECTION_KEY = "model_selection"
GRID_SEARCH_PARAM_KEY = "grid_search_params"
OVERFIT_CRITERION = 1


# Required namedtuple objects
InitializedModel = namedtuple(
    "InitializedModel",
    ["model_serial_number", "model_name", "model_object", "grid_search_params"]
)

GridSearchedModel = namedtuple(
    "GridSearchedModel",
    ["model_serial_number", "model_object", "best_model", "best_params", "best_score"]
)

BestModel = namedtuple(
    "BestModel",
    ["model_serial_number", "model_object", "best_model", "best_params", "best_score"]
)

MetricsInfoArtifact = namedtuple(
    "MetricsInfoArtifact",
    ["model_index_number", "model_object", "model_name", "train_accuracy", "test_accuracy", "model_accuracy",
     "train_f1_score", "test_f1_score", "train_precision_score", "test_precision_score", "train_recall_score", "test_recall_score"]
)


class ModelFactory:

    def __init__(self, model_config_file_path:str) -> None:
        """
        This class is a is responsible for training, cross-validating and evaluating models
        for our Training Pipeline.
        Parameters
        ----------
        model_config_file_path : str
            Path for the file containing the confiurational details for building various models.
        
        Attributes
        ----------
        model_config_file_info : dict
            Dict containing the info about the model config file.
        grid_search_class : str
            Class name for the Grid Search Cross-validation.
        grid_search_module : str
            Module name for the Grid Search Cross-validation.
        grid_search_param : dict
            Dict containing Params details for the Grid Search Cross-validation.
        model_selection_info : dict
            Dict containing the details of different models to be selected.
        initialized_model_list : list
            List containing the initialized models.
        grid_searched_model_list : list
            List containing the Grid Searched models.
        """
        try:
            self.model_config_file_info = dict(read_yaml_data(file_path=model_config_file_path))
            self.grid_search_class = self.model_config_file_info[GRID_SEARCH_KEY][CLASS_KEY]
            self.grid_search_module = self.model_config_file_info[GRID_SEARCH_KEY][MODULE_KEY]
            self.grid_search_params = self.model_config_file_info[GRID_SEARCH_KEY][PARAMS_KEY]
            self.model_selection_info = dict(self.model_config_file_info[MODEL_SELECTION_KEY])
            
            self.initialized_model_list = []
            self.grid_searched_model_list = []

        except Exception as e:
            raise BankingException(e, sys) from e

    @staticmethod
    def get_class_attribute_from_module(class_name:str, module_name:str) -> object:
        """
            This function is responsible for creating the attribute for our Model.
            Returns:
            --------
            object_reference : object
                Model Attribute(Reference object)
        """
        try:
            module = importlib.import_module(module_name)
            object_reference = getattr(module, class_name)
            logging.info(f"Executing command: from {module} import {class_name}")
            return object_reference
        except Exception as e:
            raise BankingException(e, sys) from e

    @staticmethod
    def update_object_property(model_object:object, params:dict) -> object:
        """
            This function is responsible for updating the attributes/params/property of our Model.
            Returns:
            --------
            model_object : object
                Updated Model Attribute(Reference object)
        """
        try:
        
            if not isinstance(params,dict):
                raise Exception(f"Parameters passed should be of type dict. [{type(params)}]")

            for key, value in params.items():
                setattr(model_object, key, value)
                logging.info(f"Executing:$ {str(model_object)}.{key}={value}")

            return model_object
        except Exception as e:
            raise BankingException(e, sys) from e

    @staticmethod
    def evaluate_classification_model(
            model_list:List[object], X_train:np.array, y_train:np.array, X_test:np.array, y_test:np.array, base_accuracy:float
        ) -> MetricsInfoArtifact:
        """
        This function is responsible for evaluating the classification models from the list of models
        based on certain parameters and conditions passed.
        Returns:
        --------
        MetricsInfoArtifact : namedtuple
            Namedtuple containing the artifact details of metrics about the model.
        """
        try:
            metrics_info_artifact = None
            model_index_no = 0
            for model in model_list:
                logging.info(f"{'>>'*30}Evaluating model: [{type(model).__name__}] {'<<'*30} ")
                model_name = str(model)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)

                
                train_test_diff = np.abs(train_accuracy - test_accuracy)
                
                model_accuracy = (2 * (train_accuracy * test_accuracy))/(train_accuracy + test_accuracy)
                
                train_f1_score = fbeta_score(y_train, y_train_pred, beta=1)
                test_f1_score = fbeta_score(y_test, y_test_pred, beta=1)

                train_precision_score = precision_score(y_train, y_train_pred)
                test_precision_score = precision_score(y_test, y_test_pred)

                train_recall_score = recall_score(y_train, y_train_pred)
                test_recall_score = recall_score(y_test, y_test_pred)

                logging.info(f"{'>>'*30} Score {'<<'*30}")
                logging.info(f"Train Score\t\t Test Score\t\t Average Score")
                logging.info(f"{train_accuracy}\t\t {test_accuracy}\t\t{model_accuracy}")

                logging.info(f"{'>>'*30} Statistics {'<<'*30}")
                logging.info(f"Diff test train accuracy: [{train_test_diff}].") 
                logging.info(f"Train f1 score: [{train_f1_score}].")
                logging.info(f"Test f1 score: [{test_f1_score}].")
                logging.info(f"Train precision score: [{train_precision_score}].")
                logging.info(f"Test precision score: [{test_precision_score}].")
                logging.info(f"Train recall score: [{train_recall_score}].")
                logging.info(f"Test recall score: [{test_recall_score}].")


                
                if 0.6<=model_accuracy and train_test_diff<= OVERFIT_CRITERION:

                    base_accuracy = model_accuracy 
                    metrics_info_artifact = MetricsInfoArtifact(
                                                model_index_number=model_index_no,
                                                model_object=model,
                                                model_name=model_name,
                                                train_accuracy=train_accuracy,
                                                test_accuracy=test_accuracy,
                                                model_accuracy=model_accuracy,
                                                train_f1_score=train_f1_score,
                                                test_f1_score=test_f1_score,
                                                train_precision_score=train_precision_score,
                                                test_precision_score=test_precision_score,
                                                train_recall_score=train_recall_score,
                                                test_recall_score=test_recall_score
                                            )       
                    logging.info(f"Acceptable model found {metrics_info_artifact}. ")
                model_index_no += 1
        
            if metrics_info_artifact is None:
                logging.info(f"No model found with higher accuracy than base accuracy")
            return metrics_info_artifact
        except Exception as e:
            raise BankingException(e, sys) from e


    def get_best_model_grid_searched_model_list(self, grid_searched_model_list:List[GridSearchedModel], base_accuracy:float) -> BestModel:
        """
        This function is responsible for finding the best model out of the grid searched model list.
        Returns:
        --------
        BestModel : namedtuple
            Namedtuple containing the details about the best model.
        """
        try:
            best_model = None
            for grid_searched_model in grid_searched_model_list:
                if base_accuracy <= grid_searched_model.best_score:
                    best_model = grid_searched_model
                    base_accuracy = grid_searched_model.best_score

            if not best_model:
                raise Exception(f"All the existing models have accuracy less than the base accuracy: [{base_accuracy}]")
            logging.info(f"Best model: {best_model}")
            return best_model
        except Exception as e:
            raise BankingException(e, sys) from e

    def execute_grid_search(self, X_train:np.array, y_train:np.array, initialized_model:InitializedModel) -> GridSearchedModel:
        """
        This function is responsible for executing grid search cv in order to find the best model for estimator passed.
        Returns:
        --------
        GridSearchedModel : namedtuple
            Namedtuple containing the details about the model acquired after performing grid search cv.
        """
        try:
            estimator = initialized_model.model_object
            param_grid = initialized_model.grid_search_params

            # no base_acc, change entire code

            grid_search_object_reference = ModelFactory.get_class_attribute_from_module(class_name=self.grid_search_class, module_name=self.grid_search_module)

            grid_search_object = grid_search_object_reference(estimator=estimator, param_grid=param_grid)
            
            grid_search_object = ModelFactory.update_object_property(model_object=grid_search_object, params=self.grid_search_params)

            message = f'{">>"* 30} f"Training {type(initialized_model.model_object).__name__} Started." {"<<"*30}'
            logging.info(message)
            
            grid_search_object.fit(X_train, y_train)
            
            grid_searched_model = GridSearchedModel(
                model_serial_number=initialized_model.model_serial_number,
                model_object=initialized_model.model_object,
                best_model=grid_search_object.best_estimator_,
                best_params=grid_search_object.best_params_,
                best_score=grid_search_object.best_score_
            )
            return grid_searched_model
            
        except Exception as e:
            raise BankingException(e, sys) from e

    def initiate_grid_search_for_initiallized_model(self, X_train:np.array, y_train:np.array, initialized_model:InitializedModel) -> GridSearchedModel:
        """
        This function is responsible for initiating grid search cv for the estimator passed.
        Returns:
        --------
        GridSearchedModel : namedtuple
            Namedtuple containing the details about the model acquired after performing grid search cv.
        """
        try:
            grid_searched_model = self.execute_grid_search(
                                            X_train=X_train,
                                            y_train=y_train,
                                            initialized_model=initialized_model,     
                                        )
            return grid_searched_model
        except Exception as e:
            raise BankingException(e, sys) from e

    def get_grid_searched_model_list(self, X_train:np.array, y_train:np.array, initialized_model_list: List[InitializedModel]) -> List[GridSearchedModel]:
        """
        This function is responsible for generating a list of all 
        grid searched models.
        Returns:
        --------
        grid_searched_model_list : List[GridSearchedModel]
            List of Namedtuples containing the details about the model acquired after performing grid search cv.
        """
        try: 
            for initialized_model in initialized_model_list:
                grid_searched_model = self.initiate_grid_search_for_initiallized_model(
                                                X_train=X_train,
                                                y_train=y_train,
                                                initialized_model=initialized_model,
                                                )
                self.grid_searched_model_list.append(grid_searched_model)

            return self.grid_searched_model_list
        except Exception as e:
            raise BankingException(e, sys) from e


    def get_initialized_model_list(self) -> List[InitializedModel]:
        """
        This function is responsible for generating a list of all 
        initialized models.
        Returns:
        --------
        initialized_model_list : List[InitializedModel]
            List of Namedtuples containing the details about all the initialized models.
        """
        try:
            selected_models = dict(self.model_selection_info)

            for model_info in selected_models:
                class_name = selected_models[model_info][CLASS_KEY]
                module_name = selected_models[model_info][MODULE_KEY]
                
                model_object_reference = ModelFactory.get_class_attribute_from_module(class_name=class_name, module_name=module_name)

                model_object = model_object_reference()

                if PARAMS_KEY in model_info:
                    params = dict(selected_models[model_info][PARAMS_KEY])
                    model_object = ModelFactory.update_object_property(model_object=model_object, params=params)            

                model_name = f"{module_name}.{class_name}()"
                grid_search_params = dict(selected_models[model_info][GRID_SEARCH_PARAM_KEY])

                initialized_model = InitializedModel(
                    model_serial_number=model_info,
                    model_name=model_name,
                    model_object=model_object,
                    grid_search_params=grid_search_params
                )

                self.initialized_model_list.append(initialized_model)

            return self.initialized_model_list
        except Exception as e:
            raise BankingException(e, sys) from e

    def get_best_model(self, X_train:np.array, y_train:np.array, base_accuracy:float) -> BestModel:
        """
        This function is responsible for finding the best model out of the available models after performing
        various tests.
        Returns:
        --------
        best_model : BestModel
            Namedtuple containing the details about the best model acquired.
        """
        try:
            logging.info("Started Initializing model from config file")
            initialized_model_list = self.get_initialized_model_list()
            logging.info(f"Initialized model: {initialized_model_list}")
            
            # print(type(initialized_model_list), type(initialized_model_list[0]))
            grid_searched_model_list = self.get_grid_searched_model_list(X_train=X_train , y_train=y_train , initialized_model_list=initialized_model_list)

            best_model = self.get_best_model_grid_searched_model_list(grid_searched_model_list=grid_searched_model_list, base_accuracy=base_accuracy)                                                                

            return best_model

        except Exception as e:
            raise BankingException(e, sys) from e



"""
best_model = get_best_model -> 1. get_initialized_model_list() -> get_object_from_class()@staticmethod, obj(), update_obj_param()@staticmethod
                               2. get_grid_searched_model_list() -> init_grid_search_for_init_model() -> execute_grid_search()
                               3. get_best_model_from_grid_searched_model_list() @staticmethod

model_list = [model.best_model for model in mf.grid_searched_model_list]

metric_info = evaluate_regression_model(model_list, train, test, base) -> 


InitializedModel = m_s_no, m_name, m_obj, grid_param 
GridSearchedModel = m_s_no, m_name, m_obj, best_model, best_param, best_score
BestModel = m_s_no, m_name, m_obj, ""
MetricInfoArtifact = index_no, m_name, m_obj, train-test-rmse, train-test-acc, model_acc, 
"""