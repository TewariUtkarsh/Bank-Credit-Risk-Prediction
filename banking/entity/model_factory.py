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
        try:
            module = importlib.import_module(module_name)
            object_reference = getattr(module, class_name)
            
            return object_reference
        except Exception as e:
            raise BankingException(e, sys) from e

    @staticmethod
    def update_object_property(model_object:object, params:dict) -> object:
        try:
            if not isinstance(params,dict):
                raise Exception(f"Parameters passed should be of type dict. [{type(params)}]")

            for key, value in params.items():
                setattr(model_object, key, value)

            return model_object
        except Exception as e:
            raise BankingException(e, sys) from e

    @staticmethod
    def evaluate_classification_model(
            model_list:List[object], X_train:np.array, y_train:np.array, X_test:np.array, y_test:np.array, base_accuracy:float
        ) -> MetricsInfoArtifact:
        try:
            metrics_info_artifact = None
            model_index_no = 0
            for model in model_list:
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
                
                if base_accuracy<=model_accuracy and train_test_diff<= OVERFIT_CRITERION:

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

                model_index_no += 1

            return metrics_info_artifact
        except Exception as e:
            raise BankingException(e, sys) from e

    def get_best_model_grid_searched_model_list(self, grid_searched_model_list:List[GridSearchedModel], base_accuracy:float) -> BestModel:
        try:
            best_model = None
            for grid_searched_model in grid_searched_model_list:
                if base_accuracy <= grid_searched_model.best_score:
                    best_model = grid_searched_model
                    base_accuracy = grid_searched_model.best_score

            if not best_model:
                raise Exception(f"All the existing models have accuracy less than the base accuracy: [{base_accuracy}]")
            return best_model
        except Exception as e:
            raise BankingException(e, sys) from e

    def execute_grid_search(self, X_train:np.array, y_train:np.array, initialized_model:InitializedModel) -> GridSearchedModel:
        try:
            estimator = initialized_model.model_object
            param_grid = initialized_model.grid_search_params

            # no base_acc, change entire code

            grid_search_object_reference = ModelFactory.get_class_attribute_from_module(class_name=self.grid_search_class, module_name=self.grid_search_module)

            grid_search_object = grid_search_object_reference(estimator=estimator, param_grid=param_grid)
            
            grid_search_object = ModelFactory.update_object_property(model_object=grid_search_object, params=self.grid_search_params)

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
        try:
            initialized_model_list = self.get_initialized_model_list()
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