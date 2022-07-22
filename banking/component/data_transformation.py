from banking.logger import logging
from banking.exception import BankingException
from banking.entity.config_entity import DataTransformationConfig
from banking.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from banking.utils.util import load_df_from_csv, read_yaml_data, save_numpy_array_to_file, save_model_object_to_file
from banking.constant import *
import os, sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE



class FeatureEngineering(BaseEstimator, TransformerMixin):

    def __init__(self, schema_file_info:dict) -> None:
        try:
            self.schema_file_info = schema_file_info
        except Exception as e:
            raise BankingException(e, sys) from e

    def fit(self, X, y=None):
        try:
            return self
        except Exception as e:
            raise BankingException(e, sys) from e

    def transform(self, X, y=None):
        try:
            schema_file_column = self.schema_file_info[SCHEMA_COLUMN_NAME_KEY]
            new_x = pd.DataFrame(X.copy())
            for column in schema_file_column:
                new_column_name = schema_file_column[column][0]
                new_column_dtype = schema_file_column[column][1]
                
                new_x.rename(columns={column: new_column_name}, inplace=True)
                new_x.astype(new_column_dtype)
            return new_x
        except Exception as e:
            raise BankingException(e, sys) from e
"""
x = data.drop(columns=[target_column])
y = data[target_column]
smote = SMOTE(random_state=)
smote.fit_resample(x, y)
# update name
# update col dtype
# apply smote sampling with config: at training model
# return transformed data
"""       
        




class DataTransformation:

    def __init__(self, 
        DataTransformationConfig: DataTransformationConfig,
        DataIngestionArtifact: DataIngestionArtifact,
        DataValidationArtifact: DataValidationArtifact
        ) -> None:
        """
        This class is responsible for initiating the Data Transformation phase of the Pipeline.
        Parameters
        ----------
        DataTransformationConfig : namedtuple
            Named tuple containing the details about Data Transformation Configuration.
        DataIngestionArtifact : namedtuple
            Named tuple containing the details about Data Ingestion Artifacts.
        DataValidationArtifact : namedtuple
            Named tuple containing the details about Data Validation Artifact.
        
        Attributes
        ----------
        data_transformation_config : namedtuple
            Named tuple containing the details about Data Transformation Configuration.
        data_ingestion_artifact : namedtuple
            Named tuple containing the details about Data Ingestion Artifact.
        data_validation_artifact : namedtuple
            Named tuple containing the details about Data Validation Artifact.
        schema_file_info : dict
            Dictionary containing the information about the schema file.
        """   
        try:
            self.data_transformation_config = DataTransformationConfig
            self.data_ingestion_artifact = DataIngestionArtifact
            self.data_validation_artifact = DataValidationArtifact
            self.schema_file_info = read_yaml_data(self.data_validation_artifact.schema_file_path)
        except Exception as e:
            raise BankingException(e, sys) from e


    def get_preprocessed_model_object(self) -> ColumnTransformer:
        """
        This function is responsible for creating the preprocessing model object for the 
        numerical and categorical pipelines.
        Returns:
        --------
        preprocessed_model_object : ColumnTransformer
            ColumnTransformer model object containing both numerical and categorical pipelines.
        """ 
        try:
            numerical_columns = self.schema_file_info[SCHEMA_CONTINUOUS_COLUMN_KEY]
            target_column = self.schema_file_info[SCHEMA_OLD_TARGET_COLUMN_KEY]
            categorical_columns = [column for column in self.schema_file_info[SCHEMA_DOMAIN_VALUE_KEY] if column!=target_column]

            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('feature_engineering', FeatureEngineering(
                            schema_file_info=self.schema_file_info
                        )
                    ),
                    ('scaler', StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('feature_engineering', FeatureEngineering(
                            schema_file_info=self.schema_file_info
                        )
                    ),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            preprocessed_model_object = ColumnTransformer(
                transformers=[
                    ('numerical_pipeline', numerical_pipeline, numerical_columns),
                    ('categorical_pipeline', categorical_pipeline, categorical_columns)
                ]
            )

            return preprocessed_model_object
        except Exception as e:
            raise BankingException(e, sys) from e    


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        This function is responsible for initiating the data transformation phase in the
        pipeline.
        Returns:
        --------
        DataTransformationArtifact : namedtuple
            Named tuple consisting the artifact related details of Data Transformation Phase.
        """
        try:
            train_data_file_path = self.data_ingestion_artifact.train_data_file_path
            test_data_file_path = self.data_ingestion_artifact.test_data_file_path

            train_data_file_name = os.path.basename(train_data_file_path).replace('.csv', 'npz')
            test_data_file_name = os.path.basename(test_data_file_path).replace('.csv', 'npz')

            train_df = load_df_from_csv(file_path= train_data_file_path)
            train_df.rename(columns={self.schema_file_info[SCHEMA_OLD_TARGET_COLUMN_KEY]: self.schema_file_info[SCHEMA_TARGET_COLUMN_KEY]}, inplace=True)

            test_df = load_df_from_csv(file_path= test_data_file_path)
            test_df.rename(columns={self.schema_file_info[SCHEMA_OLD_TARGET_COLUMN_KEY]: self.schema_file_info[SCHEMA_TARGET_COLUMN_KEY]}, inplace=True)

            label_column = self.schema_file_info[SCHEMA_TARGET_COLUMN_KEY]

            train_df_features = train_df.drop([label_column], axis=1)
            train_df_label = train_df[label_column]     # np.array()

            test_df_features = test_df.drop([label_column], axis=1)
            test_df_label = test_df[label_column]   # np.array()

            preprocessed_model_object = self.get_preprocessed_model_object()
            
            transformed_train_df_features = preprocessed_model_object.fit_transform(train_df_features)  # np.array()
            transformed_test_df_features = preprocessed_model_object.transform(test_df_features)    # np.array()

            transformed_train_df =  np.c_[transformed_train_df_features, train_df_label]
            transformed_test_df = np.c_[transformed_test_df_features, test_df_label]

            # Saving Transformed DF and model
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir
            os.makedirs(transformed_train_dir, exist_ok=True)
            os.makedirs(transformed_test_dir, exist_ok=True)

            transformed_train_data_file_path = os.path.join(
                    transformed_train_dir,
                    train_data_file_name
            )
            transformed_test_data_file_path = os.path.join(
                    transformed_test_dir,
                    test_data_file_name
            )
            save_numpy_array_to_file(data=transformed_train_df, file_path=transformed_train_data_file_path)
            save_numpy_array_to_file(data=transformed_test_df, file_path=transformed_test_data_file_path)

            preprocessed_model_object_file_path = self.data_transformation_config.preprocessed_model_object_file_path
            save_model_object_to_file(model=preprocessed_model_object, file_path=preprocessed_model_object_file_path)

            is_transformed = True
            message = f"Data Transformation Phase Completed"

            data_transformation_artifact = DataTransformationArtifact(
                is_transformed=is_transformed,
                message=message,
                transformed_train_data_file_path=transformed_train_data_file_path,
                transformed_test_data_file_path=transformed_test_data_file_path,
                preprocessed_model_object_file_path=preprocessed_model_object_file_path
            )
            
            return data_transformation_artifact
        except Exception as e:
            raise BankingException(e, sys) from e


    def __del__(self) -> None:
        logging.info(f"{'='*60}Data Transformation Log Completed.{'='*60}")
