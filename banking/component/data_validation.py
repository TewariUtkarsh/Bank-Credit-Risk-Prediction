from re import S
from banking.exception import BankingException
from banking.logger import logging
import os, sys
from banking.entity.config_entity import DataValidationConfig
from banking.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from banking.utils.util import read_yaml_data
from banking.constant import *
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
import pandas as pd
from typing import List
import json



class DataValidation:

    def __init__(self, 
        DataValidationConfig: DataValidationConfig,
        DataIngestionArtifact: DataIngestionArtifact
        ) -> None:
        """
        This class is responsible for initiating the Data Validation phase of the Pipeline.
        Parameters
        ----------
        DataValidationConfig : namedtuple
            Named tuple containing the details about Data Validation Configuration.
        DataIngestionArtifact : namedtuple
            Named tuple containing the details about Data Ingestion Artifacts.
        
        Attributes
        ----------
        data_validation_config : namedtuple
            Named tuple containing the details about Data Validation Configuration.
        data_ingestion_artifact : namedtuple
            Current time stamp in string format.
        """        
        try:
            logging.info(f"{'='*60}Data Validation Log Started.{'='*60}")
            self.data_validation_config = DataValidationConfig
            self.data_ingestion_artifact = DataIngestionArtifact
        except Exception as e:
            raise BankingException(e, sys) from e


    def is_train_test_file_exists(self) -> bool:
        """
        This function is responsible for validating if the train and test file exists.

        Returns
        -------
        is_exists : bool
            Boolean value which represents the existence of training and testing file.
        """   
        try:
            logging.info(f"Checking if training and tseting file exists.")
            train_data_file_path = self.data_ingestion_artifact.train_data_file_path
            test_data_file_path = self.data_ingestion_artifact.test_data_file_path

            is_train_file_exists = os.path.exists(train_data_file_path)
            is_test_file_exists = os.path.exists(test_data_file_path)

            is_validated = is_train_file_exists and is_test_file_exists
            
            logging.info(f"Is training and testing file exists? -> [{is_validated}]")

            if is_validated:
                return is_validated
            else:
                raise Exception(f"Training File:[{train_data_file_path}]\nand\nTesting File: [{test_data_file_path}]\ndoes not exists")
        except Exception as e:
            raise BankingException(e, sys) from e


    def get_train_test_dataframe(self) -> List:
        """
        This function is responsible for extracting the Training and Testing Dataframes
        from the given file paths.
        Returns
        -------
        list : List
            List consisting of Training and Testing Dataframes
        """ 
        try: 
            logging.info("Loading DataFrames from the Training and Testing Files.")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_data_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_data_file_path)
            return [train_df, test_df]
        except Exception as e:
            raise BankingException(e, sys) from e


    def validate_schema_structure(self) -> bool:
        """
        This function is responsible for validating if the train and test file exists.

        Returns
        -------
        is_exists : bool
            Boolean value which represents the existence of training and testing file.
        """ 
        try:
            logging.info(f"Validating the Schema Structure of the Training and Testing Files.")
            is_train_df_validated = False
            is_test_df_validated = False

            schema_file_info = read_yaml_data(file_path=self.data_validation_config.schema_file_path)
            schema_columns = schema_file_info[SCHEMA_COLUMN_NAME_KEY]
            
            schema_number_of_columns = len(schema_columns)
            schema_column_names = list(schema_columns.keys())
            schema_column_names.sort()
            schema_domain_values = schema_file_info[SCHEMA_DOMAIN_VALUE_KEY]

            train_df, test_df = self.get_train_test_dataframe()
            
            train_df_col_names = list(train_df.columns)
            train_df_col_names.sort()
            test_df_col_names = list(test_df.columns)
            test_df_col_names.sort()
            train_df_num_of_cols = len(train_df_col_names)
            test_df_num_of_cols = len(test_df_col_names)

            if train_df_num_of_cols==schema_number_of_columns:
                if train_df_col_names==schema_column_names:
                    for train_df_column in schema_domain_values:
                        schema_column_domain_values = list(schema_domain_values[train_df_column])
                        train_df_column_domain_values = list(train_df[train_df_column].unique())
                        if schema_column_domain_values.sort() == train_df_column_domain_values.sort():
                            is_train_df_validated = True
                        else:
                            raise Exception(f"Invalid Domain Value for Column: [{train_df_column}] in Training File: [{self.data_ingestion_artifact.train_data_file_path}]")
                else:
                    raise Exception(f"Invalid Column Name in Training File: [{self.data_ingestion_artifact.train_data_file_path}]")
            else:
                raise Exception(f"Invalid Number of Columns: [{train_df_num_of_cols}] in Training File: [{self.data_ingestion_artifact.train_data_file_path}]")


            if test_df_num_of_cols==schema_number_of_columns:
                if test_df_col_names==schema_column_names:
                    for test_df_column in schema_domain_values:
                        schema_column_domain_values = list(schema_domain_values[test_df_column])
                        test_df_column_domain_values = list(test_df[test_df_column].unique())
                        if schema_column_domain_values.sort() == test_df_column_domain_values.sort():
                            is_test_df_validated = True
                        else:
                            raise Exception(f"Invalid Domain Value for Column: [{test_df_column}] in Testing File: [{self.data_ingestion_artifact.test_data_file_path}]")
                else:
                    raise Exception(f"Invalid Column Name in Testing File: [{self.data_ingestion_artifact.test_data_file_path}]")
            else:
                raise Exception(f"Invalid Number of Columns:[{test_df_num_of_cols}] in Testing File: [{self.data_ingestion_artifact.test_data_file_path}]")


            is_validated = is_train_df_validated and is_test_df_validated

            if is_validated:
                logging.info(f"Schema Structure Validation for Training and Testing Files Successful.")
                return is_validated
            else:
                raise Exception(f"Invalid Schema Structure for Training and Testing Files.")

        ## col datatype and name transform in data transformation    
        except Exception as e:
            raise BankingException(e, sys) from e
        

    def generate_and_save_data_drift_report(self) -> List:
        try:
            logging.info(f"Generating Data Drift Profile Report.")
            train_df, test_df = self.get_train_test_dataframe()
            
            profile = Profile(sections=[DataDriftProfileSection()])
            
            profile.calculate(train_df, test_df)
            
            data_drift_report_json = json.loads(profile.json())

            report_file_path = self.data_validation_config.report_file_path
            os.makedirs(os.path.dirname(report_file_path), exist_ok=True)
            
            with open(report_file_path, 'w') as file_obj:
                json.dump(data_drift_report_json, file_obj, indent=6)
            
            logging.info(f"Data Drift Profile Report Generated and Saved Successfully at location: [{report_file_path}]")
            return [report_file_path, data_drift_report_json]
        except Exception as e:
            raise BankingException(e, sys) from e
        

    def save_data_drift_report_page(self) -> str:
        try:
            logging.info(f"Generating Data Drift Report Page.")

            train_df, test_df = self.get_train_test_dataframe()
            
            report_page_file_path = self.data_validation_config.report_page_file_path
            os.makedirs(os.path.dirname(report_page_file_path), exist_ok=True)

            dashboard = Dashboard(tabs=[DataDriftTab()])

            dashboard.calculate(train_df, test_df)
            
            dashboard.save(report_page_file_path)

            logging.info(f"Data Drift Report Page Generated and Saved Successfully at location: [{report_page_file_path}]")
            return report_page_file_path
        except Exception as e:
            raise BankingException(e, sys) from e


    def is_data_drift_present(self) -> List:
        try:
            logging.info("Checking if Data Drift present in the Dataset.")
            is_present = False
            report_file_path, data_drift_report_json = self.generate_and_save_data_drift_report()
            report_page_file_path = self.save_data_drift_report_page()
            
            is_present = data_drift_report_json[DATA_DRIFT_KEY][DATA_DRIFT_DATA_KEY][DATA_DRIFT_METRICS_KEY][DATA_DRIFT_DATASET_DRIFT_KEY]
            
            is_validated = not is_present

            logging.info(f"Is Data Drift Present in Dataset? -> {is_present}")
            if is_validated:
                return is_validated, report_file_path, report_page_file_path
            else:
                raise Exception(f"Significant Data Drift found in the Dataset.")
        except Exception as e:
            raise BankingException(e, sys) from e
    

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Initiating the Data Validation Phase.")
            is_validated = self.is_train_test_file_exists()
            is_validated = self.validate_schema_structure()
            is_validated, report_file_path, report_page_file_path = self.is_data_drift_present()
            
            logging.info(f"Data Validation Phase Completed Successfully.")
            message = f"Data Validation Phase Completed."
            
            data_validation_artifact = DataValidationArtifact(
                is_validated=is_validated,
                message=message,
                schema_file_path=self.data_validation_config.schema_file_path,
                report_file_path=report_file_path,
                report_page_file_path=report_page_file_path
            )
            logging.info(f"Data Validation Artifact: \n{data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise BankingException(e, sys) from e


    def __del__(self) -> None:
        logging.info(f"{'='*60}Data Validation Log Completed.{'='*60}")

