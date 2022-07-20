from banking.constant import DATA_INGESTION_INGESTED_DATA_DIR_KEY, DATA_INGESTION_INGESTED_TEST_DIR_KEY, DATA_INGESTION_INGESTED_TRAIN_DIR_KEY
from banking.entity.config_entity import DataIngestionConfig
from banking.entity.artifact_entity import DataIngestionArtifact
from banking.exception import BankingException
from banking.logger import logging
from banking.utils.util import save_df_to_csv
import os, sys
import pandas as pd
from typing import List
from zipfile import ZipFile
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit

class DataIngestion:

    def __init__(self,
        data_ingestion_config:DataIngestionConfig,
        ) -> None:
        """
        This class is a Component and is responsible for initiating the 
        Data Ingestion phase of the Pipeline.
        Parameters
        ----------
        data_ingestion_config : namedtuple
            Named tuple for initiating Data Ingestion.
        
        Attributes
        ----------
        data_ingestion_config : dict
            Named tuple for initiating Data Ingestion.
        """
        try:
            logging.info(f"{'='*20}Data Ingestion Log Started.{'='*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise BankingException(e, sys) from e


    def download_banking_data(self) -> str:
        """
        This function is responsible for downloading the zip file
        from the given source url to the desired location.
        Returns:
        --------
        zip_file_path : str
            File path for the downloaded zip file.
        """
        try:
            dataset_download_url = self.data_ingestion_config.dataset_download_url

            zip_download_dir = self.data_ingestion_config.zip_download_dir
            os.makedirs(zip_download_dir, exist_ok=True)

            banking_file_name = os.path.basename(dataset_download_url)

            zip_file_path = os.path.join(
                zip_download_dir,
                banking_file_name
            )
            logging.info(f"Downloading File from: [{dataset_download_url}] at: [{zip_file_path}]")
            urllib.request.urlretrieve(dataset_download_url, zip_file_path)
            logging.info(f"File: [{zip_file_path}] Downloaded Successfully.")
            return zip_file_path
        except Exception as e:
            raise BankingException(e, sys) from e

    def extract_zip_file(self, zip_file_path:str) -> str:
        """
        This function is responsible for extracting the zip file
        from the given source path to the desired path.
        Parameters
        ----------
        zip_file_path : str
            File path of the downloaded zip file.
        Returns:
        --------
        extracted_raw_file_path : str
            File path for the extracted zip file.
        """
        try:
            extracted_data_dir = self.data_ingestion_config.extracted_data_dir
            os.makedirs(extracted_data_dir, exist_ok=True)

            logging.info(f"Extracting File: [{zip_file_path}] to: [{extracted_data_dir}]")
            with ZipFile(zip_file_path, 'r') as zip_file_obj:
                zip_file_obj.extractall(extracted_data_dir)
            logging.info(f"Extraction of File: [{zip_file_path}] Completed Successfully.")

            # As there are multiple files extracted so extracting the name of the required
            # data file to get the file path
            extracted_raw_file_name = os.path.basename(zip_file_path).split('.')[0]
            
            for file in os.listdir(extracted_data_dir):
                if extracted_raw_file_name==file.split('.')[0]:
                    extracted_raw_file_path = os.path.join(
                        extracted_data_dir,
                        file
                    )
                    logging.info(f"Extracted Data File Generated: [{extracted_raw_file_path}]")
                    return extracted_raw_file_path
        except Exception as e:
            raise BankingException(e, sys) from e

    def get_raw_data_from_extracted_file(self, extracted_raw_file_path:str) -> str:
        """
        This function is responsible for converting the content of .asc file format 
        to .csv file format from the given source path to the desired path.
        Parameters:
        -----------
        extracted_raw_file_path : str
            File path of the extracted raw file.
        Returns:
        --------
        raw_data_file_path : str
            Final .csv format file path.
        """
        try: 
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            os.makedirs(raw_data_dir, exist_ok=True)

            with open(extracted_raw_file_path, 'r') as raw_file_obj:
                columns = raw_file_obj.readline()[:-1].split(' ')
                data = []
                for line in raw_file_obj.readlines():
                    data.append(line[:-1].split(' '))

            df = pd.DataFrame(data=data, columns=columns)
            
            raw_data_file_name = os.path.basename(extracted_raw_file_path).split('.')[0]

            raw_data_file_path = os.path.join(
                raw_data_dir,
                raw_data_file_name
            )
            raw_data_file_path = raw_data_file_path + '.csv'
            save_df_to_csv(df, raw_data_file_path)
            
            logging.info(f"Raw Data File Generated: [{raw_data_file_path}]")
            return raw_data_file_path

        except Exception as e:
            raise BankingException(e, sys) from e
            
    def split_train_test_data(self, raw_data_file_path:str) -> List[str]:
        """
        This function is responsible for spliting the raw data into
        train and test data files with preserving the distribution 
        of the data across the kredit columns' value count and save files 
        to the desired path.
        Parameters:
        -----------
        raw_data_file_path : str
            File path of the extracted raw file.
        Returns:
        --------
        list : list
            List consisting of train and test data file paths
        """
        try:
            ingested_train_dir = self.data_ingestion_config.ingested_train_dir
            os.makedirs(ingested_train_dir, exist_ok=True)

            ingested_test_dir = self.data_ingestion_config.ingested_test_dir
            os.makedirs(ingested_test_dir, exist_ok=True)

            ingested_data_file_name = os.path.basename(raw_data_file_path)

            train_data_file_path = os.path.join(
                ingested_train_dir,
                ingested_data_file_name
            ) 
            test_data_file_path = os.path.join(
                ingested_test_dir,
                ingested_data_file_name
            )

            logging.info(f"Reading Raw Banking Data File: [{raw_data_file_path}]")
            banking_dataframe = pd.read_csv(raw_data_file_path)

            stratified_train_dataframe = None
            stratified_test_dataframe = None
            
            """
            # We are performing Stratified Split as we want to 
            # maintain the distribution of our entire dataset
            # same for train and test dataset same as that of our
            # raw dataset based on the frequency of the credit column
            # which is a categorical column.
            
            # We are not performing pd.cut() to create categories as 
            # credit column is a discrete numerical column

            # We can perform Stratified sampling only on categorical/discrete cols
            """
            logging.info(f"Performing Stratified Split on Dataset")
            stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
            for train_idx, test_idx in stratified_split.split(banking_dataframe, banking_dataframe['kredit']):
                stratified_train_dataframe = banking_dataframe.loc[train_idx]
                stratified_test_dataframe = banking_dataframe.loc[test_idx]


            logging.info(f"Exporting train dataset to csv file: [{train_data_file_path}]")
            stratified_train_dataframe.to_csv(train_data_file_path, index=None)

            logging.info(f"Exporting test dataset to csv file: [{test_data_file_path}]")
            stratified_test_dataframe.to_csv(test_data_file_path, index=None)

            return [train_data_file_path, train_data_file_path]
        except Exception as e:
            # from e is used to route to the actual location where e was generated
            raise BankingException(e, sys) from e


    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        This function is responsible for initiating the data ingestion phase in the
        pipeline.
        Returns:
        --------
        DataIngestionArtifact : namedtuple
            Named tuple consisting the artifact related details of Data Ingestion Phase.
        """
        try:
            zip_file_path = self.download_banking_data()
            extracted_raw_file_path = self.extract_zip_file(zip_file_path=zip_file_path)
            raw_data_file_path = self.get_raw_data_from_extracted_file(extracted_raw_file_path=extracted_raw_file_path)
            train_data_file_path, test_data_file_path = self.split_train_test_data(raw_data_file_path=raw_data_file_path)
            
            is_ingested = True
            message = f"Data Ingestion Phase Completed."
            data_ingestion_artifact = DataIngestionArtifact(
                is_ingested=is_ingested,
                message=message,
                train_data_file_path=train_data_file_path,
                test_data_file_path=test_data_file_path
            )
            logging.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise BankingException(e, sys) from e


    def __del__(self) -> None:
        """
        __del__ is a destructor method which is called as soon as all references 
        of the object are deleted i.e when an object is garbage collected.
        """
        logging.info(f"{'='*20}Data Ingestion Log Completed.{'='*20}")

        