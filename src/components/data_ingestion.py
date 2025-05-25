#! In this we have 2 classes

# 1) Data Ingestion Config class --> where data is stroed info
# 2) Data Ingestion Class --> 3 methods --> handling data ingestion process and this class also contains instance of prev class i.e data_ingestion_config and MianUtils instance that is created alreay
# Method-1 --> export_collection_as_dataframe() Methods --> connect to mongoDB, Fetch data, remove id and cols and replace missing value
# Method-2 --> export_data_into_feature_store_file_path() Method --> log info, create dir, fetch data, save as CSV
# Method-3 --> initiate_data_ingestion() Method --> log info, call export mehtod, return path

import sys # to talk to the python interpreter and read some constants
import os # to read files
import numpy as np
import pandas as pd
from pymongo import MongoClient
from zipfile import path
from dataclasses import dataclass   # with the help of this dataclass we need not to write constructor (eg. __init__) and this is used to stroe the value only

# what we have created
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils

@dataclass
class DataIngestionConfig:
    artifact_folder: str = os.path.join(artifact_folder)

class DataIngestion:   # since we have to instance here so we are not using dataclass
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()    # creating a instance of that class
        self.utils = MainUtils()

    def export_collection_as_dataframe(self, collection_name, db_name):  #convert dataset into dataframe
        try:
            mongo_client = MongoClient(MONGO_DB_URL) # for modular coding we use the constants from constant folder file not from upload_data
            collection = mongo_client[db_name][collection_name]  # fetching the data 
            df = pd.DataFrame(list(collection.find())) #converting to dataframe
            if "_id" in df.columns.to_list():
                df = df.drop(columns= ['_id'], axis=1)  # if we have id column then we we will drop the col

            df.replace({"na": np.nan}, inplace=True)  #replacing the missing value using numpy

            return df
        except Exception as e:
            raise CustomException(e, sys)
        
    def export_data_into_feature_store_file_path(self)-> pd.DataFrame:
        try:
            logging.info(f"Exporting a data from mongodb")
            raw_file_path = self.data_ingestion_config.artifact_folder

            os.makedirs(raw_file_path, exist_ok=True)  # if the dir is alreay there then don't throw error

            sensor_data = self.export_collection_as_dataframe(
                collection_name= MONGO_COLLECTION_NAME,
                db_name= MONGO_DATABASE_NAME
            ) # Calling first method and stroing the result (i.e dataframe) in sensor_data --> # it will export the collection as a dataframe

            logging.info(f"saving exported data into feature store file path : {raw_file_path}")

            feature_store_file_path = os.path.join(raw_file_path, 'wafer_fault.csv')

            sensor_data.to_csv(feature_store_file_path, index=False) #converting the data frame into csv and having no index appended

            return feature_store_file_path
        
        except Exception as e:
            raise CustomException(e, sys)
        
        def initiate_data_ingestion(self) -> Path:
            logging.info("Entered initiated_data_ingestion method of data_integration class")

            try:
                feature_store_file_path = self.export_data_into_feature_store_file_path()   # Calling the second method

                logging.info("got the data from mongodb")
                logging.info("exited initiated_data_ingestion method of data_ingestion_class")

                return feature_store_file_path
            except Exception as e:
                raise CustomException(e, sys) from e