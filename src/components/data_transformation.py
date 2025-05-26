# in this we have 2 classes --> 
# 1) DataTransformationConfig classs --> defines all the required path --> artifact_dir, transformed_train_file_path, transformed_test_file_path, transformed_object_file_path
# 2) DataTransformation Class --> responsibel for performing dasta transformation-->it have 3 methods
# Method-1 --> get_data() Method --> reads raw data from csv file(made in data_ingestion file and stored it in feature_store_file_path) and return it as a panda DF and also rename good/bad to target_col
# Method-2 --> get_data_transformer_object() Method --> fill the missing value with 0 using simple imputer and scaling is also done here
# Method-3 --> initiate_data_transformation() Method --> calls 1st method, feature and target split(X, y), train-test split, call the 2nd funciton for preprocessing step like imputation and scaling, save the preprocessor pipeline obhject to a file, save transformed data


import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

# what we have made earlier
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils

@dataclass
class DataTransformationConfig:
    artifact_dir = os.path.join(artifact_folder)   # such that we can read the output of all the pre classes
    transformed_train_file_path = os.path.join(artifact_dir, 'train.npy')
    transformed_test_file_path = os.path.join(artifact_dir, 'test.npy')
    transformed_object_file_path = os.path.join(artifact_dir, 'preprocessor.pkl')  # pkl since we are strong object(preprocessed object)

class DataTransformation:
    def __init__(self, feature_store_file_path):
        self.feature_store_file_path = feature_store_file_path

        self.data_transformation_config = DataTransformationConfig()

        self.utils = MainUtils()   # as we have created a function in utils to save the object as we will get obj after preprocessing

        @staticmethod
        def get_data(feature_store_file_path: str) -> pd.DataFrame:
            try:
                data = pd.read_csv(feature_store_file_path)  # in data_ingestion class we stored csv in this path
                data.rename(columns= {"Good/Bad" : TARGET_COLUMN}, inplace=True)
                return data  # reads csv and return dataframe
            except Exception as e:
                raise CustomException(e, sys)
            
        def get_data_transformer_object(self):
            try:
                imputer_step = ('imputer', SimpleImputer(strategy='constant', fill_value = 0))   #imputaion
                scaller_step = ('scaler', RobustScaler())   # scaling
                
                preprocessor = Pipeline(
                    steps=[
                        imputer_step,
                        scaller_step
                    ]
                )

                return preprocessor   # return a preprocessed obj
            except Exception as e:
                raise CustomException(e, sys)
            
        def initiate_data_transformation(self):
            logging.info("Entered initiate data transformation method of data transformation class")

            try:
                dataframe = self.get_data(feature_store_file_path = self.feature_store_file_path) # calling 1st method

                X = dataframe.drop(columns = TARGET_COLUMN)
                y = np.where(dataframe[TARGET_COLUMN] == -1, 0, 1)   # removing -ve numbers if we have -1 then replace it with 0

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

                preprocessor = self.get_data_transformer_object()  # calling 2nd mehtod

                X_train_scaled = preprocessor.fit_transform(X_train)
                X_test_scaled = preprocessor.transform(X_test)

                preprocessor_path = self.data_transformation_config.transformed_object_file_path  #from 1st class
                os.mkdir(os.path.dirname(preprocessor_path), exist_ok = True)

                self.utils.save_object(file_path = preprocessor_path, obj = preprocessor)   # using the utils function

                train_arr = np.c[X_train_scaled,np.array(y_train)]
                test_arr = np.c[X_test_scaled, np.array(y_test)]

                return (train_arr, test_arr, preprocessor_path)   # return training and test array along with preprocessor file path
            except Exception as e:
                raise CustomException(e, sys)


