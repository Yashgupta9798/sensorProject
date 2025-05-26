

import sys
from typing import Generator, List, Tuple
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    artifact_folder = os.path.join(artifact_folder)
    trained_model_path = os.path.join(artifact_folder, "model.pkl")
    expected_accracy = 0.45
    model_config_file_path = os.path.join('config', 'model.yml')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
        self.utils = MainUtils()
        
        self.models = {
            'XGClassifier' : XGBClassifier(),
            'GradientBoostingClassifier' : GradientBoostingClassifier(),
            'SVC' : SVC(),
            'RandomForestClassifier' : RandomForestClassifier()
        }

    def evaluate_models(self, X, y, models):
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # spliting the data

            report = {}    # for storing model and accuracy

            for i in range(len(list(models))):  # looping over every model that we have definded models in prev attributes

                model = list(models.values())[i]

                model.fit(X_train, y_train)   # Train model

                y_train_pred = model.predict(X_train)   # X_train for training and it's ans will be in y_train
                y_test_pred = model.predict(X_test)

                train_model_score = accuracy_score(y_train, y_train_pred)   # accuracy of the train data
                test_model_score = accuracy_score(y_test, y_test_pred);

                report[list(model.keys())[i]] = test_model_score

            return report   # Return a dict with model name as key and test accuracy as value
        except Exception as e:
            raise CustomException(e, sys)
        
    def get_best_model(self,
                    x_train: np.array,
                    y_train: np.array,
                    x_test: np.array,
                    y_test: np.array):
        try:

            model_report: dict = self.evaluate_models(
                x_train = x_train, 
                y_train = y_train,
                x_test = x_test,
                y_test = y_test,
                models = self.models
            )  # calling 1st method

            print(model_report)

            best_model_score = max(sorted(model_report.values()))


            ## To get the best model from the dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model_object = self.models[best_model_name]

            return best_model_name, best_model_object, best_model_score   # returns name of best model, socre and model obj
        except Exception as e:
            raise CustomException(e, sys)
        
        
        
    def finetune_best_model(self,
                            best_model_object:object,
                            best_model_name,
                            X_train,
                            y_train)->object:
        try:
            model_param_grid = self.utils.read_yaml_file(self.model_trainer_config.model_config_file_path)["model selection"]["model"][best_model_name]["search_param_grid"]  # reading the yaml file for hyper paramerter of every model

            grid_search = GridSearchCV(
                best_model_object, param_grid=model_param_grid, cv = 5, n_jobs=-1, verbose=1)  # applying grid search CV
            
            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_

            print("best params are:", best_params)

            finetuned_model = best_model_object.set_params(**best_params)  # updating

            return finetuned_model   # return fine tuned model
        except Exception as e:
            raise CustomException(e, sys)
        

        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info(f"splitting training and testing input and target feature")

            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )  # we have created in data_transformatin component

            logging.info(f"Extracting model config file path")
            model_report: dict = self.evaluate_models(X = x_train, y = y_train, models = self.models)  # calling the 1st function

            ## To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get the best model name from the dict
            best_model_name = list(model_report.keys())[
                list(model_report.values().index(best_model_score))
            ]

            best_model = self.models[best_model_name]   # intead we should call the 2nd method



            best_model = self.finetune_best_model(
                best_model_name = best_model_name,
                best_model_object = best_model,
                X_train = x_train,
                y_train = y_train
            )   # Calling 3rd method

            best_model.fit(x_train, y_train)  # again trainig the model on training set
            y_pred = best_model.predict(x_test)  # predicting with the best model
            best_model_score = accuracy_score(y_test, y_pred)  # finding the accuracy score using best model

            print(f"best model name {best_model_name} and score: {best_model_score}")


            if best_model_score < 0.5:  # checking with the threshold
                raise Exception("no best model is found with accuracy greater than threshold 0.6")
            
            logging.info(f"Best found model on both training and testing dataset")

            logging.info(f"saving model at path: {self.model_trainer_config.trained_model_path}")

            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)  # path to save the model


            self.utils.save_object(
                file_path = self.model_trainer_config.trained_model_path,
                obj = best_model
            )

            return self.model_trainer_config.trained_model_path
        except Exception as e:
            raise CustomException(e, sys)

            
        


