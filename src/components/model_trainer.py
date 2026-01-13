import pandas as pd
import numpy as np
from src.logger.logger import logging
from src.exception.exception import CustomException
import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path
from src.utils.utils import save_object,load_object,evaluate_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact','model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Chia du lieu")
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            model ={
                'SVM':SVC(),
                'logistic_regression':LogisticRegression(),
                'random_forest':RandomForestClassifier()
            }
            model_report:dict = evaluate_model(x_train,y_train,x_test,y_test,model)
            print(model_report)
            print('\n============================================================')
            logging.info(f'Model Report:{model_report}')
            #laays best model
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = model[best_model_name]
            print(f'best model found, model name:{best_model_name},Recall score :{best_model_score}')
            print('\n=============================================================')
            logging.info(f'Best model found, model name:{best_model_name},Recall score:{best_model_score}')
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )
        except Exception as e:
            logging.info('loi')
            raise CustomException(e,sys) 


