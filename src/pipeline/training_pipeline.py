import os
import sys
from src.logger.logger import logging
from src.exception.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluationConfig

class TrainingPipeline:
    def data_ingestion(self):
        try:
            obj=DataIngestion()
            train_data_path,test_data_path=obj.initiate_data_ingestion()
            return train_data_path,test_data_path
        except Exception as e:
            raise CustomException(e,sys)
    def data_trans(self,train_data_path,test_data_path):
        try:
            data_transformation=DataTransformation()
            train_arr,test_arr=data_transformation.initialize_data_transformation(train_data_path,test_data_path)
            return train_arr,test_arr
        except Exception as e:
            raise CustomException(e,sys)
    def model_train(self,train_arr,test_arr):
        try:
            model_trainer_obj=ModelTrainer()
            model_trainer_obj.initiate_model_training(train_arr,test_arr)
        except Exception as e:
            raise CustomException(e,sys)
    def model_eval(self,train_arr,test_arr):
        try:    
            model_eval_obj =ModelEvaluationConfig()
            model_eval_obj.initiate_model_evaluation(train_arr,test_arr)
        except Exception as e:
            raise CustomException(e,sys)