import os
import sys
from sklearn.metrics import precision_score,recall_score,f1_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import pickle
from src.logger.logger import logging
from src.exception.exception import CustomException
from src.utils.utils import load_object

class ModelEvaluationConfig:
    def __init__(self):
        logging.info("evaluation started")
    def eval_metrics(self,actual,pred):
        precision = precision_score(actual,pred)
        recall = recall_score(actual,pred)
        f1 = f1_score(actual,pred)
        logging.info("metrics captured")
        return precision,recall,f1
    def initiate_model_evaluation(self,train_array,test_array):
        try:
            x_test,y_test = (test_array[:,:-1],test_array[:,-1])
            model_path  = os.path.join("artifact","model.pkl")
            model = load_object(model_path)
            mlflow.set_registry_uri("http://127.0.0.1:5000")
            logging.info("Model register")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            print(tracking_url_type_store)
            with mlflow.start_run():
                prediction =  model.predict(x_test)
                (pre,recall,f1) =self.eval_metrics(y_test,prediction)
                mlflow.log_metric("precision",pre)
                mlflow.log_metric("recall",recall)
                mlflow.log_metric("f1",f1)
                if tracking_url_type_store !="file":
                    mlflow.sklearn.log_model(model,"model",registered_model_name= "FitnessModel")
                else:
                    mlflow.sklearn.log_model(model,"model")




        except Exception as e:
             raise CustomException(e,sys) 


