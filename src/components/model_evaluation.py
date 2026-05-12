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
    def eval_metrics(self, actual, pred):
        precision = precision_score(actual, pred, pos_label=0)
        recall    = recall_score(actual, pred, pos_label=0)
        f1        = f1_score(actual, pred, pos_label=0)
        return precision, recall, f1
    def initiate_model_evaluation(self,test_array):
        try:
            x_test,y_test = (test_array[:,:-1],test_array[:,-1])
            model_path  = os.path.join("artifact","model.pkl")
            model = load_object(model_path)
            mlflow.set_registry_uri("http://127.0.0.1:5000")
            logging.info("Model register")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            print(tracking_url_type_store)
            with mlflow.start_run():
                prediction = model.predict(x_test)
                (pre, recall, f1) = self.eval_metrics(y_test, prediction)
                
                # Thêm 3 dòng này
                print(f"Precision : {pre:.4f}")
                print(f"Recall    : {recall:.4f}")
                print(f"F1 Score  : {f1:.4f}")
                
                mlflow.log_metric("precision", pre)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1", f1)
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name="FitnessModel")
                else:
                    mlflow.sklearn.log_model(model, "model")




        except Exception as e:
             raise CustomException(e,sys) 

if __name__ == "__main__":
    import numpy as np
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    # Load lại test data
    ingestion = DataIngestion()
    train_path, val_path, test_path = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()
    train_arr, val_arr, test_arr = transformation.initialize_data_transformation(
        train_path, test_path, val_path
    )

    # Chạy evaluation trên test set
    evaluator = ModelEvaluationConfig()
    evaluator.initiate_model_evaluation(test_arr)
