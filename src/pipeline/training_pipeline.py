import os
import sys
import time
import subprocess
import requests
from src.logger.logger import logging
from src.exception.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluationConfig  # da sua tu ModelEvaluationConfig


def start_mlflow_server():
    """Tu dong kiem tra va khoi dong MLflow server neu chua chay."""
    try:
        response = requests.get("http://127.0.0.1:5000/health", timeout=2)
        if response.status_code == 200:
            print("MLflow server da chay san roi tai http://127.0.0.1:5000")
            return
    except requests.exceptions.ConnectionError:
        pass

    print("Dang khoi dong MLflow server...")
    subprocess.Popen(
        [
            "mlflow", "server",
            "--backend-store-uri", "sqlite:///mlflow.db",
            "--default-artifact-root", "./mlruns",
            "--host", "127.0.0.1",
            "--port", "5000",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Cho server khoi dong xong (toi da 15 giay)
    for i in range(15):
        time.sleep(1)
        try:
            requests.get("http://127.0.0.1:5000/health", timeout=1)
            print(f"MLflow server san sang sau {i + 1} giay — http://127.0.0.1:5000")
            return
        except requests.exceptions.ConnectionError:
            print(f"  Dang cho server khoi dong... ({i + 1}s)")

    raise RuntimeError(
        "Khong the khoi dong MLflow server sau 15 giay. "
        "Hay thu chay thu cong: mlflow server --backend-store-uri sqlite:///mlflow.db "
        "--default-artifact-root ./mlruns --host 127.0.0.1 --port 5000"
    )


class TrainingPipeline:
    def data_ingestion(self):
        try:
            obj = DataIngestion()
            train_data_path, test_data_path = obj.initiate_data_ingestion()
            return train_data_path, test_data_path
        except Exception as e:
            raise CustomException(e, sys)

    def data_trans(self, train_data_path, test_data_path):
        try:
            data_transformation = DataTransformation()
            train_arr, test_arr = data_transformation.initialize_data_transformation(
                train_data_path, test_data_path
            )
            return train_arr, test_arr
        except Exception as e:
            raise CustomException(e, sys)

    def model_train(self, train_arr):
        try:
            model_trainer_obj = ModelTrainer()
            model_trainer_obj.initiate_model_training(train_arr)
        except Exception as e:
            raise CustomException(e, sys)

    def model_eval(self, test_arr):
        try:
            model_eval_obj = ModelEvaluationConfig()  # da sua tu ModelEvaluationConfig
            model_eval_obj.initiate_model_evaluation(test_arr)
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Buoc 0: dam bao MLflow server dang chay
    start_mlflow_server()

    # Buoc 1: lay data
    print("\nIngestion......")
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # Buoc 2: transform
    print("\nTransformation......")
    transformation = DataTransformation()
    train_arr, test_arr = transformation.initialize_data_transformation(
        train_path, test_path
    )

    # Buoc 3: train
    print("\nTraining......")
    trainer = ModelTrainer()
    trainer.initiate_model_training(train_arr)

    # Buoc 4: evaluate + register model
    print("\nEvaluation......")
    evaluator = ModelEvaluationConfig()  # da sua tu ModelEvaluationConfig
    evaluator.initiate_model_evaluation(test_arr)

    print("\nHoan tat! Vao http://127.0.0.1:5000 de xem ket qua.")