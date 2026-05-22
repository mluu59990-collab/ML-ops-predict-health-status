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
from src.components.model_evaluation import ModelEvaluation

# ── Cấu hình MinIO tập trung 1 chỗ ──────────────────────────────────────────
MINIO_ENDPOINT   = "http://127.0.0.1:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MLFLOW_BUCKET    = "s3://mlflow-artifacts/"
MLFLOW_PORT      = 5000


def _set_minio_env():
    """Set tất cả env vars cần thiết cho MinIO/S3 — gọi trước mọi thứ."""
    os.environ["MLFLOW_S3_ENDPOINT_URL"]  = MINIO_ENDPOINT
    os.environ["AWS_ACCESS_KEY_ID"]       = MINIO_ACCESS_KEY
    os.environ["AWS_SECRET_ACCESS_KEY"]   = MINIO_SECRET_KEY
    os.environ["AWS_DEFAULT_REGION"]      = "us-east-1"
    os.environ["AWS_S3_ADDRESSING_STYLE"] = "path"


def start_mlflow_server():
    """Khởi động MLflow server nếu chưa chạy, kèm đầy đủ env vars MinIO."""

    # Kiểm tra server đã chạy chưa
    try:
        r = requests.get(f"http://127.0.0.1:{MLFLOW_PORT}/health", timeout=2)
        if r.status_code == 200:
            print(f" MLflow server đã chạy tại http://127.0.0.1:{MLFLOW_PORT}")
            return
    except requests.exceptions.ConnectionError:
        pass

    print("Đang khởi động MLflow server...")

    # Truyền đầy đủ env vars cho subprocess
    env = os.environ.copy()  # đã có MinIO env vì _set_minio_env() gọi trước

    proc = subprocess.Popen(
        [
            "mlflow", "server",
            "--backend-store-uri", "sqlite:///mlflow.db",
            "--default-artifact-root", MLFLOW_BUCKET,
            "--host", "127.0.0.1",
            "--port", str(MLFLOW_PORT),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Chờ tối đa 20 giây
    for i in range(20):
        time.sleep(1)
        try:
            r = requests.get(f"http://127.0.0.1:{MLFLOW_PORT}/health", timeout=1)
            if r.status_code == 200:
                print(f"MLflow server sẵn sàng sau {i + 1}s — http://127.0.0.1:{MLFLOW_PORT}")
                return
        except requests.exceptions.ConnectionError:
            print(f"   Đang chờ... ({i + 1}s)")

    # Kiểm tra process có bị crash không
    if proc.poll() is not None:
        raise RuntimeError(
            " MLflow server bị crash khi khởi động. "
            "Kiểm tra: (1) MinIO đang chạy, (2) bucket 'mlflow-artifacts' đã tạo, "
            "(3) port 5000 không bị chiếm."
        )
    print("  MLflow server có thể chưa sẵn sàng — tiếp tục...")


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
            trained_models = model_trainer_obj.initiate_model_training(train_arr)
            return trained_models
        except Exception as e:
            raise CustomException(e, sys)

    def model_eval(self, trained_models, test_arr):
        try:
            model_eval_obj = ModelEvaluation()
            model_eval_obj.initiate_model_evaluation(trained_models, test_arr)
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # ── BƯỚC 0: Set env vars MinIO TRƯỚC TIÊN — trước mọi import khác ──────
    _set_minio_env()

    # ── BƯỚC 1: Đảm bảo MLflow server đang chạy ─────────────────────────────
    start_mlflow_server()

    # ── BƯỚC 2: Data ingestion ───────────────────────────────────────────────
    print("\n Ingestion......")
    import datetime
    version_tag = f"data-{datetime.date.today()}"   # vd: data-2026-05-22
    ingestion   = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion(dvc_version_tag=version_tag)

    # ── BƯỚC 3: Transform ────────────────────────────────────────────────────
    print("\n Transformation......")
    transformation = DataTransformation()
    train_arr, test_arr = transformation.initialize_data_transformation(
        train_path, test_path
    )

    # ── BƯỚC 4: Train ────────────────────────────────────────────────────────
    print("\n Training......")
    trainer = ModelTrainer()
    trained_models = trainer.initiate_model_training(train_arr)

    # ── BƯỚC 5: Evaluate + register lên MLflow/MinIO ─────────────────────────
    print("\n Evaluation......")
    evaluator = ModelEvaluation()
    evaluator.initiate_model_evaluation(trained_models, test_arr)

    print(f"\n Hoàn tất! Vào http://127.0.0.1:{MLFLOW_PORT} để xem kết quả.")