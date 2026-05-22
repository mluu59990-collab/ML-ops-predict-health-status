import pandas as pd
import numpy as np
import os
import sys
import subprocess
from src.logger.logger import logging
from src.exception.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pymongo import MongoClient


@dataclass
class DataIngestionConfig:
    raw_data_path:   str = os.path.join("artifact", "raw.csv")
    train_data_path: str = os.path.join("artifact", "train.csv")
    test_data_path:  str = os.path.join("artifact", "test.csv")


def _dvc_track_and_push(file_paths: list, version_tag: str = None):
    """
    Track các file bằng DVC rồi push lên MinIO.
    Gọi sau khi đã lưu CSV xong.
    """
    try:
        # dvc add từng file
        for fp in file_paths:
            subprocess.run(["dvc", "add", fp], check=True)
            logging.info(f"DVC tracked: {fp}")

        # push lên MinIO remote
        subprocess.run(["dvc", "push"], check=True)
        logging.info("DVC push lên MinIO thành công")

        # git commit .dvc files để lưu version
        dvc_files = [f"{fp}.dvc" for fp in file_paths]
        gitignore  = [os.path.join(os.path.dirname(fp), ".gitignore") for fp in file_paths]
        subprocess.run(["git", "add"] + dvc_files + gitignore, check=True)

        tag_msg = version_tag or "data-update"
        subprocess.run(
            ["git", "commit", "-m", f"[DVC] track data: {tag_msg}"],
            check=True
        )

        if version_tag:
            subprocess.run(["git", "tag", version_tag], check=True)
            logging.info(f"Git tag tạo: {version_tag}")

        print(f"✅ DVC: data đã được version và push lên MinIO (tag: {tag_msg})")

    except subprocess.CalledProcessError as e:
        # Không raise — DVC lỗi không nên dừng toàn bộ pipeline
        logging.warning(f"DVC track/push thất bại (bỏ qua): {e}")
        print(f"⚠️  DVC warning: {e} — pipeline vẫn tiếp tục")


class DataIngestion:
    def __init__(self):
        self.ingestion_data = DataIngestionConfig()

    def _load_from_mongodb(self) -> pd.DataFrame:
        """Load raw data từ MongoDB."""
        try:
            mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
            client    = MongoClient(mongo_uri)
            db         = client["fitness_db"]
            collection = db["raw_data"]

            data = pd.DataFrame(list(collection.find({}, {"_id": 0})))
            client.close()

            if data.empty:
                raise ValueError("Không tìm thấy data trong MongoDB. Hãy chạy script upload trước!")

            logging.info(f"Đọc được {len(data)} records từ MongoDB")
            return data

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self, dvc_version_tag: str = None):
        """
        Parameters
        ----------
        dvc_version_tag : str, optional
            Nếu truyền vào (vd: "data-v2") thì DVC sẽ tạo git tag này.
            Nếu None thì vẫn track nhưng không tạo tag.
        """
        logging.info("Bắt đầu quá trình chèn dữ liệu")
        try:
            data = self._load_from_mongodb()
            logging.info("Đọc file dữ liệu thô từ MongoDB")

            os.makedirs(os.path.dirname(self.ingestion_data.raw_data_path), exist_ok=True)
            data.to_csv(self.ingestion_data.raw_data_path, index=False)
            logging.info("Lưu raw.csv")

            train_data, test_data = train_test_split(
                data,
                test_size=0.2,
                stratify=data["is_fit"],
                random_state=42
            )
            logging.info(f"Train: {len(train_data)} | Test: {len(test_data)}")

            train_data.to_csv(self.ingestion_data.train_data_path, index=False)
            test_data.to_csv(self.ingestion_data.test_data_path,   index=False)
            logging.info("Lưu train.csv và test.csv")

            # ── Track & push lên MinIO qua DVC ───────────────────────────────
            _dvc_track_and_push(
                file_paths=[
                    self.ingestion_data.raw_data_path,
                    self.ingestion_data.train_data_path,
                    self.ingestion_data.test_data_path,
                ],
                version_tag=dvc_version_tag,
            )

            return (
                self.ingestion_data.train_data_path,
                self.ingestion_data.test_data_path,
            )

        except Exception as e:
            logging.info("Lỗi ở DataIngestion")
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion(dvc_version_tag="data-v1")