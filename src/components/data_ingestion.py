import pandas as pd
import numpy as np
from src.logger.logger import logging
from src.exception.exception import CustomException
import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path
@dataclass
class DataIngestionConfig: 
    raw_data_path:str = os.path.join("artifact","raw.csv")
    train_data_path:str = os.path.join("artifact","train.csv")
    test_data_path:str = os.path.join("artifact","test.csv")
class DataIngestion:
    def __init__(self):
        self.ingestion_data = DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("bat dau qua trinh chen du lieu")
        try:
            data = pd.read_csv("https://raw.githubusercontent.com/hoangminh125HY/data_ML_isfit/refs/heads/master/data/fitness_dataset.csv")
            logging.info("Doc file du lieu tho")
            os.makedirs(os.path.dirname(self.ingestion_data.raw_data_path), exist_ok=True)
            data.to_csv(self.ingestion_data.raw_data_path,index=False)
            logging.info("Luu data thanh file csv")
            logging.info("Bat dau qua trinh chia du lieu train test split")
            train_data,test_data = train_test_split(data,test_size=0.2,stratify= data["is_fit"],random_state=42)
            logging.info("Da hoan thanh chia du lieu train, test")
            train_data.to_csv(self.ingestion_data.train_data_path,index = False)
            test_data.to_csv(self.ingestion_data.test_data_path,index = False)
            logging.info("Chen du lieu da hoan thanh")
            return(
                self.ingestion_data.train_data_path,
                self.ingestion_data.test_data_path
            )
        except Exception as e:
            logging.info("Loi o day")
            raise CustomException(e,sys) 

if __name__ =="__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()