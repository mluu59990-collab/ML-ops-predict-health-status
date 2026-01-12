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
    pass
class ModelTrainer:
    def __init__(self):
        pass
    def initiate_model_trainning(self):
        try:
            pass
        except Exception as e:
            logging.info()
            raise CustomException(e,sys) 


